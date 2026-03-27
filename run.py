import argparse
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime

import yaml

from lightewm.runner.base_runner_pipeline import BaseRunnerPipeline
from lightewm.utils.config import ConfigNode


def deep_merge(base: dict, update: dict):
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config_with_base(config_path: str) -> tuple[dict, list[Path]]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)

    base_files = cfg.pop("_base_", None)
    if base_files is None:
        return cfg, [config_path]

    if isinstance(base_files, str):
        base_files = [base_files]
    if not isinstance(base_files, list):
        raise ValueError(f"_base_ must be string or list: {config_path}")

    merged = {}
    source_paths = []
    for base_file in base_files:
        base_path = (config_path.parent / base_file).resolve()
        base_cfg, base_sources = load_config_with_base(str(base_path))
        merged = deep_merge(merged, base_cfg)
        source_paths.extend(base_sources)

    merged = deep_merge(merged, cfg)
    source_paths.append(config_path)

    dedup_paths = []
    seen = set()
    for path in source_paths:
        if str(path) in seen:
            continue
        seen.add(str(path))
        dedup_paths.append(path)
    return merged, dedup_paths


def set_dot_key(cfg: dict, key: str, value):
    parts = key.split(".")
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, value_text = item.split("=", 1)
        value = yaml.safe_load(value_text)
        set_dot_key(cfg, key, value)
    return cfg


def apply_ckpt_override(cfg: dict, ckpt_path: str) -> dict:
    if not ckpt_path.endswith(".safetensors"):
        raise ValueError(f"--ckpt must be a .safetensors file, got: {ckpt_path}")

    model_cfg = cfg.setdefault("model", {})
    model_params = model_cfg.setdefault("params", {})
    model_paths = model_params.get("model_paths")
    if model_paths is None:
        raise ValueError("Config has no model.params.model_paths to apply --ckpt override.")
    if not isinstance(model_paths, list) or len(model_paths) == 0:
        raise ValueError("model.params.model_paths must be a non-empty list for --ckpt override.")

    model_paths[0] = ckpt_path
    model_params["model_paths"] = model_paths
    return cfg


def derive_config_name(config_path: str):
    p = Path(config_path)
    parent = p.parent.name
    stem = p.stem
    return f"{parent}_{stem}" if parent not in {"", "."} else stem


def _get_launch_key():
    master_addr = os.environ.get("MASTER_ADDR", "")
    master_port = os.environ.get("MASTER_PORT", "")
    world_size = os.environ.get("WORLD_SIZE", "1")
    return f"{master_addr}:{master_port}:{world_size}"


def resolve_run_id(config_name: str):
    base_dir = Path("logs") / config_name
    base_dir.mkdir(parents=True, exist_ok=True)
    run_context_path = base_dir / ".run_context.json"
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launch_key = _get_launch_key()

    if world_size <= 1:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if rank == 0:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        payload = {"launch_key": launch_key, "run_id": run_id}
        tmp_path = run_context_path.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, run_context_path)
        return run_id

    start_time = time.time()
    while True:
        if run_context_path.exists():
            try:
                with open(run_context_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if payload.get("launch_key") == launch_key and payload.get("run_id"):
                    return str(payload["run_id"])
            except Exception:
                pass
        if time.time() - start_time > 600:
            raise TimeoutError(f"Timed out waiting for run context: {run_context_path}")
        time.sleep(0.2)


def apply_output_convention(cfg: dict, config_name: str, run_id: str):
    run_dir = Path("logs") / config_name / run_id
    task = cfg.get("task", "")
    runner = cfg.setdefault("runner", {})
    runner_params = runner.setdefault("params", {})
    if task == "train":
        runner_params["output_path"] = str(run_dir)
    elif task == "infer":
        runner_params["output_dir"] = str(run_dir)
    return run_dir


def snapshot_run_configs(run_dir: Path, source_paths: list[Path], merged_cfg: dict):
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    mapping_lines = []
    for idx, src in enumerate(source_paths):
        dst = configs_dir / f"{idx:02d}__{src.name}"
        shutil.copy2(src, dst)
        mapping_lines.append(f"{src} -> {dst.name}")

    with open(run_dir / "config_sources.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(mapping_lines) + "\n")
    with open(run_dir / "merged_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="LightEWM unified config runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--overrides", nargs="*", default=[], help="Dotlist overrides, e.g. runner.params.fps=12")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional .safetensors DiT checkpoint override for model.params.model_paths[0]")
    parser.add_argument("--print-config", action="store_true", help="Print merged config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without running")
    args = parser.parse_args()

    cfg, source_paths = load_config_with_base(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    if args.ckpt:
        cfg = apply_ckpt_override(cfg, args.ckpt)

    config_name = derive_config_name(args.config)
    run_id = resolve_run_id(config_name)
    run_dir = apply_output_convention(cfg, config_name, run_id)

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))

    if args.dry_run:
        print("[DryRun] Config validated.")
        return

    if int(os.environ.get("RANK", "0")) == 0:
        snapshot_run_configs(run_dir, source_paths, cfg)

    config = ConfigNode.from_dict(cfg)
    pipeline = BaseRunnerPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
