from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from lightewm.eval import evaluate_video_quality
from lightewm.runner.backend_result import read_backend_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_ASSET_ROOT = "/pfs-verdent/zhangyu/robot-trial"


@dataclass
class WeightSpec:
    name: str
    ckpt_path: str | None


def parse_weight_spec(text: str) -> WeightSpec:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"Invalid weight spec '{text}', expected NAME=CKPT_OR_BASE")
    name, ckpt_path = text.split("=", 1)
    name = name.strip()
    ckpt_path = ckpt_path.strip()
    if not name:
        raise argparse.ArgumentTypeError("Weight name cannot be empty")
    if ckpt_path.lower() in {"", "base", "none", "null"}:
        ckpt_path = None
    return WeightSpec(name=name, ckpt_path=ckpt_path)


def default_weight_specs(asset_root: str) -> list[WeightSpec]:
    root = Path(asset_root)
    return [
        WeightSpec("wan22_ti2v_5b_base", None),
        WeightSpec("wan22_5b_robot", str(root / "checkpoints/Wan2.2-5B-Robot/checkpoint.safetensors")),
        WeightSpec("wan22_5b_libero", str(root / "checkpoints/Wan2.2-5B-Libero/checkpoint.safetensors")),
    ]



def wan_base_model_paths(base_model_dir: str) -> list:
    base = Path(base_model_dir)
    return [
        [
            str(base / "diffusion_pytorch_model-00001-of-00003.safetensors"),
            str(base / "diffusion_pytorch_model-00002-of-00003.safetensors"),
            str(base / "diffusion_pytorch_model-00003-of-00003.safetensors"),
        ],
        str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
        str(base / "Wan2.2_VAE.pth"),
    ]


def wan_base_model_overrides(base_model_dir: str) -> list[str]:
    model_paths = wan_base_model_paths(base_model_dir)
    tokenizer_path = str(Path(base_model_dir) / "google/umt5-xxl")
    return [
        f"model.params.tokenizer_path={tokenizer_path}",
        f"model.params.model_paths={json.dumps(model_paths)}",
    ]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def derive_config_name(config_path: str) -> str:
    p = Path(config_path)
    parent = p.parent.name
    stem = p.stem
    return f"{parent}_{stem}" if parent not in {"", "."} else stem


def default_output_root() -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "outputs" / "video_quality_eval" / timestamp


def require_path(path: str, description: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def check_base_model_paths(args):
    if args.base_model_dir is not None:
        tokenizer_path = Path(args.base_model_dir) / "google/umt5-xxl"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Base Wan tokenizer path is missing: {tokenizer_path}")
        model_paths = wan_base_model_paths(args.base_model_dir)
    else:
        config = load_yaml(REPO_ROOT / args.config)
        model_paths = config.get("model", {}).get("params", {}).get("model_paths", [])
    flat_paths = []
    for item in model_paths:
        if isinstance(item, list):
            flat_paths.extend(item)
        else:
            flat_paths.append(item)
    missing = [path for path in flat_paths if isinstance(path, str) and not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Base Wan2.2 files required by the inference config are missing. "
            f"First missing paths: {missing[:5]}"
        )


def build_inference_command(args, weight: WeightSpec, run_id: str) -> list[str]:
    overrides = []
    if args.base_model_dir is not None:
        overrides.extend(wan_base_model_overrides(args.base_model_dir))
    overrides.extend(
        [
            f"dataset.params.base_path={args.dataset_base_path}",
            f"dataset.params.metadata_path={args.metadata_path}",
            f"dataset.params.max_samples={args.max_samples}",
            f"runner.params.infer_kwargs.num_inference_steps={args.infer_steps}",
        ]
    )
    command = [
        sys.executable,
        "run.py",
        "--config",
        args.config,
        "--overrides",
    ]
    command.extend(overrides)
    if args.height is not None:
        command.append(f"runner.params.infer_kwargs.height={args.height}")
    if args.width is not None:
        command.append(f"runner.params.infer_kwargs.width={args.width}")
    if args.num_frames is not None:
        command.append(f"runner.params.infer_kwargs.num_frames={args.num_frames}")
    for override in args.override:
        command.append(override)
    if weight.ckpt_path is not None:
        command.extend(["--ckpt", weight.ckpt_path])
    return command


def run_inference(args, weight: WeightSpec, generated_dir: Path):
    check_base_model_paths(args)
    if weight.ckpt_path is not None:
        require_path(weight.ckpt_path, f"Checkpoint for {weight.name}")

    env = os.environ.copy()
    env["LIGHTEWM_RUN_ID"] = generated_dir.name
    command = build_inference_command(args, weight, generated_dir.name)
    print(f"[Eval][{weight.name}] Launch inference: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def append_summary(summary_csv: Path, summary: dict):
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "weight_name",
        "num_pairs",
        "num_missing",
        "fvd",
        "ssim",
        "psnr",
        "lpips",
        "fvd_backend",
        "generated_dir",
        "backend",
        "artifact_type",
        "backend_manifest",
    ]
    exists = summary_csv.exists()
    row = {
        "weight_name": summary["weight_name"],
        "num_pairs": summary["num_pairs"],
        "num_missing": summary["num_missing"],
        "fvd": summary["metrics"].get("fvd"),
        "ssim": summary["metrics"].get("ssim"),
        "psnr": summary["metrics"].get("psnr"),
        "lpips": summary["metrics"].get("lpips"),
        "fvd_backend": summary.get("metric_config", {}).get("fvd_backend"),
        "generated_dir": summary["generated_dir"],
        "backend": summary.get("backend"),
        "artifact_type": summary.get("artifact_type"),
        "backend_manifest": summary.get("backend_manifest"),
    }
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LightEWM LIBERO inference for multiple downloadable checkpoints and compute video quality metrics."
    )
    parser.add_argument("--config", default="examples/LIBERO/infer_ti2v_5b.yaml")
    parser.add_argument("--asset-root", default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--base-model-dir", default=None)
    parser.add_argument("--dataset-base-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--weight", action="append", type=parse_weight_spec, default=[])
    parser.add_argument("--generated-dir", action="append", default=[], help="Metrics-only mapping NAME=PATH.")
    parser.add_argument(
        "--backend-manifest",
        action="append",
        default=[],
        help="Backend manifest path, or NAME=PATH. When provided without --weight, manifests define the evaluated runs.",
    )
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--infer-steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--metrics", default="fvd,ssim,psnr,lpips")
    parser.add_argument("--pair-num-frames", type=int, default=None)
    parser.add_argument("--pair-height", type=int, default=None)
    parser.add_argument("--pair-width", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--fvd-no-pretrained", action="store_true")
    parser.add_argument("--fvd-num-frames", type=int, default=16)
    parser.add_argument("--fvd-image-size", type=int, default=112)
    parser.add_argument("--allow-missing-pairs", action="store_true")
    return parser.parse_args()


def parse_generated_dirs(items: list[str]) -> dict[str, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid generated-dir mapping '{item}', expected NAME=PATH")
        name, path = item.split("=", 1)
        result[name] = Path(path)
    return result


def parse_backend_manifests(items: list[str]) -> dict[str, tuple[Path, object]]:
    result = {}
    for item in items:
        explicit_name = None
        manifest_text = item
        if "=" in item:
            explicit_name, manifest_text = item.split("=", 1)
            explicit_name = explicit_name.strip()
        manifest_path = Path(manifest_text)
        manifest = read_backend_manifest(manifest_path)
        name = explicit_name or manifest.extra.get("weight_name") or manifest.backend
        if not name:
            raise ValueError(f"Cannot derive run name from backend manifest: {manifest_path}")
        result[str(name)] = (manifest_path, manifest)
    return result


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)

    asset_root = Path(args.asset_root)
    if args.dataset_base_path is None:
        args.dataset_base_path = str(asset_root / "data/libero_i2v_train")
    if args.metadata_path is None:
        args.metadata_path = str(asset_root / "data/libero_i2v_train/metadata_dense_prompt.csv")
    if args.base_model_dir is None:
        args.base_model_dir = str(asset_root / "checkpoints/Wan2.2-TI2V-5B")

    backend_manifests = parse_backend_manifests(args.backend_manifest)

    require_path(args.config, "Inference config")
    if not backend_manifests:
        require_path(args.dataset_base_path, "Dataset base path")
        require_path(args.metadata_path, "Metadata path")

    weights = args.weight if args.weight else [
        WeightSpec(name, None) for name in backend_manifests
    ] if backend_manifests else default_weight_specs(args.asset_root)
    output_root = Path(args.output_root) if args.output_root else default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary.csv"
    generated_dir_overrides = parse_generated_dirs(args.generated_dir)

    config_name = derive_config_name(args.config)
    run_prefix = output_root.name
    metric_names = [item.strip().lower() for item in args.metrics.split(",") if item.strip()]
    pair_resize = None
    if args.pair_height is not None or args.pair_width is not None:
        if args.pair_height is None or args.pair_width is None:
            raise ValueError("--pair-height and --pair-width must be provided together")
        pair_resize = (args.pair_height, args.pair_width)

    all_summaries = []
    for weight in weights:
        manifest_entry = backend_manifests.get(weight.name)
        manifest_path = None
        manifest = None
        generated_dir = generated_dir_overrides.get(weight.name)
        metadata_path = args.metadata_path
        dataset_base_path = args.dataset_base_path
        video_key = "video"
        artifact_type = "video"
        backend_name = None

        if manifest_entry is not None:
            manifest_path, manifest = manifest_entry
            generated_dir = Path(manifest.generated_dir)
            metadata_path = manifest.metadata_path or metadata_path
            dataset_base_path = manifest.dataset_base_path or dataset_base_path
            video_key = str(manifest.extra.get("video_key") or video_key)
            artifact_type = manifest.artifact_type
            backend_name = manifest.backend
            if artifact_type != "video":
                raise ValueError(
                    f"Backend manifest for {weight.name} has artifact_type={artifact_type!r}; "
                    "this eval runner currently supports video artifacts only."
                )
        elif generated_dir is None:
            run_id = f"{run_prefix}_{weight.name}"
            generated_dir = REPO_ROOT / "logs" / config_name / run_id

        if not args.skip_inference and manifest_entry is None:
            run_inference(args, weight, generated_dir)
        require_path(str(generated_dir), f"Generated video directory for {weight.name}")
        require_path(metadata_path, f"Metadata path for {weight.name}")
        require_path(dataset_base_path, f"Dataset base path for {weight.name}")

        metric_output_dir = output_root / "metrics" / weight.name
        print(f"[Eval][{weight.name}] Computing metrics from {generated_dir}", flush=True)
        summary = evaluate_video_quality(
            metadata_path=metadata_path,
            dataset_base_path=dataset_base_path,
            generated_dir=str(generated_dir),
            weight_name=weight.name,
            output_dir=str(metric_output_dir),
            max_samples=args.max_samples,
            metrics=metric_names,
            pair_num_frames=args.pair_num_frames,
            pair_resize=pair_resize,
            device=args.device,
            lpips_net=args.lpips_net,
            metric_batch_size=args.metric_batch_size,
            fvd_pretrained=not args.fvd_no_pretrained,
            fvd_num_frames=args.fvd_num_frames,
            fvd_image_size=args.fvd_image_size,
            video_key=video_key,
            strict=not args.allow_missing_pairs,
        )
        summary["backend"] = backend_name
        summary["artifact_type"] = artifact_type
        summary["backend_manifest"] = str(manifest_path) if manifest_path is not None else None
        append_summary(summary_csv, summary)
        all_summaries.append(summary)
        print(
            "[Eval][{name}] DONE: FVD={fvd} SSIM={ssim} LPIPS={lpips}".format(
                name=weight.name,
                fvd=summary["metrics"].get("fvd"),
                ssim=summary["metrics"].get("ssim"),
                lpips=summary["metrics"].get("lpips"),
            ),
            flush=True,
        )

    with open(output_root / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"[Eval] Summary CSV: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
