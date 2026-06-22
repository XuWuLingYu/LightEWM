from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lightewm.runner.backend_result import BackendRunResult
from lightewm.runner.runner_util.instantiation import instantiate_component_from_section

from .config_adapter import adapt_official_config


class CausalForcingRunner:
    """LightEWM adapter for the vendored Causal-Forcing backend."""

    DEFAULT_BACKEND_ROOT = Path("lightewm/vendor/causal_forcing")
    DEFAULT_CONFIG_PATH = DEFAULT_BACKEND_ROOT / "configs/ar_diffusion_tf_framewise_wan22_ti2v_5b_maze.yaml"

    def __init__(self, config):
        self.config = config
        self.repo_root = Path.cwd()

    def _full_config(self):
        return self.config.full_config

    def _runner_params(self) -> dict:
        params = self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)
        for key in ("class_path", "section_name", "full_config"):
            params.pop(key, None)
        return params

    def _task(self) -> str:
        return str(getattr(self._full_config(), "task", "train"))

    def _run_root(self) -> Path:
        params = self._runner_params()
        if self._task() == "infer":
            return Path(params.get("output_dir", "./outputs/causal_forcing"))
        return Path(params.get("output_path", "./logs/causal_forcing"))

    def _repo_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _relative_to_backend(self, path: str | Path, backend_root: Path) -> str:
        return os.path.relpath(
            self._repo_path(path).resolve(),
            start=self._repo_path(backend_root).resolve(),
        )

    def _relative_to(self, path: str | Path, base: str | Path) -> str:
        return os.path.relpath(
            self._repo_path(path).resolve(),
            start=self._repo_path(base).resolve(),
        )

    def _python_for_backend(self, backend_root: Path) -> str:
        executable = Path(sys.executable)
        if executable.is_absolute():
            return self._relative_to_backend(executable, backend_root)
        return str(executable)

    def _prepare_dataset(self, run_root: Path) -> Path:
        full_config = self._full_config()
        dataset, _ = instantiate_component_from_section(
            full_config.dataset,
            full_config,
            section_name="dataset",
        )
        if not hasattr(dataset, "write_jsonl"):
            raise TypeError("CausalForcingRunner dataset must provide write_jsonl(output_path).")
        if hasattr(dataset, "base_path") and hasattr(dataset, "jsonl_base_path"):
            dataset.jsonl_base_path = self._relative_to(dataset.base_path, run_root)
        output_path = run_root / "causal_forcing_dataset.jsonl"
        count = dataset.write_jsonl(str(output_path))
        print(f"[CausalForcing] Wrote {count} samples to {output_path}")
        return output_path

    def _prepare_config(self, run_root: Path, jsonl_path: Path, backend_root: Path) -> Path:
        params = self._runner_params()
        official_config_path = params.get("official_config_path", str(self.DEFAULT_CONFIG_PATH))
        output_path = run_root / "causal_forcing_config.yaml"
        output_overrides = dict(params.get("causal_config_overrides", {}))
        for checkpoint_key in ("generator_ckpt", "action_dit_ckpt", "action_dit_pretrained_path"):
            if checkpoint_key in output_overrides and output_overrides[checkpoint_key]:
                output_overrides[checkpoint_key] = self._relative_to_backend(
                    output_overrides[checkpoint_key],
                    backend_root,
                )
        adapt_official_config(
            official_config_path=official_config_path,
            output_config_path=str(output_path),
            data_path=self._relative_to_backend(jsonl_path, backend_root),
            model_root=self._relative_to_backend(params.get("model_root", "checkpoints"), backend_root),
            output_overrides=output_overrides,
            dot_overrides=params.get("causal_config_dot_overrides", {}),
        )
        print(f"[CausalForcing] Wrote adapted config to {output_path}")
        return output_path

    def _base_env(self, backend_root: Path) -> dict:
        params = self._runner_params()
        env = os.environ.copy()
        backend_abs = str(self._repo_path(backend_root).resolve())
        pythonpath = [backend_abs, "."]
        local_pythonpath = params.get("pythonpath", ["data/python-packages"])
        if isinstance(local_pythonpath, str):
            local_pythonpath = [local_pythonpath]
        for path in local_pythonpath:
            if self._repo_path(path).exists():
                pythonpath.append(self._relative_to_backend(path, backend_root))
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        return env

    def _launch_cmd(
        self,
        backend_root: Path,
        default_master_port: int,
        force_distributed: bool = False,
    ) -> list[str]:
        params = self._runner_params()
        launch_cmd = [self._python_for_backend(backend_root)]
        num_processes = int(params.get("num_processes", 1))
        if force_distributed or num_processes > 1:
            launch_cmd.extend([
                "-m",
                "torch.distributed.run",
                "--nproc_per_node",
                str(num_processes),
                "--master_port",
                str(params.get("master_port", default_master_port)),
            ])
        return launch_cmd

    def _run_train(self, backend_root: Path, config_path: Path, run_root: Path) -> int:
        params = self._runner_params()
        launch_cmd = self._launch_cmd(
            backend_root,
            default_master_port=29577,
            force_distributed=True,
        )
        cmd = launch_cmd + [
            "train.py",
            "--config_path",
            self._relative_to_backend(config_path, backend_root),
            "--logdir",
            self._relative_to_backend(run_root, backend_root),
            "--wandb-save-dir",
            self._relative_to_backend(run_root / "wandb", backend_root),
        ]
        if params.get("disable_wandb", False):
            cmd.append("--disable-wandb")
        if params.get("no_save", False):
            cmd.append("--no_save")
        if params.get("no_visualize", True):
            cmd.append("--no_visualize")
        if params.get("tf_flag", False):
            cmd.append("--tf")
        print("[CausalForcing] Running:", " ".join(cmd))
        return subprocess.run(cmd, cwd=str(backend_root), env=self._base_env(backend_root), check=True).returncode

    def _run_infer(self, backend_root: Path, config_path: Path, run_root: Path, jsonl_path: Path) -> int:
        params = self._runner_params()
        num_processes = int(params.get("num_processes", 1))
        if num_processes > 1 and params.get("i2v", False):
            raise ValueError("Causal-Forcing I2V inference does not support distributed launch.")
        cmd = self._launch_cmd(backend_root, default_master_port=29578) + [
            "inference.py",
            "--config_path",
            self._relative_to_backend(config_path, backend_root),
            "--data_path",
            self._relative_to_backend(jsonl_path, backend_root),
            "--output_folder",
            self._relative_to_backend(run_root, backend_root),
            "--num_output_frames",
            str(params.get("num_output_frames", 21)),
            "--seed",
            str(params.get("seed", 0)),
        ]
        checkpoint_path = params.get("checkpoint_path")
        if checkpoint_path:
            cmd.extend(["--checkpoint_path", self._relative_to_backend(checkpoint_path, backend_root)])
        if params.get("use_ema", False):
            cmd.append("--use_ema")
        if params.get("detail_log", False):
            cmd.append("--detail-log")
        if int(params.get("sampling_steps", 0)) > 0:
            cmd.extend(["--sampling_steps", str(params["sampling_steps"])])
        if int(params.get("vertical_infer_fixed_denoise_steps", -1)) >= 0:
            cmd.extend([
                "--vertical_infer_fixed_denoise_steps",
                str(params["vertical_infer_fixed_denoise_steps"]),
            ])
        if params.get("vertical_infer_preserve_budget_ratio", False):
            cmd.append("--vertical_infer_preserve_budget_ratio")
        if int(params.get("vertical_infer_reference_total_steps", 0)) > 0:
            cmd.extend([
                "--vertical_infer_reference_total_steps",
                str(params["vertical_infer_reference_total_steps"]),
            ])
        print("[CausalForcing] Running:", " ".join(cmd))
        return subprocess.run(cmd, cwd=str(backend_root), env=self._base_env(backend_root), check=True).returncode

    def run(self):
        params = self._runner_params()
        backend_root = Path(params.get("backend_root", str(self.DEFAULT_BACKEND_ROOT)))
        if not (backend_root / "train.py").exists():
            raise FileNotFoundError(f"Causal-Forcing backend not found: {backend_root}")

        run_root = self._run_root()
        run_root.mkdir(parents=True, exist_ok=True)
        jsonl_path = self._prepare_dataset(run_root)
        config_path = self._prepare_config(run_root, jsonl_path, backend_root)

        if self._task() == "infer":
            self._run_infer(backend_root, config_path, run_root, jsonl_path)
        else:
            self._run_train(backend_root, config_path, run_root)

        dataset_params = {}
        full_config = self._full_config()
        if hasattr(full_config, "dataset") and hasattr(full_config.dataset, "params"):
            dataset_params = (
                full_config.dataset.params.to_dict()
                if hasattr(full_config.dataset.params, "to_dict")
                else dict(full_config.dataset.params)
            )
        causal_overrides = params.get("causal_config_overrides", {}) or {}
        result = BackendRunResult(
            backend="causal_forcing",
            task=self._task(),
            generated_dir=str(run_root),
            artifact_type="video" if self._task() == "infer" else "training_log",
            metadata_path=str(jsonl_path) if self._task() == "infer" else dataset_params.get("metadata_path"),
            dataset_base_path=str(jsonl_path.parent) if self._task() == "infer" else dataset_params.get("base_path"),
            fps=params.get("fps"),
            num_frames=causal_overrides.get("num_frames"),
            extra={
                "backend_root": str(backend_root),
                "config_path": str(config_path),
                "jsonl_path": str(jsonl_path),
                "source_metadata_path": dataset_params.get("metadata_path"),
                "source_dataset_base_path": dataset_params.get("base_path"),
                "video_key": "video_path" if self._task() == "infer" else None,
            },
        )
        manifest_path = result.write_manifest(run_root)
        print(f"[CausalForcing] Wrote backend manifest to {manifest_path}")
        return result
