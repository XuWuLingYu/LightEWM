from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class FastWAMRunner:
    """LightEWM runner for the vendored FastWAM implementation.

    The FastWAM model code, MoT wrapper, configs, training script, and LIBERO
    evaluator live under `lightewm/vendor/fastwam`. LightEWM provides the
    unified config entrypoint and run-directory convention around that vendored
    implementation.
    """

    def __init__(self, config):
        self.config = config
        self.repo_root = Path.cwd()

    def _full_config(self):
        return self.config.full_config

    def _runner_params(self) -> dict[str, Any]:
        params = self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)
        for key in ("class_path", "section_name", "full_config"):
            params.pop(key, None)
        return params

    def _task(self) -> str:
        return str(getattr(self._full_config(), "task", "train"))

    def _repo_path(self, path: str | Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        return self.repo_root / path

    def _relative_to_fastwam(self, path: str | Path, fastwam_root: Path) -> str:
        return os.path.relpath(self._repo_path(path).resolve(), start=fastwam_root.resolve())

    def _run_root(self) -> Path:
        params = self._runner_params()
        if self._task() == "eval":
            return Path(params.get("output_dir", "./logs/fastwam_eval"))
        run_root = Path(params.get("output_path", "./logs/fastwam_train"))
        if self._task() == "train" and bool(params.get("timestamped_output", True)):
            run_name = str(params.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S")))
            return run_root / run_name
        return run_root

    def _base_env(self, fastwam_root: Path) -> dict[str, str]:
        params = self._runner_params()
        env = os.environ.copy()
        extra_pythonpath = params.get(
            "extra_pythonpath",
            ["data/python-packages/fastwam_pydeps", "third_parties/LIBERO"],
        )
        if isinstance(extra_pythonpath, (str, Path)):
            extra_pythonpath = [extra_pythonpath]
        pythonpath = [
            str(self._repo_path(path).resolve())
            for path in extra_pythonpath
            if str(path).strip()
        ]
        pythonpath.append(str(fastwam_root))
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str((self.repo_root / "checkpoints").resolve()))
        env.setdefault("LIBERO_CONFIG_PATH", str(self._ensure_libero_config()))
        return env

    def _ensure_libero_config(self) -> Path:
        config_dir = self.repo_root / "data" / "libero_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.yaml"
        benchmark_root = self.repo_root / "third_parties" / "LIBERO" / "libero" / "libero"
        if not config_file.exists():
            config_file.write_text(
                "\n".join(
                    [
                        f"benchmark_root: {benchmark_root}",
                        f"bddl_files: {benchmark_root / 'bddl_files'}",
                        f"init_states: {benchmark_root / 'init_files'}",
                        f"datasets: {self.repo_root / 'data' / 'LIBERO-datasets'}",
                        f"assets: {benchmark_root / 'assets'}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        return config_dir

    @staticmethod
    def _python() -> str:
        return sys.executable

    def _check_training_data(self, dataset_dirs: list[str]) -> None:
        missing = [path for path in dataset_dirs if not self._repo_path(path).exists()]
        if missing:
            joined = "\n".join(f"  - {path}" for path in missing)
            raise FileNotFoundError(
                "FastWAM LIBERO training expects the official LeRobot-format dataset. "
                "Missing dataset directories:\n"
                f"{joined}\n"
                "See examples/LIBERO-FASTWAM/README.md for download instructions."
            )

    def _common_overrides(self, fastwam_root: Path, run_root: Path) -> list[str]:
        params = self._runner_params()
        task_name = str(params.get("fastwam_task", "libero_joint_2cam224_1e-4"))
        model_name = str(params.get("fastwam_model", "fastwam_joint"))
        data_name = str(params.get("fastwam_data", "libero_2cam"))
        dataset_dirs = [str(p) for p in params.get("dataset_dirs", [])]
        if self._task() in {"train", "precompute_text"}:
            self._check_training_data(dataset_dirs)

        overrides = [
            f"task={task_name}",
            f"model={model_name}",
            f"data={data_name}",
            f"output_dir={self._relative_to_fastwam(run_root, fastwam_root)}",
        ]
        if dataset_dirs:
            rel_dirs = [self._relative_to_fastwam(path, fastwam_root) for path in dataset_dirs]
            overrides.append("data.train.dataset_dirs=[" + ",".join(rel_dirs) + "]")

        text_cache_dir = params.get("text_embedding_cache_dir")
        if text_cache_dir:
            overrides.append(
                "data.train.text_embedding_cache_dir="
                + self._relative_to_fastwam(text_cache_dir, fastwam_root)
            )

        action_dit_path = params.get("action_dit_pretrained_path")
        if action_dit_path:
            overrides.append(
                "model.action_dit_pretrained_path="
                + str(self._repo_path(action_dit_path).resolve())
            )

        video_dit_path = params.get("video_dit_pretrained_path")
        if video_dit_path:
            overrides.append(
                "model.video_dit_pretrained_path="
                + str(self._repo_path(video_dit_path).resolve())
            )

        resume_path = params.get("resume")
        if resume_path:
            overrides.append("resume=" + str(self._repo_path(resume_path).resolve()))

        model_id = params.get("model_id")
        if model_id:
            overrides.append(f"model.model_id={model_id}")
        tokenizer_model_id = params.get("tokenizer_model_id")
        if tokenizer_model_id:
            overrides.append(f"model.tokenizer_model_id={tokenizer_model_id}")

        overrides.extend(str(item) for item in params.get("hydra_overrides", []))
        return overrides

    def _run_precompute_text(self, fastwam_root: Path, run_root: Path) -> int:
        params = self._runner_params()
        overrides = self._common_overrides(fastwam_root, run_root)
        num_processes = int(params.get("num_processes", 8))
        cmd = [
            self._python(),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(num_processes),
            "scripts/precompute_text_embeds.py",
            *overrides,
        ]
        print("[FastWAM] Running:", " ".join(cmd))
        return subprocess.run(cmd, cwd=str(fastwam_root), env=self._base_env(fastwam_root), check=True).returncode

    def _run_train(self, fastwam_root: Path, run_root: Path) -> int:
        params = self._runner_params()
        overrides = self._common_overrides(fastwam_root, run_root)
        num_processes = int(params.get("num_processes", 8))
        zero_stage = int(params.get("zero_stage", 1))
        script_name = "train_zero2.sh" if zero_stage == 2 else "train_zero1.sh"
        cmd = ["bash", f"scripts/{script_name}", str(num_processes), *overrides]
        print("[FastWAM] Running:", " ".join(cmd))
        return subprocess.run(cmd, cwd=str(fastwam_root), env=self._base_env(fastwam_root), check=True).returncode

    def _run_eval(self, fastwam_root: Path, run_root: Path) -> int:
        params = self._runner_params()
        ckpt_path = params.get("checkpoint_path")
        if not ckpt_path:
            raise ValueError("FastWAM eval requires runner.params.checkpoint_path.")
        dataset_stats_path = params.get("dataset_stats_path")
        if not dataset_stats_path:
            raise ValueError("FastWAM eval requires runner.params.dataset_stats_path.")

        overrides = self._common_overrides(fastwam_root, run_root)
        overrides.extend([
            "ckpt=" + self._relative_to_fastwam(ckpt_path, fastwam_root),
            "EVALUATION.dataset_stats_path=" + self._relative_to_fastwam(dataset_stats_path, fastwam_root),
            "EVALUATION.output_dir=" + self._relative_to_fastwam(run_root, fastwam_root),
            f"MULTIRUN.num_gpus={int(params.get('num_gpus', params.get('num_processes', 8)))}",
            f"MULTIRUN.max_tasks_per_gpu={int(params.get('max_tasks_per_gpu', 2))}",
            f"EVALUATION.num_trials={int(params.get('num_trials', 10))}",
        ])
        task_suites = params.get("task_suite_names")
        if task_suites:
            overrides.append("MULTIRUN.task_suite_names=[" + ",".join(str(x) for x in task_suites) + "]")

        cmd = [self._python(), "experiments/libero/run_libero_manager.py", *overrides]
        print("[FastWAM] Running:", " ".join(cmd))
        return subprocess.run(cmd, cwd=str(fastwam_root), env=self._base_env(fastwam_root), check=True).returncode

    def run(self):
        params = self._runner_params()
        fastwam_root = self._repo_path(params.get("fastwam_root", "lightewm/vendor/fastwam")).resolve()
        if not (fastwam_root / "scripts" / "train.py").exists():
            raise FileNotFoundError(f"FastWAM backend not found: {fastwam_root}")

        run_root = self._run_root()
        run_root.mkdir(parents=True, exist_ok=True)
        task = self._task()
        if task == "precompute_text":
            return self._run_precompute_text(fastwam_root, run_root)
        if task == "eval":
            return self._run_eval(fastwam_root, run_root)
        return self._run_train(fastwam_root, run_root)
