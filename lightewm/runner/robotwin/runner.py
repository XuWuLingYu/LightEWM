from __future__ import annotations

import os
import subprocess
from pathlib import Path

from lightewm.runner.backend_result import BackendRunResult


class RoboTwinRunner:
    """Manifest-compatible adapter for RoboTwin evaluation commands.

    The runner intentionally keeps RoboTwin-specific execution outside LightEWM:
    pass a command when this runner should launch RoboTwin, or point output_dir at
    an existing rollout directory when another process already produced artifacts.
    """

    def __init__(self, config):
        self.config = config

    def _params(self) -> dict:
        params = self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)
        for key in ("class_path", "section_name", "full_config"):
            params.pop(key, None)
        return params

    def run(self):
        params = self._params()
        output_dir = Path(params.get("output_dir", "./outputs/robotwin"))
        output_dir.mkdir(parents=True, exist_ok=True)

        command = params.get("command")
        if command:
            env = os.environ.copy()
            for key, value in dict(params.get("env", {})).items():
                env[str(key)] = str(value)
            cwd = params.get("cwd")
            shell = isinstance(command, str)
            print("[RoboTwin] Running:", command if shell else " ".join(map(str, command)))
            subprocess.run(command, cwd=cwd, env=env, shell=shell, check=True)

        result = BackendRunResult(
            backend="robotwin",
            task=str(getattr(self.config.full_config, "task", "eval")) if hasattr(self.config, "full_config") else "eval",
            generated_dir=str(output_dir),
            artifact_type=params.get("artifact_type", "robotwin_rollout"),
            metadata_path=params.get("metadata_path"),
            dataset_base_path=params.get("dataset_base_path"),
            video_pattern=params.get("video_pattern", "*.mp4"),
            fps=params.get("fps"),
            num_frames=params.get("num_frames"),
            extra={
                "metrics": params.get("metrics", ["success_rate"]),
                "success_log": params.get("success_log"),
                "trajectory_pattern": params.get("trajectory_pattern"),
            },
        )
        manifest_path = result.write_manifest(output_dir)
        print(f"[RoboTwin] Wrote backend manifest to {manifest_path}")
        return result
