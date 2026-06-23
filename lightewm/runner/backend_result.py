from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MANIFEST_FILENAME = "backend_manifest.json"


@dataclass
class BackendRunResult:
    """Stable artifact contract between backend runners and evaluators."""

    backend: str
    task: str
    generated_dir: str
    artifact_type: str = "video"
    metadata_path: str | None = None
    dataset_base_path: str | None = None
    video_pattern: str = "*.mp4"
    fps: int | None = None
    num_frames: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    def write_manifest(self, output_dir: str | Path | None = None) -> Path:
        target_dir = Path(output_dir or self.generated_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return manifest_path


def read_backend_manifest(path: str | Path) -> BackendRunResult:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    extra = data.pop("extra", {}) or {}
    known = set(BackendRunResult.__dataclass_fields__)
    unknown = {key: data.pop(key) for key in list(data) if key not in known}
    if unknown:
        extra.update(unknown)
    data["extra"] = extra
    return BackendRunResult(**data)
