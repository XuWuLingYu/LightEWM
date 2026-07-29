#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _torch_load(path: Path) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to validate a torch checkpoint"
        ) from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"checkpoint is not loadable by torch.load: {path}") from exc


def validate_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if checkpoint_path.stat().st_size == 0:
        raise ValueError(f"checkpoint is empty: {checkpoint_path}")

    payload = _torch_load(checkpoint_path)
    if not isinstance(payload, Mapping):
        raise ValueError(
            "checkpoint root must be a mapping/state_dict, "
            f"got {type(payload).__name__}"
        )

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return {
        "path": str(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "root_type": type(payload).__name__,
        "root_keys": len(payload),
        "valid": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail unless a file is a loadable mapping-style torch checkpoint."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = validate_checkpoint(args.checkpoint)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
