from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.validate_torch_checkpoint import validate_checkpoint


class _FakeTorch:
    @staticmethod
    def load(path: Path, *, map_location: str, weights_only: bool):
        assert path.is_file()
        assert map_location == "cpu"
        assert weights_only is False
        return {"model.weight": object(), "model.bias": object()}


def test_validate_checkpoint_reports_mapping_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"valid-checkpoint-placeholder")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)

    report = validate_checkpoint(checkpoint)

    assert report["valid"] is True
    assert report["root_type"] == "dict"
    assert report["root_keys"] == 2
    assert report["size_bytes"] == len(b"valid-checkpoint-placeholder")
    assert len(report["sha256"]) == 64


def test_validate_checkpoint_rejects_non_mapping_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"invalid-root")

    class FakeTorch:
        @staticmethod
        def load(path: Path, *, map_location: str, weights_only: bool):
            return [1, 2, 3]

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)

    with pytest.raises(ValueError, match="root must be a mapping"):
        validate_checkpoint(checkpoint)
