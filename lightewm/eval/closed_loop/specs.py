from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _require_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int | str, ...]
    dtype: str
    semantics: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorSpec":
        shape = tuple(data.get("shape", ()))
        if not shape:
            raise ValueError("tensor shape must not be empty")
        for dimension in shape:
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError(f"tensor dimensions must be positive: {shape}")
            if not isinstance(dimension, (int, str)):
                raise ValueError(f"unsupported tensor dimension: {dimension!r}")
        return cls(
            shape=shape,
            dtype=_require_text(data.get("dtype"), "tensor dtype"),
            semantics=data.get("semantics"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"shape": list(self.shape), "dtype": self.dtype}
        if self.semantics is not None:
            data["semantics"] = self.semantics
        return data


@dataclass(frozen=True)
class CheckpointSpec:
    format: str
    uri: str
    sha256: str | None = None
    size_bytes: int | None = None
    backbone_uri: str | None = None
    config_uri: str | None = None
    stats_uri: str | None = None
    processors_uri: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointSpec":
        sha256 = data.get("sha256")
        if sha256 is not None:
            sha256 = str(sha256).lower()
            if len(sha256) != 64 or any(
                character not in "0123456789abcdef" for character in sha256
            ):
                raise ValueError("checkpoint sha256 must be 64 hexadecimal characters")
        size_bytes = data.get("size_bytes")
        if size_bytes is not None:
            size_bytes = int(size_bytes)
            if size_bytes <= 0:
                raise ValueError("checkpoint size_bytes must be positive")
        return cls(
            format=_require_text(data.get("format"), "checkpoint format"),
            uri=_require_text(data.get("uri"), "checkpoint uri"),
            sha256=sha256,
            size_bytes=size_bytes,
            backbone_uri=data.get("backbone_uri"),
            config_uri=data.get("config_uri"),
            stats_uri=data.get("stats_uri"),
            processors_uri=data.get("processors_uri"),
        )


@dataclass(frozen=True)
class RuntimeSpec:
    mode: str
    endpoint: str | None = None
    command: tuple[str, ...] = ()
    environment: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeSpec":
        mode = _require_text(data.get("mode"), "runtime mode")
        if mode not in {"local", "server"}:
            raise ValueError(f"runtime mode must be local or server, got {mode!r}")
        endpoint = data.get("endpoint")
        if mode == "server" and not endpoint:
            raise ValueError("server runtime requires an endpoint")
        return cls(
            mode=mode,
            endpoint=endpoint,
            command=tuple(str(value) for value in data.get("command", ())),
            environment=data.get("environment"),
        )


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str
    checkpoint: CheckpointSpec
    runtime: RuntimeSpec
    observations: dict[str, TensorSpec]
    action: TensorSpec
    action_chunk_size: int = 1
    execution_horizon: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicySpec":
        observations = {
            str(name): TensorSpec.from_dict(spec)
            for name, spec in dict(data.get("observations", {})).items()
        }
        if not observations:
            raise ValueError("policy observations must not be empty")
        action_chunk_size = int(data.get("action_chunk_size", 1))
        execution_horizon = int(data.get("execution_horizon", action_chunk_size))
        if action_chunk_size <= 0 or execution_horizon <= 0:
            raise ValueError("action chunk size and execution horizon must be positive")
        if execution_horizon > action_chunk_size:
            raise ValueError("execution horizon cannot exceed action chunk size")
        return cls(
            name=_require_text(data.get("name"), "policy name"),
            family=_require_text(data.get("family"), "policy family"),
            checkpoint=CheckpointSpec.from_dict(dict(data.get("checkpoint", {}))),
            runtime=RuntimeSpec.from_dict(dict(data.get("runtime", {}))),
            observations=observations,
            action=TensorSpec.from_dict(dict(data.get("action", {}))),
            action_chunk_size=action_chunk_size,
            execution_horizon=execution_horizon,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PolicySpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "checkpoint": {
                key: value
                for key, value in asdict(self.checkpoint).items()
                if value is not None
            },
            "runtime": {
                key: list(value) if key == "command" else value
                for key, value in asdict(self.runtime).items()
                if value not in (None, (), [])
            },
            "observations": {
                name: spec.to_dict() for name, spec in self.observations.items()
            },
            "action": self.action.to_dict(),
            "action_chunk_size": self.action_chunk_size,
            "execution_horizon": self.execution_horizon,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def content_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    revision: str
    protocol: str
    official_root: str
    tasks: tuple[str, ...]
    settings: tuple[str, ...]
    seeds: tuple[int, ...]
    episodes_per_case: int
    max_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSpec":
        tasks = tuple(str(value) for value in data.get("tasks", ()))
        settings = tuple(str(value) for value in data.get("settings", ("default",)))
        seeds = tuple(int(value) for value in data.get("seeds", (0,)))
        if not tasks:
            raise ValueError("benchmark tasks must not be empty")
        if not settings:
            raise ValueError("benchmark settings must not be empty")
        if not seeds:
            raise ValueError("benchmark seeds must not be empty")
        episodes_per_case = int(data.get("episodes_per_case", 1))
        max_steps = int(data.get("max_steps", 1))
        if episodes_per_case <= 0 or max_steps <= 0:
            raise ValueError("episodes_per_case and max_steps must be positive")
        return cls(
            name=_require_text(data.get("name"), "benchmark name"),
            revision=_require_text(data.get("revision"), "benchmark revision"),
            protocol=_require_text(data.get("protocol"), "benchmark protocol"),
            official_root=_require_text(data.get("official_root"), "official root"),
            tasks=tasks,
            settings=settings,
            seeds=seeds,
            episodes_per_case=episodes_per_case,
            max_steps=max_steps,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkSpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("tasks", "settings", "seeds"):
            data[key] = list(data[key])
        if not self.metadata:
            data.pop("metadata")
        return data

    def content_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
