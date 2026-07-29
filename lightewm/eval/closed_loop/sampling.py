from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .harness import CommandStep
from .protocols import RoboLabProtocol, RoboTwinProtocol


SUPPORTED_GATES = {
    "robotwin": {"ground-truth"},
    "robolab": {"recorded-replay", "deterministic-hold"},
}


@dataclass(frozen=True)
class SampleCase:
    task: str
    gate: str
    setting: str = "default"
    steps: int = 50
    rationale: str = ""
    attributes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleCase":
        task = str(data.get("task", "")).strip()
        gate = str(data.get("gate", "")).strip()
        if not task or not gate:
            raise ValueError("sample case task and gate must not be empty")
        steps = int(data.get("steps", 50))
        if steps <= 0:
            raise ValueError(f"sample case steps must be positive: {task}")
        return cls(
            task=task,
            gate=gate,
            setting=str(data.get("setting", "default")),
            steps=steps,
            rationale=str(data.get("rationale", "")),
            attributes=tuple(str(value) for value in data.get("attributes", ())),
        )


@dataclass(frozen=True)
class BenchmarkSample:
    name: str
    population: int
    fraction: float
    repository_revision: str = ""
    cases: tuple[SampleCase, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(
        cls,
        *,
        name: str,
        fraction: float,
        data: dict[str, Any],
    ) -> "BenchmarkSample":
        if name not in SUPPORTED_GATES:
            raise ValueError(f"unsupported sampled benchmark: {name}")
        population = int(data.get("population", 0))
        if population <= 0:
            raise ValueError(f"{name} population must be positive")
        repository_revision = str(data.get("repository_revision", "")).strip()
        if repository_revision and (
            len(repository_revision) != 40
            or any(value not in "0123456789abcdef" for value in repository_revision)
        ):
            raise ValueError(
                f"{name} repository_revision must be a 40-character lowercase SHA"
            )
        cases = tuple(
            SampleCase.from_dict(dict(value)) for value in data.get("cases", ())
        )
        expected = math.ceil(population * fraction)
        if len(cases) != expected:
            raise ValueError(
                f"{name} expected {expected} cases for fraction {fraction}, "
                f"got {len(cases)}"
            )
        tasks = [case.task for case in cases]
        if len(tasks) != len(set(tasks)):
            raise ValueError(f"{name} sample contains duplicate tasks")
        unsupported = sorted(
            {case.gate for case in cases} - SUPPORTED_GATES[name]
        )
        if unsupported:
            raise ValueError(f"{name} sample contains unsupported gates: {unsupported}")
        return cls(
            name=name,
            population=population,
            fraction=fraction,
            repository_revision=repository_revision,
            cases=cases,
        )


@dataclass(frozen=True)
class CorrectnessSample:
    schema_version: int
    fraction: float
    benchmarks: dict[str, BenchmarkSample]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrectnessSample":
        schema_version = int(data.get("schema_version", 0))
        if schema_version != 1:
            raise ValueError(
                f"unsupported correctness sample schema: {schema_version}"
            )
        fraction = float(data.get("fraction", 0.0))
        if not 0.0 < fraction <= 1.0:
            raise ValueError("correctness sample fraction must be in (0, 1]")
        benchmarks = {
            str(name): BenchmarkSample.from_dict(
                name=str(name),
                fraction=fraction,
                data=dict(payload),
            )
            for name, payload in dict(data.get("benchmarks", {})).items()
        }
        if not benchmarks:
            raise ValueError("correctness sample must contain benchmarks")
        return cls(
            schema_version=schema_version,
            fraction=fraction,
            benchmarks=benchmarks,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CorrectnessSample":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(dict(payload))


def build_sample_step(
    benchmark: str,
    case: SampleCase,
    *,
    official_root: str,
    python: str,
) -> CommandStep:
    if benchmark == "robotwin":
        protocol = RoboTwinProtocol(root=official_root, python=python)
        if case.gate == "ground-truth":
            return protocol.collect(case.task, case.setting)
    elif benchmark == "robolab":
        protocol = RoboLabProtocol(root=official_root, python=python)
        if case.gate == "recorded-replay":
            return protocol.recorded_replay(case.task)
        if case.gate == "deterministic-hold":
            return protocol.deterministic_hold(case.task, steps=case.steps)
    raise ValueError(f"unsupported sample gate: {benchmark}/{case.gate}")
