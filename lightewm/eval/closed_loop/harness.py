from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CommandStep:
    name: str
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    expected_output: tuple[str, ...] = ()
    required_log_patterns: tuple[str, ...] = ()
    forbidden_log_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("command step name must not be empty")
        if not self.argv or not all(str(value) for value in self.argv):
            raise ValueError("command step argv must not be empty")


@dataclass(frozen=True)
class CommandResult:
    name: str
    returncode: int
    command_path: str
    log_path: str
    started_at: str
    finished_at: str
    expected_output: tuple[str, ...]
    missing_expected_output: tuple[str, ...] = ()
    missing_log_patterns: tuple[str, ...] = ()
    forbidden_log_patterns_found: tuple[str, ...] = ()


class OfficialHarness:
    """Run official benchmark entrypoints while preserving command evidence."""

    def __init__(self, evidence_root: str | Path):
        self.evidence_root = Path(evidence_root)

    def run(self, step: CommandStep) -> CommandResult:
        cwd = Path(step.cwd)
        if not cwd.is_dir():
            raise FileNotFoundError(f"official harness root not found: {cwd}")
        step_dir = self.evidence_root / step.name
        step_dir.mkdir(parents=True, exist_ok=True)
        command_path = step_dir / "command.json"
        log_path = step_dir / "stdout_stderr.log"
        started_at = datetime.now(timezone.utc).isoformat()
        command_payload = {
            "name": step.name,
            "argv": list(step.argv),
            "cwd": str(cwd),
            "env_overrides": step.env,
            "expected_output": list(step.expected_output),
            "required_log_patterns": list(step.required_log_patterns),
            "forbidden_log_patterns": list(step.forbidden_log_patterns),
            "started_at": started_at,
        }
        command_path.write_text(
            json.dumps(command_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update(step.env)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                list(step.argv),
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        finished_at = datetime.now(timezone.utc).isoformat()
        missing = [
            value for value in step.expected_output if not (cwd / value).exists()
        ]
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        missing_log_patterns = [
            pattern
            for pattern in step.required_log_patterns
            if pattern not in log_text
        ]
        forbidden_log_patterns_found = [
            pattern
            for pattern in step.forbidden_log_patterns
            if pattern in log_text
        ]
        result = CommandResult(
            name=step.name,
            returncode=completed.returncode,
            command_path=str(command_path),
            log_path=str(log_path),
            started_at=started_at,
            finished_at=finished_at,
            expected_output=step.expected_output,
            missing_expected_output=tuple(missing),
            missing_log_patterns=tuple(missing_log_patterns),
            forbidden_log_patterns_found=tuple(forbidden_log_patterns_found),
        )
        (step_dir / "result.json").write_text(
            json.dumps(
                asdict(result),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode:
            raise subprocess.CalledProcessError(
                completed.returncode, list(step.argv), output=str(log_path)
            )
        if missing:
            raise FileNotFoundError(
                f"{step.name} did not produce expected outputs: {missing}"
            )
        if missing_log_patterns or forbidden_log_patterns_found:
            raise RuntimeError(
                f"{step.name} log validation failed: "
                f"missing={missing_log_patterns}, "
                f"forbidden={forbidden_log_patterns_found}"
            )
        return result
