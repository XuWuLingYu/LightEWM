from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .records import EpisodeRecord


def _max_abs_delta(left: list[Any], right: list[Any]) -> float:
    if len(left) != len(right):
        raise ValueError(f"state vector length mismatch: {len(left)} != {len(right)}")
    return max((abs(float(a) - float(b)) for a, b in zip(left, right)), default=0.0)


def analyze_control_probe(
    payload: dict[str, Any],
    *,
    qpos_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Validate one deterministic negative-control state probe."""

    mode = str(payload["action_mode"])
    initial = payload["initial"]
    final = payload["final"]
    qpos_delta = _max_abs_delta(initial["qpos"], final["qpos"])

    initial_objects = initial.get("objects", {})
    final_objects = final.get("objects", {})
    common_objects = sorted(set(initial_objects) & set(final_objects))
    object_delta = max(
        (
            _max_abs_delta(initial_objects[name], final_objects[name])
            for name in common_objects
        ),
        default=0.0,
    )

    qpos_motion_detected = qpos_delta > qpos_tolerance
    failures = []
    if payload.get("success") is not False:
        failures.append("unexpected_success")
    if not common_objects:
        failures.append("object_state_missing")
    if mode == "hold" and qpos_motion_detected:
        failures.append("hold_qpos_moved")
    if mode != "hold" and not qpos_motion_detected:
        failures.append("active_control_no_qpos_motion")

    return {
        "passed": not failures,
        "action_mode": mode,
        "success": payload.get("success"),
        "qpos_max_abs_delta": qpos_delta,
        "qpos_motion_detected": qpos_motion_detected,
        "object_state_observed": bool(common_objects),
        "object_count": len(common_objects),
        "object_max_abs_delta": object_delta,
        "failures": failures,
    }


def analyze_gripper_toggle_probe(
    payload: dict[str, Any],
    *,
    gripper_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Validate that a gripper-toggle command reached the simulated robot."""

    mode = str(payload.get("action_mode"))
    commanded = [float(value) for value in payload.get("commanded_gripper", [])]
    observed = [float(value) for value in payload.get("observed_gripper", [])]
    commanded_range = max(commanded) - min(commanded) if commanded else 0.0
    observed_range = max(observed) - min(observed) if observed else 0.0

    failures = []
    if mode != "gripper_toggle":
        failures.append("unexpected_action_mode")
    if len(commanded) < 2:
        failures.append("commanded_gripper_missing")
    elif commanded_range <= gripper_tolerance:
        failures.append("commanded_gripper_did_not_toggle")
    if len(observed) < 2:
        failures.append("observed_gripper_missing")
    elif observed_range <= gripper_tolerance:
        failures.append("observed_gripper_did_not_move")

    return {
        "passed": not failures,
        "action_mode": mode,
        "samples": len(observed),
        "commanded_range": commanded_range,
        "observed_range": observed_range,
        "gripper_tolerance": gripper_tolerance,
        "failures": failures,
    }


def analyze_episode_gate(
    records: Iterable[EpisodeRecord],
    *,
    expectation: str,
    min_episodes: int = 1,
) -> dict[str, Any]:
    """Mechanically validate a normalized closed-loop acceptance gate."""

    if expectation not in {
        "all-success",
        "all-failure",
        "at-least-one-success",
    }:
        raise ValueError(f"unsupported episode expectation: {expectation}")
    if min_episodes < 1:
        raise ValueError("min_episodes must be positive")

    episodes = list(records)
    evaluated = [record for record in episodes if record.success is not None]
    successes = sum(record.success is True for record in evaluated)
    failures_count = sum(record.success is False for record in evaluated)
    unknown = len(episodes) - len(evaluated)
    gate_failures = []

    if len(evaluated) < min_episodes:
        gate_failures.append("insufficient_evaluated_episodes")
    if unknown:
        gate_failures.append("unknown_success_status")
    if expectation == "all-success" and failures_count:
        gate_failures.append("unexpected_failure")
    elif expectation == "all-failure" and successes:
        gate_failures.append("unexpected_success")
    elif expectation == "at-least-one-success" and successes < 1:
        gate_failures.append("no_successful_episode")

    return {
        "passed": not gate_failures,
        "expectation": expectation,
        "min_episodes": min_episodes,
        "episodes": len(episodes),
        "evaluated": len(evaluated),
        "successes": successes,
        "failures_count": failures_count,
        "unknown": unknown,
        "success_rate": successes / len(evaluated) if evaluated else None,
        "failures": gate_failures,
    }
