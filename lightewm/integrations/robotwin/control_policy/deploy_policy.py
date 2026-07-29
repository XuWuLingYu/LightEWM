from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_CONTROL_MODES = {"hold", "zero", "wrong_gripper", "reverse"}


@dataclass
class ControlProbeModel:
    control_mode: str
    control_probe_path: Path
    gripper_indices: tuple[int, int] = (6, 13)
    episode: int = -1
    steps: int = 0
    initial: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    last_action: list[float] = field(default_factory=list)


def get_model(usr_args: dict[str, Any]) -> ControlProbeModel:
    mode = str(usr_args.get("control_mode", "hold"))
    if mode not in _CONTROL_MODES:
        raise ValueError(f"unsupported control_mode {mode!r}; expected one of {sorted(_CONTROL_MODES)}")
    return ControlProbeModel(
        control_mode=mode,
        control_probe_path=Path(usr_args.get("control_probe_path", "control_probe.json")),
    )


def reset_model(model: ControlProbeModel) -> None:
    model.episode += 1
    model.steps = 0
    model.initial = None
    model.final = None
    model.last_action = []


def _joint_vector(observation: dict[str, Any]) -> np.ndarray:
    joint_action = observation["joint_action"]
    if "vector" in joint_action:
        return np.asarray(joint_action["vector"], dtype=np.float32)
    return np.asarray(
        [
            *joint_action["left_arm"],
            joint_action["left_gripper"],
            *joint_action["right_arm"],
            joint_action["right_gripper"],
        ],
        dtype=np.float32,
    )


def _objects(task_env: Any) -> dict[str, list[float]]:
    objects: dict[str, list[float]] = {}
    for actor in task_env.scene.get_all_actors():
        name = str(actor.get_name())
        pose = actor.get_pose()
        objects[name] = [
            *np.asarray(pose.p, dtype=np.float64).tolist(),
            *np.asarray(pose.q, dtype=np.float64).tolist(),
        ]
    return objects


def _snapshot(task_env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "qpos": np.round(_joint_vector(observation).astype(float), decimals=8).tolist(),
        "objects": _objects(task_env),
    }


def _action(model: ControlProbeModel, qpos: np.ndarray) -> np.ndarray:
    if model.control_mode == "hold":
        return qpos.copy()
    if model.control_mode == "zero":
        return np.zeros_like(qpos)
    if model.control_mode == "reverse":
        return -qpos
    action = qpos.copy()
    for index in model.gripper_indices:
        if index >= len(action):
            raise ValueError(f"gripper index {index} is outside {len(action)}-D qpos")
        action[index] = 0.0
    return action


def _write_probe(model: ControlProbeModel, success: bool) -> None:
    if model.control_probe_path == Path("/dev/null"):
        return
    payload = {
        "schema_version": 1,
        "action_mode": model.control_mode,
        "episode": model.episode,
        "steps": model.steps,
        "success": bool(success),
        "initial": model.initial,
        "final": model.final,
        "last_action": model.last_action,
    }
    path = model.control_probe_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def eval(task_env: Any, model: ControlProbeModel, observation: dict[str, Any]) -> dict[str, Any]:
    if model.initial is None:
        model.initial = _snapshot(task_env, observation)
    action = _action(model, _joint_vector(observation))
    task_env.take_action(action, action_type="qpos")
    next_observation = task_env.get_obs()
    model.steps += 1
    model.last_action = action.astype(float).tolist()
    model.final = _snapshot(task_env, next_observation)
    _write_probe(model, bool(getattr(task_env, "eval_success", False)))
    return next_observation
