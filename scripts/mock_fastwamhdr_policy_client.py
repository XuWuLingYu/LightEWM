#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import socket
from typing import Any, Mapping

import numpy as np


def _to_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return {"__bytes__": True, "data": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Mapping):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_numpy(v) for v in obj]
    return obj


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(_to_numpy(data), cls=NumpyEncoder, ensure_ascii=False)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct: dict[str, Any]) -> Any:
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=np.dtype(dct["dtype"])).reshape(dct["shape"])
        if "__bytes__" in dct:
            return base64.b64decode(dct["data"])
        return dct

    return json.loads(json_str, object_hook=object_hook)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    while size > 0:
        chunk = sock.recv(min(size, 1 << 20))
        if not chunk:
            raise ConnectionError("connection closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


class Client:
    def __init__(self, host: str, port: int, timeout: float = 120.0) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((host, port))

    def call(self, cmd: str, obs: Any = None) -> Any:
        payload = numpy_to_json({"cmd": cmd, "obs": obs}).encode("utf-8")
        self.sock.sendall(len(payload).to_bytes(4, "big"))
        self.sock.sendall(payload)
        header = recv_exact(self.sock, 4)
        response = json_to_numpy(recv_exact(self.sock, int.from_bytes(header, "big")).decode("utf-8"))
        if "error" in response:
            raise RuntimeError(response["error"] + "\n" + response.get("traceback", ""))
        return response

    def close(self) -> None:
        self.sock.close()


def make_obs(env_idx: int, instruction: str) -> dict[str, Any]:
    rng = np.random.default_rng(1000 + env_idx)
    image = rng.integers(0, 255, size=(240, 320, 3), dtype=np.uint8)
    return {
        "env_idx": env_idx,
        "task_instruction": instruction,
        "vision": {
            "cam_head": {"color": image},
            "cam_left_wrist": {"color": np.roll(image, shift=env_idx + 1, axis=1)},
            "cam_right_wrist": {"color": np.roll(image, shift=env_idx + 2, axis=0)},
        },
        "state": {
            "left_arm_joint_state": np.zeros((6,), dtype=np.float32),
            "left_ee_joint_state": np.zeros((1,), dtype=np.float32),
            "right_arm_joint_state": np.zeros((6,), dtype=np.float32),
            "right_ee_joint_state": np.zeros((1,), dtype=np.float32),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    client = Client(args.host, args.port)
    try:
        client.call("reset")
        obs = [make_obs(i, "Pour the liquid from the bottle into the cup.") for i in range(args.batch_size)]
        print(client.call("update_obs_batch", obs)["server_timing"])
        response = client.call("get_action_batch", list(range(args.batch_size)))
        actions = response["res"]
        print(response["server_timing"])
        print("batch", len(actions), "chunk", len(actions[0]) if actions else 0)
        first = actions[0][0]
        print({k: np.asarray(v).shape for k, v in first.items()})
    finally:
        client.close()


if __name__ == "__main__":
    main()
