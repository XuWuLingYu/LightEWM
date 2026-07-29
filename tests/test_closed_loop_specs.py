import tempfile
import unittest
from pathlib import Path

import yaml

from lightewm.eval.closed_loop.specs import BenchmarkSpec, PolicySpec


class ClosedLoopSpecTest(unittest.TestCase):
    def test_policy_spec_round_trip_and_hash(self):
        data = {
            "name": "starwam-robotwin",
            "family": "starwam",
            "checkpoint": {
                "format": "state_dict",
                "uri": "/models/starwam.pt",
                "sha256": "a" * 64,
                "size_bytes": 1234,
                "backbone_uri": "/models/Wan2.2-TI2V-5B",
                "stats_uri": "/models/action_stats.json",
            },
            "runtime": {"mode": "server", "endpoint": "tcp://127.0.0.1:8765"},
            "observations": {
                "cam_high": {"shape": [256, 320, 3], "dtype": "uint8"},
                "proprio": {"shape": [14], "dtype": "float32"},
            },
            "action": {
                "shape": [32, 14],
                "dtype": "float32",
                "semantics": "dual_arm_joint_position_chunk",
            },
            "action_chunk_size": 32,
            "execution_horizon": 24,
        }
        spec = PolicySpec.from_dict(data)
        self.assertEqual(spec.action.shape, (32, 14))
        self.assertEqual(spec.runtime.mode, "server")
        self.assertEqual(len(spec.content_hash()), 64)
        self.assertEqual(spec.to_dict(), data)

    def test_policy_spec_rejects_invalid_checkpoint_identity(self):
        data = {
            "name": "policy",
            "family": "test",
            "checkpoint": {
                "format": "state_dict",
                "uri": "/models/policy.pt",
                "sha256": "not-a-digest",
            },
            "runtime": {"mode": "local"},
            "observations": {"state": {"shape": [8], "dtype": "float32"}},
            "action": {"shape": [1, 8], "dtype": "float32"},
        }
        with self.assertRaisesRegex(ValueError, "sha256"):
            PolicySpec.from_dict(data)

    def test_benchmark_spec_loads_yaml_and_rejects_empty_revision(self):
        data = {
            "name": "robotwin",
            "revision": "13c3c47",
            "protocol": "reference-smoke-v1",
            "official_root": "/opt/RoboTwin",
            "tasks": ["adjust_bottle"],
            "settings": ["demo_clean", "demo_randomized"],
            "seeds": [100000],
            "episodes_per_case": 5,
            "max_steps": 500,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.yaml"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            spec = BenchmarkSpec.from_yaml(path)
        self.assertEqual(spec.tasks, ("adjust_bottle",))
        self.assertEqual(spec.settings, ("demo_clean", "demo_randomized"))

        data["revision"] = ""
        with self.assertRaisesRegex(ValueError, "revision"):
            BenchmarkSpec.from_dict(data)


if __name__ == "__main__":
    unittest.main()
