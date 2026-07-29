import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lightewm.integrations.robotwin import control_policy
from lightewm.integrations.robotwin.control_policy import deploy_policy


class _Pose:
    p = np.array([0.1, 0.2, 0.3])
    q = np.array([1.0, 0.0, 0.0, 0.0])


class _Actor:
    def get_name(self):
        return "test_object"

    def get_pose(self):
        return _Pose()


class _Scene:
    def get_all_actors(self):
        return [_Actor()]


class _TaskEnv:
    def __init__(self, qpos):
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.scene = _Scene()
        self.eval_success = False

    def take_action(self, action, action_type):
        self.qpos = np.asarray(action, dtype=np.float32)
        self.action_type = action_type

    def get_obs(self):
        return {"joint_action": {"vector": self.qpos.copy()}}


class RoboTwinControlPolicyTest(unittest.TestCase):
    def test_package_exports_robotwin_policy_entrypoints(self):
        self.assertIs(control_policy.get_model, deploy_policy.get_model)
        self.assertIs(control_policy.reset_model, deploy_policy.reset_model)
        self.assertIs(control_policy.eval, deploy_policy.eval)

    def test_zero_control_records_before_after_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.json"
            model = deploy_policy.get_model(
                {"control_mode": "zero", "control_probe_path": str(probe)}
            )
            deploy_policy.reset_model(model)
            env = _TaskEnv([0.2] * 14)
            deploy_policy.eval(env, model, env.get_obs())
            payload = json.loads(probe.read_text())

        self.assertEqual(payload["action_mode"], "zero")
        self.assertEqual(payload["initial"]["qpos"], [0.2] * 14)
        self.assertEqual(payload["final"]["qpos"], [0.0] * 14)
        self.assertIn("test_object", payload["initial"]["objects"])
        self.assertFalse(payload["success"])
        self.assertEqual(env.action_type, "qpos")

    def test_wrong_gripper_holds_arm_and_closes_grippers(self):
        model = deploy_policy.get_model(
            {"control_mode": "wrong_gripper", "control_probe_path": "/dev/null"}
        )
        deploy_policy.reset_model(model)
        qpos = np.arange(14, dtype=np.float32)
        env = _TaskEnv(qpos)
        deploy_policy.eval(env, model, env.get_obs())
        expected = qpos.copy()
        expected[[6, 13]] = 0.0
        np.testing.assert_array_equal(env.qpos, expected)


if __name__ == "__main__":
    unittest.main()
