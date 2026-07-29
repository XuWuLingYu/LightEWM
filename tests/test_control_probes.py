import unittest

from lightewm.eval.closed_loop.gates import (
    analyze_control_probe,
    analyze_gripper_toggle_probe,
)


class ControlProbeTest(unittest.TestCase):
    def test_hold_probe_requires_no_success_and_stationary_qpos(self):
        payload = {
            "action_mode": "hold",
            "success": False,
            "initial": {
                "qpos": [0.1, 0.2],
                "objects": {"block": [0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0]},
            },
            "final": {
                "qpos": [0.10001, 0.2],
                "objects": {"block": [0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0]},
            },
        }
        result = analyze_control_probe(payload, qpos_tolerance=1e-3)
        self.assertTrue(result["passed"])
        self.assertFalse(result["qpos_motion_detected"])
        self.assertTrue(result["object_state_observed"])

    def test_active_negative_requires_qpos_motion_and_failure(self):
        payload = {
            "action_mode": "zero",
            "success": False,
            "initial": {
                "qpos": [0.4, -0.2],
                "objects": {"block": [0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0]},
            },
            "final": {
                "qpos": [0.0, 0.0],
                "objects": {"block": [0.01, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0]},
            },
        }
        result = analyze_control_probe(payload, qpos_tolerance=1e-3)
        self.assertTrue(result["passed"])
        self.assertTrue(result["qpos_motion_detected"])
        self.assertGreater(result["object_max_abs_delta"], 0.0)

    def test_negative_probe_fails_if_success_condition_fires(self):
        payload = {
            "action_mode": "zero",
            "success": True,
            "initial": {"qpos": [0.4], "objects": {"block": [0.0]}},
            "final": {"qpos": [0.0], "objects": {"block": [0.0]}},
        }
        result = analyze_control_probe(payload)
        self.assertFalse(result["passed"])
        self.assertIn("unexpected_success", result["failures"])

    def test_gripper_toggle_requires_command_and_observed_motion(self):
        payload = {
            "action_mode": "gripper_toggle",
            "commanded_gripper": [0.0, 0.785398163, 0.0],
            "observed_gripper": [0.02, 0.72, 0.04],
        }
        result = analyze_gripper_toggle_probe(payload, gripper_tolerance=0.1)
        self.assertTrue(result["passed"])
        self.assertGreater(result["commanded_range"], 0.7)
        self.assertGreater(result["observed_range"], 0.6)

    def test_gripper_toggle_fails_without_observed_motion(self):
        payload = {
            "action_mode": "gripper_toggle",
            "commanded_gripper": [0.0, 0.785398163],
            "observed_gripper": [0.2, 0.21],
        }
        result = analyze_gripper_toggle_probe(payload, gripper_tolerance=0.1)
        self.assertFalse(result["passed"])
        self.assertIn("observed_gripper_did_not_move", result["failures"])


if __name__ == "__main__":
    unittest.main()
