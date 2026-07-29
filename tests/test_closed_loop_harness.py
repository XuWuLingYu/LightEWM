import json
import sys
import tempfile
import unittest
from pathlib import Path

from lightewm.eval.closed_loop.harness import CommandStep, OfficialHarness
from lightewm.eval.closed_loop.protocols import RoboLabProtocol, RoboTwinProtocol


class ClosedLoopHarnessTest(unittest.TestCase):
    def test_command_step_records_command_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = CommandStep(
                name="probe",
                argv=(
                    sys.executable,
                    "-c",
                    "from pathlib import Path; "
                    "Path('native.txt').write_text('ok'); print('success=true')",
                ),
                cwd=str(root),
                expected_output=("native.txt",),
            )
            result = OfficialHarness(root / "evidence").run(step)
            self.assertEqual(result.returncode, 0)
            self.assertIn("success=true", Path(result.log_path).read_text())
            command = json.loads(Path(result.command_path).read_text())
            self.assertEqual(command["argv"], list(step.argv))

    def test_command_step_enforces_required_and_forbidden_log_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step = CommandStep(
                name="render_probe",
                argv=(sys.executable, "-c", "print('Render Error')"),
                cwd=str(root),
                required_log_patterns=("Render Well",),
                forbidden_log_patterns=("Render Error",),
            )
            with self.assertRaisesRegex(RuntimeError, "log validation failed"):
                OfficialHarness(root / "evidence").run(step)
            result = json.loads(
                (root / "evidence" / "render_probe" / "result.json").read_text()
            )
            self.assertEqual(result["missing_log_patterns"], ["Render Well"])
            self.assertEqual(result["forbidden_log_patterns_found"], ["Render Error"])

    def test_robotwin_protocol_uses_official_entrypoints(self):
        protocol = RoboTwinProtocol(
            root="/opt/RoboTwin",
            python="/envs/robotwin/bin/python",
        )
        render = protocol.render()
        self.assertEqual(render.argv[-1], "script/test_render.py")
        self.assertEqual(render.required_log_patterns, ("Render Well",))
        self.assertEqual(render.forbidden_log_patterns, ("Render Error",))
        collect = protocol.collect("adjust_bottle", "demo_clean")
        self.assertEqual(
            collect.argv[-3:], ("script/collect_data.py", "adjust_bottle", "demo_clean")
        )
        evaluate = protocol.evaluate(
            config="policy/starwam_client/deploy_policy_client.yml",
            policy_name="starwam_client",
            task="adjust_bottle",
            setting="demo_clean",
            checkpoint_tag="starwam27500",
            seed=0,
            episodes=5,
        )
        test_num_index = evaluate.argv.index("test_num")
        self.assertEqual(evaluate.argv[test_num_index + 1], "5")
        negative = protocol.negative_control(
            config="policy/lightewm_control/deploy_policy.yml",
            task="adjust_bottle",
            setting="demo_clean",
            seed=0,
            mode="wrong_gripper",
            probe_path="/evidence/probe.json",
        )
        self.assertIn("lightewm_control", negative.argv)
        self.assertIn("wrong_gripper", negative.argv)
        self.assertIn("/evidence/probe.json", negative.argv)
        self.assertEqual(negative.expected_output, ("/evidence/probe.json",))
        self.assertIn("Fail!", negative.required_log_patterns)
        self.assertIn("Success!", negative.forbidden_log_patterns)

    def test_robolab_protocol_uses_official_entrypoints(self):
        protocol = RoboLabProtocol(
            root="/opt/RoboLab",
            python="/envs/robolab/bin/python",
        )
        replay = protocol.recorded_replay("RubiksCubeTask")
        self.assertIn("examples/run_recorded.py", replay.argv)
        self.assertIn("--validate-states", replay.argv)
        self.assertIn("--disable-subtask", replay.argv)
        self.assertEqual(replay.env["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertIn("Restored recorded env config", replay.required_log_patterns)
        self.assertIn(
            "Traceback (most recent call last):",
            replay.forbidden_log_patterns,
        )
        self.assertIn("WARNING: no recorded initial state", replay.forbidden_log_patterns)
        self.assertIn(
            "STATE VALIDATION: replay diverged",
            replay.forbidden_log_patterns,
        )
        hold = protocol.deterministic_hold("RubiksCubeTask", steps=25)
        self.assertIn("examples/run_empty.py", hold.argv)
        self.assertIn("hold", hold.argv)
        self.assertIn("25", hold.argv)
        self.assertEqual(hold.env["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertEqual(
            hold.expected_output,
            (
                "output/run_empty_env/RubiksCubeTask/control_probe_0.json",
                "output/run_empty_env/episode_results.jsonl",
            ),
        )
        reference = protocol.reference_policy(
            "RubiksCubeTask",
            num_runs=5,
            output_folder_name="lightewm_reference_pi05",
        )
        self.assertIn("--output-folder-name", reference.argv)
        self.assertIn("lightewm_reference_pi05", reference.argv)
        self.assertEqual(reference.env["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertEqual(
            reference.expected_output,
            ("output/lightewm_reference_pi05/episode_results.jsonl",),
        )


if __name__ == "__main__":
    unittest.main()
