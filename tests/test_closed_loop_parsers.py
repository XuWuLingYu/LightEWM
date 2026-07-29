import json
import tempfile
import unittest
from pathlib import Path

from lightewm.eval.closed_loop.parsers import (
    parse_libero_results,
    parse_robolab_episode_results,
    parse_robotwin_collection_log,
    parse_robotwin_progress_log,
)


class ClosedLoopParserTest(unittest.TestCase):
    def test_robotwin_progress_log_to_episode_records(self):
        log = """
Success!
Success rate: 1/1 => 100.0%, current seed: 100000
Fail!
Success rate: 1/2 => 50.0%, current seed: 100001
"""
        records = parse_robotwin_progress_log(
            log,
            revision="13c3c47",
            policy="starwam",
            task="adjust_bottle",
            setting="demo_clean",
        )
        self.assertEqual([record.success for record in records], [True, False])
        self.assertEqual([record.seed for record in records], [100000, 100001])

    def test_robotwin_progress_log_attaches_native_probe_and_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp)
            (artifact_root / "episode0.json").write_text("{}", encoding="utf-8")
            (artifact_root / "episode0.mp4").write_bytes(b"video")
            records = parse_robotwin_progress_log(
                "Success rate: 0/1 => 0.0%, current seed: 0\n",
                revision="13c3c47",
                policy="deterministic-control",
                task="adjust_bottle",
                setting="hold",
                artifacts_dir=artifact_root,
            )
        self.assertEqual(
            [artifact.kind for artifact in records[0].artifacts],
            ["event_log", "video"],
        )

    def test_robotwin_collection_log_preserves_success_and_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact_root = Path(tmp)
            trajectory = artifact_root / "adjust_bottle" / "episode0.hdf5"
            trajectory.parent.mkdir()
            trajectory.write_bytes(b"trajectory")
            records = parse_robotwin_collection_log(
                "simulate data episode 0 success! (seed = 17)\n",
                revision="13c3c47",
                policy="scripted-expert",
                task="adjust_bottle",
                setting="lightewm_demo_clean_gate",
                artifacts_dir=artifact_root,
            )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].success)
        self.assertEqual(records[0].seed, 17)
        self.assertTrue(records[0].source["plan_success"])
        self.assertEqual(records[0].artifacts[0].kind, "hdf5")

    def test_robolab_native_jsonl_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "episode_results.jsonl"
            native.write_text(
                json.dumps(
                    {
                        "env_name": "RubiksCubeTaskHomeOffice",
                        "run": 7,
                        "env_id": 0,
                        "success": True,
                        "episode_step": 42,
                        "score": 1.0,
                        "reason": None,
                        "instruction": "Put the cube in the bowl.",
                        "metrics": {"ee_path_length": 0.42},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task_dir = root / "RubiksCubeTaskHomeOffice"
            task_dir.mkdir()
            (task_dir / "log_7_env0.json").write_text("{}", encoding="utf-8")
            (task_dir / "run_7.hdf5").write_bytes(b"hdf5")
            (task_dir / "Put_the_cube_in_the_bowl_7.mp4").write_bytes(b"video")
            (task_dir / "Put_the_cube_in_the_bowl_7_viewport.mp4").write_bytes(
                b"viewport"
            )
            records = parse_robolab_episode_results(
                native,
                revision="0aef241",
                policy="recorded-replay",
                protocol="ground-truth-v1",
            )
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].success)
        self.assertEqual(records[0].steps, 42)
        self.assertEqual(records[0].episode_id, "7")
        self.assertEqual(records[0].metrics["ee_path_length"], 0.42)
        self.assertEqual(records[0].source["env_id"], 0)
        self.assertEqual(
            [artifact.kind for artifact in records[0].artifacts],
            ["event_log", "hdf5", "video", "viewport_video"],
        )

    def test_libero_suite_summary_expands_task_counts(self):
        payload = {
            "checkpoint": "openvla-oft",
            "task_suite_name": "libero_spatial",
            "task_results": [
                {"task_id": 0, "task_name": "pick block", "successes": 2, "trials": 3},
                {"task_id": 1, "task_name": "place block", "successes": 1, "trials": 2},
            ],
            "total_successes": 3,
            "total_trials": 5,
            "success_rate": 0.6,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = parse_libero_results(
                path,
                revision="libero-local",
                policy="openvla-oft",
                protocol="compatibility-v1",
            )
        self.assertEqual(len(records), 5)
        self.assertEqual(sum(record.success for record in records), 3)
        self.assertEqual(records[0].setting, "libero_spatial")

    def test_libero_episode_csv_preserves_real_case_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "suite": "libero_10",
                        "task": "turn on stove",
                        "num_cases": 2,
                        "successes": 1,
                    }
                ),
                encoding="utf-8",
            )
            csv_path = root / "openvla_results.csv"
            csv_path.write_text(
                "case_id,demo_id,success,num_replay_images,init_source\n"
                "0,demo_0,1,220,hdf5\n"
                "1,demo_1,0,300,hdf5\n",
                encoding="utf-8",
            )
            records = parse_libero_results(
                csv_path,
                revision="libero-local",
                policy="openvla-oft",
                protocol="compatibility-v1",
            )
        self.assertEqual([record.success for record in records], [True, False])
        self.assertEqual([record.episode_id for record in records], ["0", "1"])
        self.assertEqual(records[0].task, "turn on stove")
        self.assertEqual(records[0].setting, "libero_10")
        self.assertEqual(records[0].steps, 220)

    def test_fastwam_manager_summary_expands_suite_task_results(self):
        payload = {
            "run_id": "fastwam_40case_step_043400",
            "ckpt": "step_043400.pt",
            "config": "libero_joint_2cam224_1e-4",
            "suite_stats": {
                "libero_spatial": {
                    "total_tasks": 2,
                    "total_trials": 2,
                    "total_successes": 1,
                },
                "libero_10": {
                    "total_tasks": 1,
                    "total_trials": 1,
                    "total_successes": 1,
                },
            },
            "task_results": {
                "libero_spatial_0": {
                    "total_episodes": 1,
                    "successes": 1,
                    "task_description": "pick up the bowl",
                },
                "libero_spatial_1": {
                    "total_episodes": 1,
                    "successes": 0,
                    "task_description": "place the bowl",
                },
                "libero_10_0": {
                    "total_episodes": 1,
                    "successes": 1,
                    "task_description": "turn on the stove",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = parse_libero_results(
                path,
                revision="libero-local",
                policy="fastwam-step-043400",
                protocol="compatibility-v1",
            )
        self.assertEqual(len(records), 3)
        self.assertEqual(sum(record.success for record in records), 2)
        self.assertEqual(
            [record.setting for record in records],
            ["libero_spatial", "libero_spatial", "libero_10"],
        )
        self.assertEqual(records[2].task, "turn on the stove")
        self.assertEqual(records[0].source["native_task_key"], "libero_spatial_0")


if __name__ == "__main__":
    unittest.main()
