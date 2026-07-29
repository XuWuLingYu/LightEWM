import re
import tempfile
import unittest
from pathlib import Path

from lightewm.eval.closed_loop.sampling import (
    CorrectnessSample,
    SampleCase,
    build_sample_step,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = (
    REPO_ROOT / "examples" / "closed_loop" / "samples" / "correctness_10pct.yaml"
)
ROBOLAB_ASSET_MANIFEST = (
    REPO_ROOT / "env" / "robolab_correctness_sample_assets.sha256"
)


class CorrectnessSamplingTest(unittest.TestCase):
    def test_checked_in_sample_has_exact_ten_percent_counts(self):
        sample = CorrectnessSample.from_yaml(SAMPLE_PATH)

        self.assertEqual(sample.fraction, 0.10)
        self.assertEqual(len(sample.benchmarks["robotwin"].cases), 5)
        self.assertEqual(len(sample.benchmarks["robolab"].cases), 12)
        self.assertEqual(
            sample.benchmarks["robotwin"].repository_revision,
            "13c3c47ff4312dd62484bcd51be034af55c062d1",
        )
        self.assertEqual(
            sample.benchmarks["robolab"].repository_revision,
            "0aef241fb088ca21bb4ebd24448940ed56620d17",
        )
        self.assertTrue(
            all(
                case.setting == "lightewm_correctness_10pct"
                for case in sample.benchmarks["robotwin"].cases
            )
        )
        self.assertEqual(
            sample.benchmarks["robolab"].cases[0].gate,
            "recorded-replay",
        )

    def test_duplicate_task_is_rejected(self):
        payload = """
schema_version: 1
fraction: 0.5
benchmarks:
  robotwin:
    population: 4
    cases:
      - {task: task_a, gate: ground-truth}
      - {task: task_a, gate: ground-truth}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.yaml"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                CorrectnessSample.from_yaml(path)

    def test_wrong_sample_count_is_rejected(self):
        payload = """
schema_version: 1
fraction: 0.1
benchmarks:
  robolab:
    population: 120
    cases:
      - {task: only_one, gate: deterministic-hold}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.yaml"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 12"):
                CorrectnessSample.from_yaml(path)

    def test_malformed_repository_revision_is_rejected(self):
        payload = """
schema_version: 1
fraction: 1.0
benchmarks:
  robotwin:
    population: 1
    repository_revision: main
    cases:
      - {task: task_a, gate: ground-truth}
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.yaml"
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "40-character lowercase SHA"):
                CorrectnessSample.from_yaml(path)

    def test_robolab_asset_manifest_is_checked_and_complete_for_sample_scenes(self):
        entries = []
        for line in ROBOLAB_ASSET_MANIFEST.read_text(encoding="utf-8").splitlines():
            digest, path = line.split("  ", 1)
            self.assertRegex(digest, re.compile(r"^[0-9a-f]{64}$"))
            self.assertTrue(path.startswith("assets/"))
            entries.append(path)

        self.assertEqual(len(entries), 56)
        self.assertEqual(entries, sorted(entries))
        self.assertEqual(len(entries), len(set(entries)))
        self.assertTrue(
            {
                "assets/scenes/banana_bowl.usda",
                "assets/scenes/butter_raisin_box_grey_bin.usda",
                "assets/scenes/cartons_in_vertical_crate.usda",
                "assets/scenes/colored_blocks.usda",
                "assets/scenes/mugs4_measuringcup_drill_bowl_v2.usda",
                "assets/scenes/rubiks_cube_banana_bowl.usda",
                "assets/scenes/tools_container.usda",
                "assets/scenes/tools_picking.usda",
                "assets/scenes/wire_shelf_mugs_plate_spatula.usda",
            }.issubset(entries)
        )

    def test_builds_official_robotwin_ground_truth_step(self):
        step = build_sample_step(
            "robotwin",
            SampleCase(
                task="adjust_bottle",
                gate="ground-truth",
                setting="lightewm_demo_clean_gate",
            ),
            official_root="/opt/RoboTwin",
            python="/envs/robotwin/bin/python",
        )

        self.assertEqual(
            step.argv,
            (
                "/envs/robotwin/bin/python",
                "script/collect_data.py",
                "adjust_bottle",
                "lightewm_demo_clean_gate",
            ),
        )
        self.assertIn("success!", step.required_log_patterns)

    def test_builds_official_robolab_hold_step(self):
        step = build_sample_step(
            "robolab",
            SampleCase(
                task="BananaInBowlTask",
                gate="deterministic-hold",
                steps=25,
            ),
            official_root="/opt/RoboLab",
            python="/envs/robolab/bin/python",
        )

        self.assertIn("examples/run_empty.py", step.argv)
        self.assertIn("--action-mode", step.argv)
        self.assertIn("hold", step.argv)
        self.assertIn("25", step.argv)


if __name__ == "__main__":
    unittest.main()
