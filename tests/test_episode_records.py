import json
import tempfile
import unittest
from pathlib import Path

from lightewm.eval.closed_loop.records import (
    ArtifactIndex,
    ArtifactRef,
    EpisodeRecord,
    read_episode_records,
    summarize_episodes,
    write_episode_records,
)


class EpisodeRecordTest(unittest.TestCase):
    def test_artifact_index_writes_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact_index.json"
            index = ArtifactIndex(
                run_id="run-1",
                policy_spec="policy.yaml",
                policy_spec_hash="a" * 64,
                benchmark_spec="benchmark.yaml",
                benchmark_spec_hash="b" * 64,
                command_log="commands",
                native_output_dir="native",
                episodes_path="episodes.jsonl",
                summary_path="summary.json",
                source_revisions={"robolab": "0aef241"},
            )
            self.assertEqual(index.write(path), path)
            payload = json.loads(path.read_text())
        self.assertEqual(payload["source_revisions"]["robolab"], "0aef241")

    def test_jsonl_round_trip_and_micro_summary(self):
        records = [
            EpisodeRecord(
                benchmark="robotwin",
                benchmark_revision="13c3c47",
                protocol="reference-smoke-v1",
                policy="starwam-robotwin",
                task="adjust_bottle",
                setting="demo_clean",
                seed=100000 + index,
                episode_id=str(index),
                success=index == 0,
                status="completed",
                steps=50 + index,
                metrics={"score": float(index == 0)},
                artifacts=(ArtifactRef(kind="video", path=f"episode{index}.mp4"),),
            )
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            write_episode_records(path, records)
            loaded = read_episode_records(path)
            self.assertEqual(loaded, records)
            summary = summarize_episodes(loaded)
        self.assertEqual(summary["total_episodes"], 2)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["by_setting"]["demo_clean"]["success_rate"], 0.5)

    def test_unknown_success_is_not_counted_as_evaluated(self):
        record = EpisodeRecord(
            benchmark="robolab",
            benchmark_revision="0aef241",
            protocol="environment-smoke-v1",
            policy="none",
            task="BananaInBowlTask",
            setting="default",
            seed=0,
            episode_id="render-only",
            success=None,
            status="completed",
        )
        summary = summarize_episodes([record])
        self.assertEqual(summary["evaluated_episodes"], 0)
        self.assertIsNone(summary["success_rate"])
        self.assertIsNone(json.loads(record.to_json())["success"])


if __name__ == "__main__":
    unittest.main()
