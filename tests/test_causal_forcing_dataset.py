import json
import tempfile
import unittest
from pathlib import Path

from lightewm.dataset.causal_forcing import write_causal_forcing_jsonl
from lightewm.eval.utils import collect_video_pairs


class CausalForcingJsonlTest(unittest.TestCase):
    def test_filter_preserves_source_row_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata_path = root / "metadata.json"
            output_path = root / "filtered.jsonl"
            metadata_path.write_text(json.dumps([
                {
                    "video": "videos/first.mp4",
                    "prompt": "first",
                    "camera_key": "agentview",
                    "demo_id": "demo_0",
                },
                {
                    "video": "videos/second.mp4",
                    "prompt": "second",
                    "camera_key": "robot0_eye_in_hand",
                    "demo_id": "demo_1",
                },
            ]))

            count = write_causal_forcing_jsonl(
                base_path=str(root),
                metadata_path=str(metadata_path),
                output_path=str(output_path),
                filter_key="camera_key",
                filter_value="robot0_eye_in_hand",
            )

            self.assertEqual(count, 1)
            row = json.loads(output_path.read_text().strip())
            self.assertEqual(row["row_id"], 1)
            self.assertEqual(row["demo_id"], "demo_1")
            self.assertEqual(row["camera_key"], "robot0_eye_in_hand")

    def test_eval_uses_preserved_row_id_for_generated_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real_video = root / "real.mp4"
            real_video.write_bytes(b"not-a-real-video-but-path-exists")
            metadata_path = root / "filtered.jsonl"
            generated_dir = root / "generated"
            generated_dir.mkdir()
            metadata_path.write_text(json.dumps({
                "video_path": str(real_video),
                "prompt": "second",
                "row_id": 1,
                "demo_id": "demo_1",
                "camera_key": "robot0_eye_in_hand",
            }) + "\n")
            generated = generated_dir / "000001__demo_1__robot0_eye_in_hand.mp4"
            generated.write_bytes(b"generated")

            pairs, missing = collect_video_pairs(
                metadata_path=str(metadata_path),
                dataset_base_path=str(root),
                generated_dir=str(generated_dir),
                video_key="video_path",
            )

            self.assertEqual(missing, [])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].row_id, 1)
            self.assertEqual(pairs[0].generated_path, str(generated))


if __name__ == "__main__":
    unittest.main()
