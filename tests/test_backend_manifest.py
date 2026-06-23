import json
import tempfile
import unittest
from pathlib import Path

from lightewm.runner.backend_result import BackendRunResult, read_backend_manifest


class BackendManifestTest(unittest.TestCase):
    def test_round_trip_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = BackendRunResult(
                backend="wan_ti2v",
                task="infer",
                generated_dir=tmp,
                metadata_path="data/metadata.csv",
                dataset_base_path="data",
                fps=10,
                num_frames=49,
                extra={"runner": "example.Runner"},
            )
            path = result.write_manifest()
            self.assertEqual(path, Path(tmp) / "backend_manifest.json")
            payload = json.loads(path.read_text())
            self.assertEqual(payload["backend"], "wan_ti2v")
            self.assertEqual(payload["artifact_type"], "video")
            loaded = read_backend_manifest(path)
            self.assertEqual(loaded.backend, "wan_ti2v")
            self.assertEqual(loaded.extra["runner"], "example.Runner")


if __name__ == "__main__":
    unittest.main()
