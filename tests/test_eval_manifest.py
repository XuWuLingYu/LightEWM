import json
import tempfile
import unittest
from pathlib import Path

from lightewm.eval.runner import parse_backend_manifests


class EvalManifestParsingTest(unittest.TestCase):
    def test_explicit_manifest_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "backend_manifest.json"
            manifest_path.write_text(json.dumps({
                "backend": "causal_forcing",
                "task": "infer",
                "generated_dir": "logs/example",
                "artifact_type": "video",
                "metadata_path": "data/metadata.csv",
                "dataset_base_path": "data",
            }))
            parsed = parse_backend_manifests([f"causal={manifest_path}"])
            path, manifest = parsed["causal"]
            self.assertEqual(path, manifest_path)
            self.assertEqual(manifest.backend, "causal_forcing")
            self.assertEqual(manifest.artifact_type, "video")


if __name__ == "__main__":
    unittest.main()
