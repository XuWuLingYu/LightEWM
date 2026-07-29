import unittest
from pathlib import Path
import tempfile

from scripts.download_public_gcs_prefix import (
    file_md5_base64,
    media_url,
    relative_object_path,
    validate_object,
)


class PublicGcsPrefixTest(unittest.TestCase):
    def test_relative_object_path_strips_prefix(self):
        self.assertEqual(
            relative_object_path(
                "checkpoint/params/shard",
                "checkpoint/",
            ),
            Path("params/shard"),
        )

    def test_relative_object_path_rejects_traversal(self):
        with self.assertRaisesRegex(ValueError, "unsafe"):
            relative_object_path("checkpoint/../secret", "checkpoint/")

    def test_media_url_pins_generation(self):
        url = media_url(
            "bucket",
            {"name": "path/with space", "generation": "123"},
        )
        self.assertIn("path%2Fwith%20space", url)
        self.assertIn("generation=123", url)

    def test_validate_object_checks_size_and_md5(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "object"
            path.write_bytes(b"checkpoint")
            item = {
                "name": "prefix/object",
                "size": len(b"checkpoint"),
                "md5Hash": file_md5_base64(path),
            }
            validate_object(path, item)
            item["md5Hash"] = "invalid"
            with self.assertRaisesRegex(IOError, "MD5 mismatch"):
                validate_object(path, item)


if __name__ == "__main__":
    unittest.main()
