import tempfile
import unittest
from pathlib import Path

from scripts.export_uv_lock_wheels import (
    export_aria2_manifest,
    is_compatible_wheel,
)


class ExportUvLockWheelsTest(unittest.TestCase):
    def test_compatibility_filter(self):
        self.assertTrue(
            is_compatible_wheel(
                "torch-2.7.1-cp311-cp311-manylinux_2_28_x86_64.whl",
                python_tag="cp311",
            )
        )
        self.assertTrue(
            is_compatible_wheel(
                "opencv-1.0-cp39-abi3-manylinux2014_x86_64.whl",
                python_tag="cp311",
            )
        )
        self.assertTrue(
            is_compatible_wheel("typing-1.0-py3-none-any.whl", python_tag="cp311")
        )
        self.assertFalse(
            is_compatible_wheel(
                "torch-2.7.1-cp312-cp312-manylinux_2_28_x86_64.whl",
                python_tag="cp311",
            )
        )
        self.assertFalse(
            is_compatible_wheel(
                "torch-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl",
                python_tag="cp311",
            )
        )
        self.assertFalse(
            is_compatible_wheel(
                "aiohttp-3.12.4-cp311-cp311-musllinux_1_2_x86_64.whl",
                python_tag="cp311",
            )
        )

    def test_manifest_preserves_url_hash_and_size(self):
        lock = """
version = 1

[[package]]
name = "demo"
version = "1.0"
wheels = [
  { url = "https://example.test/demo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", size = 2048 },
  { url = "https://example.test/demo-1.0-cp312-cp312-manylinux_2_28_x86_64.whl", hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", size = 2048 },
]
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / "uv.lock"
            output_path = root / "wheels.txt"
            lock_path.write_text(lock, encoding="utf-8")
            count, size = export_aria2_manifest(
                lock_path,
                output_path,
                python_tag="cp311",
                min_size=1024,
            )
            output = output_path.read_text(encoding="utf-8")
        self.assertEqual(count, 1)
        self.assertEqual(size, 2048)
        self.assertIn("demo-1.0-cp311-cp311", output)
        self.assertIn("checksum=sha-256=" + "a" * 64, output)
        self.assertNotIn("cp312", output)

    def test_manifest_prefers_newer_manylinux_tag(self):
        lock = """
version = 1

[[package]]
name = "cmake"
version = "1.0"
wheels = [
  { url = "https://example.test/cmake-1.0-py3-none-manylinux_2_12_x86_64.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", size = 2048 },
  { url = "https://example.test/cmake-1.0-py3-none-manylinux_2_17_x86_64.whl", hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", size = 2048 },
]
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / "uv.lock"
            output_path = root / "wheels.txt"
            lock_path.write_text(lock, encoding="utf-8")
            export_aria2_manifest(
                lock_path,
                output_path,
                python_tag="cp311",
                min_size=1024,
            )
            output = output_path.read_text(encoding="utf-8")
        self.assertIn("manylinux_2_17", output)
        self.assertNotIn("manylinux_2_12", output)

    def test_manifest_only_exports_resolved_package_versions(self):
        lock = """
version = 1

[[package]]
name = "nvidia-cublas-cu12"
version = "12.6.4.1"
wheels = [
  { url = "https://example.test/nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.whl", hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", size = 2048 },
]

[[package]]
name = "nvidia-cublas-cu12"
version = "12.9.0.13"
wheels = [
  { url = "https://example.test/nvidia_cublas_cu12-12.9.0.13-py3-none-manylinux_2_27_x86_64.whl", hash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", size = 2048 },
]
"""
        requirements = """
nvidia-cublas-cu12==12.6.4.1
colorama==0.4.6 ; sys_platform == "win32"
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_path = root / "uv.lock"
            requirements_path = root / "requirements.txt"
            output_path = root / "wheels.txt"
            lock_path.write_text(lock, encoding="utf-8")
            requirements_path.write_text(requirements, encoding="utf-8")
            export_aria2_manifest(
                lock_path,
                output_path,
                python_tag="cp311",
                min_size=1024,
                requirements_path=requirements_path,
                url_rewrites=(
                    (
                        "https://example.test",
                        "https://mirror.example.test/pypi",
                    ),
                ),
            )
            output = output_path.read_text(encoding="utf-8")
        self.assertIn("12.6.4.1", output)
        self.assertNotIn("12.9.0.13", output)
        self.assertIn("https://mirror.example.test/pypi/nvidia_cublas", output)
        self.assertNotIn("https://example.test/", output)


if __name__ == "__main__":
    unittest.main()
