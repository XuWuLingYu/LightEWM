import unittest

from scripts.normalize_closed_loop_results import parse_source_revisions


class NormalizeClosedLoopResultsTest(unittest.TestCase):
    def test_parse_source_revisions(self):
        self.assertEqual(
            parse_source_revisions(["robolab=abc123", "openpi=def456"]),
            {"robolab": "abc123", "openpi": "def456"},
        )

    def test_parse_source_revisions_rejects_missing_revision(self):
        with self.assertRaisesRegex(ValueError, "NAME=REVISION"):
            parse_source_revisions(["openpi="])


if __name__ == "__main__":
    unittest.main()
