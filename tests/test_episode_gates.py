import unittest

from lightewm.eval.closed_loop.gates import analyze_episode_gate
from lightewm.eval.closed_loop.records import EpisodeRecord


def _record(episode_id: str, success: bool | None) -> EpisodeRecord:
    return EpisodeRecord(
        benchmark="robolab",
        benchmark_revision="revision",
        protocol="test",
        policy="policy",
        task="task",
        setting="default",
        seed=0,
        episode_id=episode_id,
        success=success,
        status="completed",
    )


class EpisodeGateTest(unittest.TestCase):
    def test_ground_truth_requires_all_success(self):
        result = analyze_episode_gate(
            [_record("0", True), _record("1", True)],
            expectation="all-success",
            min_episodes=2,
        )
        self.assertTrue(result["passed"])

    def test_negative_rejects_any_success(self):
        result = analyze_episode_gate(
            [_record("0", False), _record("1", True)],
            expectation="all-failure",
            min_episodes=2,
        )
        self.assertFalse(result["passed"])
        self.assertIn("unexpected_success", result["failures"])

    def test_reference_requires_minimum_and_one_success(self):
        result = analyze_episode_gate(
            [_record("0", False), _record("1", True)],
            expectation="at-least-one-success",
            min_episodes=3,
        )
        self.assertFalse(result["passed"])
        self.assertIn("insufficient_evaluated_episodes", result["failures"])

    def test_unknown_success_is_not_accepted(self):
        result = analyze_episode_gate(
            [_record("0", None)],
            expectation="all-failure",
        )
        self.assertFalse(result["passed"])
        self.assertIn("unknown_success_status", result["failures"])


if __name__ == "__main__":
    unittest.main()
