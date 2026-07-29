from __future__ import annotations

__all__ = ["evaluate_video_quality"]


def evaluate_video_quality(*args, **kwargs):
    """Load optional video metric dependencies only when the evaluator is used."""
    from .video_quality import evaluate_video_quality as _evaluate_video_quality

    return _evaluate_video_quality(*args, **kwargs)
