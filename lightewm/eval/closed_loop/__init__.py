"""Contracts and native-output adapters for closed-loop robot benchmarks."""

from .gates import analyze_control_probe, analyze_episode_gate
from .records import ArtifactIndex, ArtifactRef, EpisodeRecord, summarize_episodes
from .specs import BenchmarkSpec, PolicySpec, TensorSpec

__all__ = [
    "ArtifactIndex",
    "ArtifactRef",
    "analyze_control_probe",
    "analyze_episode_gate",
    "BenchmarkSpec",
    "EpisodeRecord",
    "PolicySpec",
    "TensorSpec",
    "summarize_episodes",
]
