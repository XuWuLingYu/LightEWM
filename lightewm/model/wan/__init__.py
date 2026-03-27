"""WAN 1.3B I2V model components."""

from .infer_module import WanI2VInferModel
from .training_module import WanTrainingModule

__all__ = [
    "WanTrainingModule",
    "WanI2VInferModel",
]
