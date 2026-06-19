from __future__ import annotations

import torch
from torch import nn


class ActionHeadMVP(nn.Module):
    """Small image/proprio/task-conditioned 7D behavior-cloning policy."""

    def __init__(
        self,
        num_tasks: int,
        proprio_dim: int = 9,
        action_dim: int = 7,
        task_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.task_embedding = nn.Embedding(max(1, self.num_tasks), task_dim)
        self.head = nn.Sequential(
            nn.Linear(128 + self.proprio_dim + task_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.action_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, proprio: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        task_feat = self.task_embedding(task_id)
        x = torch.cat([image_feat, proprio, task_feat], dim=-1)
        return self.head(x)
