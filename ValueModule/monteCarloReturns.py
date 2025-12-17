import torch
import torch.nn as nn
import Buffers.types as Batch

class MonteCarloReturn(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    @torch.no_grad()
    def forward(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        """
        Returns:
            returns: [B, T]
        """
        rewards = batch.rewards
        dones = batch.dones
        B, T = rewards.shape
        returns = torch.zeros_like(rewards)

        G = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

        for t in reversed(range(T)):
            G = rewards[:, t] + self.gamma * G * (1.0 - dones[:, t])
            returns[:, t] = G

        return returns
