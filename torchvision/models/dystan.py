import torch
from torch import nn


class DyStan(nn.Module):
    """
    Dynamic Standardization (DyStan): A fully adaptive, per-channel dynamic normalization
    intended as a drop-in replacement for nn.BatchNorm2d.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.mu = nn.Parameter(torch.zeros(num_features))
        self.sigma = nn.Parameter(torch.ones(num_features))
        self.eps = eps
        # self.step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        mu = self.mu.view(1, -1, 1, 1)
        sigma = self.sigma.view(1, -1, 1, 1)

        x = x - mu
        x = x / torch.sqrt(sigma ** 2 + x ** 2 + self.eps)
        output = gamma * x + beta

        # Log every 100 steps
        # if self.training and self.step % 100 == 0:
        #     print(f"Step {self.step}:")
        #     print(f"  gamma mean: {self.gamma.mean().item():.4f}, beta mean: {self.beta.mean().item():.4f}")
        #     print(f"  sigma mean: {self.sigma.mean().item():.4f}, mu mean: {self.mu.mean().item():.4f}")
        # self.step += 1
        return output
