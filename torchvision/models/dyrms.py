import torch
from torch import nn


class DyRMS(nn.Module):
    """
    Dynamic RMS (DyRMS): A fully adaptive, per-channel dynamic normalization
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
        self.sigma = nn.Parameter(torch.ones(num_features) * 2.0)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape parameters for broadcasting (N, C, H, W)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        sigma = self.sigma.view(1, -1, 1, 1)

        x = x / torch.sqrt(sigma ** 2 + x ** 2 + self.eps)
        return gamma * x + beta
