import torch
from torch import nn


class DyAS(nn.Module):
    """
    Dynamic Algebraic Sigmoid (DyAS): A fully adaptive, per-channel dynamic normalization
    intended as a drop-in replacement for nn.BatchNorm2d.
    """

    def __init__(
        self,
        num_features: int,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        alpha = self.alpha.view(1, -1, 1, 1)

        x = x / torch.sqrt(1 + (alpha * x) ** 2)
        x = gamma * x + beta
        return x
