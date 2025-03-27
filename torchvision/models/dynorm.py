from typing import Callable

import torch
from torch import nn


class DyT(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        x = torch.tanh(self.alpha * x)
        return gamma * x + beta


class DyTV(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        x = torch.tanh(alpha * x)
        return gamma * x + beta


class DyTVMC(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.mu = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        mu = self.mu.view(1, -1, 1, 1)

        x = torch.tanh(alpha * (x - mu))
        return gamma * x + beta


class DyAS(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.mu = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        mu = self.mu.view(1, -1, 1, 1)

        x = alpha * (x - mu)
        x = x / torch.sqrt(1 + x ** 2)
        return gamma * x + beta


def get_norm_layer(norm_type: str) -> Callable[[int], nn.Module]:
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'dyt':
        return DyT
    elif norm_type == 'dytv':
        return DyTV
    elif norm_type == 'dytvmc':
        return DyTVMC
    elif norm_type == 'dyas':
        return DyAS
    else:
        raise ValueError(f'Unknown normalization type {norm_type}.')
