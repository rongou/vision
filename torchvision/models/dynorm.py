from typing import Callable

import torch
from torch import nn


class DyT(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)

        out = gamma * torch.tanh(self.alpha * x) + beta
        return out


class DyTV(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(num_features) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        alpha = self.alpha.view(1, -1, 1, 1)

        out = gamma * torch.tanh(alpha * x) + beta
        return out


class DyTVMC(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(num_features) * 0.5)
        self.mu = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        alpha = self.alpha.view(1, -1, 1, 1)
        mu = self.mu.view(1, -1, 1, 1)

        out = gamma * torch.tanh(alpha * (x - mu)) + beta
        return out


class DyAS(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.ones(num_features) * 0.5)
        self.mu = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        alpha = self.alpha.view(1, -1, 1, 1)
        mu = self.mu.view(1, -1, 1, 1)

        out = alpha * (x - mu)
        out = x / torch.sqrt(1 + out ** 2)
        out = gamma * out + beta
        return out


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
