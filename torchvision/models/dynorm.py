from typing import Callable

import torch
from torch import nn


class AffineTransform(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x + beta


class DyT(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.affine = AffineTransform(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        return self.affine(x)


class DyTV(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        self.affine = AffineTransform(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        x = torch.tanh(alpha * x)
        return self.affine(x)


class DyAS(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.affine = AffineTransform(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.sqrt(1 + (self.alpha * x) ** 2)
        return self.affine(x)


class DyASV(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        self.affine = AffineTransform(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        x = x / torch.sqrt(1 + (alpha * x) ** 2)
        return self.affine(x)


class DyS(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.affine = AffineTransform(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.softsign(self.alpha * x)
        return self.affine(x)


class DySV(nn.Module):
    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        self.affine = AffineTransform(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        x = torch.softsign(alpha * x)
        return self.affine(x)


def get_norm_layer(norm_type: str) -> Callable[[int], nn.Module]:
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'dyt':
        return DyT
    elif norm_type == 'dytv':
        return DyTV
    elif norm_type == 'dyas':
        return DyAS
    elif norm_type == 'dyasv':
        return DyASV
    elif norm_type == 'dys':
        return DyS
    elif norm_type == 'dysv':
        return DySV
    else:
        raise ValueError(f'Unknown normalization type {norm_type}.')
