import math
from typing import Callable

import torch
from torch import nn


class DyT(nn.Module):
    """Dynamic Tanh activation with scalar alpha."""

    def __init__(self, num_features: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.num_features = num_features
        self.init_alpha = init_alpha
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.num_features}, "
                f"init_alpha={self.init_alpha:.1f})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        x = torch.tanh(self.alpha * x)
        return gamma * x + beta


class DyAS(nn.Module):
    """Dynamic Algebraic Sigmoid with scalar alpha."""

    def __init__(self, num_features: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.num_features = num_features
        self.init_alpha = init_alpha
        # log-space parameterization of alpha to ensure positivity
        self.theta = nn.Parameter(torch.tensor(math.log(self.init_alpha)))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.num_features}, "
                f"init_alpha={self.init_alpha:.1f})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.theta)
        x = x / torch.sqrt(1 + (alpha * x) ** 2)
        return x + self.beta.view(1, -1, 1, 1)


class DyASV(nn.Module):
    """Dynamic Algebraic Sigmoid with vector alpha."""

    def __init__(self, num_features: int, init_alpha: float = 0.5) -> None:
        super().__init__()
        self.num_features = num_features
        self.init_alpha = init_alpha
        # log-space parameterization of alpha to ensure positivity
        self.theta = nn.Parameter(torch.full((num_features,), math.log(self.init_alpha)))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.num_features}, "
                f"init_alpha={self.init_alpha:.1f})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.theta).view(1, -1, 1, 1)
        x = x / torch.sqrt(1 + (alpha * x) ** 2)
        return x + self.beta.view(1, -1, 1, 1)


def get_norm_layer(norm_type: str, init_alpha: float = 1.0) -> Callable[
    [int], nn.Module]:
    """
    Returns a normalization layer constructor based on the specified type.

    Args:
        norm_type: Type of normalization layer ('batch', 'dyt', 'dyas', etc.)
        init_alpha: Initial value for alpha parameter in dynamic activations

    Returns:
        A function that takes num_features as input and returns a normalization module
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'dyt':
        return lambda num_features: DyT(num_features, init_alpha=init_alpha)
    elif norm_type == 'dyas':
        return lambda num_features: DyAS(num_features, init_alpha=init_alpha)
    elif norm_type == 'dyasv':
        return lambda num_features: DyASV(num_features, init_alpha=init_alpha)
    else:
        raise ValueError(f'Unknown normalization type {norm_type}.')
