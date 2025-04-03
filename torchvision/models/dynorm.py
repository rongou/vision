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

    def __init__(self, num_features: int, init_alpha: float = 0.5, alpha_noise_std: float = 0.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.init_alpha = init_alpha
        self.alpha_noise_std = alpha_noise_std
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.num_features}, "
                f"init_alpha={self.init_alpha:.1f}, "
                f"alpha_noise_std={self.alpha_noise_std:.1f})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        if self.training and self.alpha_noise_std > 0:
            noise = torch.randn_like(alpha) * self.alpha_noise_std
            alpha = alpha * torch.exp(noise)

        beta = self.beta.view(1, -1, 1, 1)
        x = x / torch.sqrt(1 + (alpha * x) ** 2)
        return x + beta



class DyASV(nn.Module):
    """Dynamic Algebraic Sigmoid with vector alpha."""

    def __init__(self, num_features: int, init_alpha: float = 0.5, alpha_noise_std: float = 0.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.init_alpha = init_alpha
        self.alpha_noise_std = alpha_noise_std
        self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        self.beta = nn.Parameter(torch.zeros(num_features))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"{self.num_features}, "
                f"init_alpha={self.init_alpha:.1f}, "
                f"alpha_noise_std={self.alpha_noise_std:.1f})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.view(1, -1, 1, 1)
        if self.training and self.alpha_noise_std > 0:
            noise = torch.randn_like(alpha) * self.alpha_noise_std
            alpha = alpha * torch.exp(noise)

        beta = self.beta.view(1, -1, 1, 1)
        x = x / torch.sqrt(1 + (alpha * x) ** 2)
        return x + beta


def get_norm_layer(norm_type: str, init_alpha: float = 1.0, alpha_noise_std: float = 0.0) -> Callable[[int], nn.Module]:
    """
    Returns a normalization layer constructor based on the specified type.

    Args:
        norm_type: Type of normalization layer ('batch', 'dyt', 'dyas', etc.)
        init_alpha: Initial value for alpha parameter in dynamic activations
        alpha_noise_std: Standard deviation for noise added to alpha during training

    Returns:
        A function that takes num_features as input and returns a normalization module
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'dyt':
        return lambda num_features: DyT(num_features, init_alpha=init_alpha)
    elif norm_type == 'dyas':
        return lambda num_features: DyAS(num_features, init_alpha=init_alpha, alpha_noise_std=alpha_noise_std)
    elif norm_type == 'dyasv':
        return lambda num_features: DyASV(num_features, init_alpha=init_alpha, alpha_noise_std=alpha_noise_std)
    else:
        raise ValueError(f'Unknown normalization type {norm_type}.')
