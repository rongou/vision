from typing import Callable

import torch
from torch import nn


class AffineTransform(nn.Module):
    """Affine transformation layer for dynamic activation functions."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def __repr__(self) -> str:
        return f"AffineTransform({self.num_features})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x + beta


class DyBase(nn.Module):
    """Base class for dynamic activation functions with affine transforms."""

    def __init__(self, num_features: int, init_alpha: float = 1.0, vector_alpha: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.init_alpha = init_alpha
        self.vector_alpha = vector_alpha

        # Initialize alpha parameter (scalar or vector based on vector_alpha flag)
        if vector_alpha:
            self.alpha = nn.Parameter(torch.ones(num_features) * init_alpha)
        else:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))

        self.affine = AffineTransform(num_features)

    def __repr__(self) -> str:
        alpha_type = "vector" if self.vector_alpha else "scalar"
        return (f"{self.__class__.__name__}("
                f"{self.num_features}, "
                f"init_alpha={self.init_alpha:.1f}, "
                f"alpha_type={alpha_type})")

    def get_alpha(self) -> torch.Tensor:
        """Return alpha in the right shape for operations based on vector_alpha flag."""
        if self.vector_alpha:
            return self.alpha.view(1, -1, 1, 1)
        return self.alpha

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Apply activation function. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement activation method")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.get_alpha()
        x = self.activation(x, alpha)
        return self.affine(x)


class DyT(DyBase):
    """Dynamic Tanh activation with scalar alpha."""

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return torch.tanh(alpha * x)


class DyTV(DyBase):
    """Dynamic Tanh activation with vector alpha."""

    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__(num_features, init_alpha, vector_alpha=True)

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return torch.tanh(alpha * x)


class DyAS(DyBase):
    """Dynamic Algebraic Sigmoid with scalar alpha."""

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(1 + (alpha * x) ** 2)


class DyASV(DyBase):
    """Dynamic Algebraic Sigmoid with vector alpha."""

    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__(num_features, init_alpha, vector_alpha=True)

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(1 + (alpha * x) ** 2)


class DyS(DyBase):
    """Dynamic Softsign with scalar alpha."""

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return torch.softsign(alpha * x)


class DySV(DyBase):
    """Dynamic Softsign with vector alpha."""

    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__(num_features, init_alpha, vector_alpha=True)

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return torch.softsign(alpha * x)


def get_norm_layer(norm_type: str) -> Callable[[int], nn.Module]:
    """Return a normalization layer based on the specified type."""
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
