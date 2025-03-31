from typing import Callable

import torch
from torch import nn


class AffineTransform(nn.Module):
    """Affine transformation layer for dynamic activation functions."""

    def __init__(self, num_features: int, use_gamma: bool = False) -> None:
        super().__init__()
        self.num_features = num_features
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma = nn.Parameter(torch.ones(num_features))

    def __repr__(self) -> str:
        return f"AffineTransform({self.num_features}, use_gamma={self.use_gamma})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_gamma:
            return self.gamma.view(1, -1, 1, 1) * x + self.beta.view(1, -1, 1, 1)
        else:
            return x + self.beta.view(1, -1, 1, 1)


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
                f"alpha_type={alpha_type}), "
                f"affine={self.affine})")

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

    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__(num_features, init_alpha, vector_alpha=False)
        self.softsign = nn.Softsign()

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return self.softsign(alpha * x)


class DySV(DyBase):
    """Dynamic Softsign with vector alpha."""

    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__(num_features, init_alpha, vector_alpha=True)
        self.softsign = nn.Softsign()

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return self.softsign(alpha * x)


class DyRMS(DyBase):
    """Dynamic RMS-like activation with scalar alpha."""

    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__(num_features, init_alpha, vector_alpha=False)
        self.epsilon = 1e-5

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(alpha ** 2 + x ** 2 + self.epsilon)


class DyRMSV(DyBase):
    """Dynamic RMS-like activation with vector alpha."""

    def __init__(self, num_features: int, init_alpha: float = 1.0) -> None:
        super().__init__(num_features, init_alpha, vector_alpha=True)
        self.epsilon = 1e-5

    def activation(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(alpha ** 2 + x ** 2 + self.epsilon)


def get_norm_layer(norm_type: str, init_alpha: float = 1.0) -> Callable[[int], nn.Module]:
    """
    Returns a normalization layer constructor based on the specified type.

    Args:
        norm_type: Type of normalization layer ('batch', 'dyt', 'dytv', etc.)
        init_alpha: Initial value for alpha parameter in dynamic activations

    Returns:
        A function that takes num_features as input and returns a normalization module
    """
    if norm_type == 'batch':
        return nn.BatchNorm2d
    elif norm_type == 'dyt':
        return lambda num_features: DyT(num_features, init_alpha=init_alpha)
    elif norm_type == 'dytv':
        return lambda num_features: DyTV(num_features, init_alpha=init_alpha)
    elif norm_type == 'dyas':
        return lambda num_features: DyAS(num_features, init_alpha=init_alpha)
    elif norm_type == 'dyasv':
        return lambda num_features: DyASV(num_features, init_alpha=init_alpha)
    elif norm_type == 'dys':
        return lambda num_features: DyS(num_features, init_alpha=init_alpha)
    elif norm_type == 'dysv':
        return lambda num_features: DySV(num_features, init_alpha=init_alpha)
    elif norm_type == 'dyrms':
        return lambda num_features: DyRMS(num_features, init_alpha=init_alpha)
    elif norm_type == 'dyrmsv':
        return lambda num_features: DyRMSV(num_features, init_alpha=init_alpha)
    else:
        raise ValueError(f'Unknown normalization type {norm_type}.')
