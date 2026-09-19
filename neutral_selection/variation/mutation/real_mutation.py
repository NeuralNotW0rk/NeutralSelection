from __future__ import annotations

import math
import random
import dataclasses
from typing import Optional, Tuple, Any
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.representation.hierarchy import flatten_hierarchy, unflatten_hierarchy
from .base import MutationStrategy
from neutral_selection.registry import register_mutation


def _validate_composite_genome(genome: Any) -> None:
    if isinstance(genome, (str, bytes, bytearray, int, float, bool)) or not (
        isinstance(genome, (Genome, list, tuple))
        or dataclasses.is_dataclass(genome)
        or hasattr(genome, "shape")
        or hasattr(genome, "__hierarchical_flatten__")
    ):
        raise TypeError(f"Genome must be a Genome, sequence, dataclass or tensor, got {type(genome).__name__}")


def _clamp(val: float, bounds: Optional[Tuple[float, float]]) -> float:
    """Clamps a numeric value within optional (lower, upper) bounds."""
    if bounds is None:
        return val
    low, high = bounds
    return max(low, min(high, val))


@register_mutation(["gaussian", "normal"])
class GaussianMutation(MutationStrategy):
    """
    Gaussian mutation strategy (Evolution Strategies).

    Adds zero-mean Gaussian noise N(0, sigma^2) to numeric gene values with probability `mutation_rate`.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        mutation_rate: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or sigma <= 0.0:
            raise ValueError(f"sigma must be a positive float (> 0.0), got {sigma}.")
        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
            raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")
        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

        self.sigma = float(sigma)
        self.mutation_rate = float(mutation_rate)
        self.bounds = bounds
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)

        mutated_items: list[float] = []
        for val in leaves:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"GaussianMutation requires numeric genome elements, got {type(val).__name__}.")
            if random.random() < self.mutation_rate:
                new_val = _clamp(float(val) + random.gauss(0.0, self.sigma), self.bounds)
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["uniform_real", "real_uniform"])
class UniformRealMutation(MutationStrategy):
    """
    Uniform real-valued mutation strategy.

    Perturbs numeric genes by adding uniform noise in [-delta, delta] with probability `mutation_rate`.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        delta: float = 1.0,
        mutation_rate: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or delta <= 0.0:
            raise ValueError(f"delta must be a positive float (> 0.0), got {delta}.")
        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
            raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")
        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

        self.delta = float(delta)
        self.mutation_rate = float(mutation_rate)
        self.bounds = bounds
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)

        mutated_items: list[float] = []
        for val in leaves:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"UniformRealMutation requires numeric genome elements, got {type(val).__name__}.")
            if random.random() < self.mutation_rate:
                noise = random.uniform(-self.delta, self.delta)
                new_val = _clamp(float(val) + noise, self.bounds)
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["polynomial", "pm"])
class PolynomialMutation(MutationStrategy):
    """
    Polynomial mutation strategy (Deb & Agrawal, 1995; Deb, 2001 - NSGA-II).

    Applies bounded polynomial distribution perturbation. Standard in real-parameter genetic algorithms.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        eta_m: float = 20.0,
        mutation_rate: Optional[float] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError("bounds must be a tuple of (lower, upper) with lower < upper.")
        if not isinstance(eta_m, (int, float)) or isinstance(eta_m, bool) or eta_m < 0.0:
            raise ValueError(f"eta_m must be a non-negative float (>= 0.0), got {eta_m}.")
        if mutation_rate is not None:
            if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
                raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
            if not (0.0 <= mutation_rate <= 1.0):
                raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")
            self.mutation_rate: Optional[float] = float(mutation_rate)
        else:
            self.mutation_rate = None
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.eta_m = float(eta_m)
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)

        n = len(leaves)
        p_m = (1.0 / n) if (self.mutation_rate is None and n > 0) else (self.mutation_rate or 1.0)
        low, high = self.bounds
        delta_max = high - low

        mutated_items: list[float] = []
        for val in leaves:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"PolynomialMutation requires numeric genome elements, got {type(val).__name__}.")

            x = float(val)
            if random.random() < p_m and delta_max > 0:
                u = random.random()
                delta1 = (x - low) / delta_max
                delta2 = (high - x) / delta_max

                mut_pow = 1.0 / (self.eta_m + 1.0)
                if u <= 0.5:
                    xy = 1.0 - delta1
                    val_inner = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (self.eta_m + 1.0))
                    delta_q = (val_inner ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val_inner = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (self.eta_m + 1.0))
                    delta_q = 1.0 - (val_inner ** mut_pow)

                new_val = _clamp(x + delta_q * delta_max, self.bounds)
            else:
                new_val = _clamp(x, self.bounds)

            mutated_items.append(new_val)

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["cauchy", "cauchy_mutation"])
class CauchyMutation(MutationStrategy):
    """
    Cauchy mutation strategy (Fast Evolutionary Programming - Yao & Liu, 1996).

    Adds heavy-tailed Cauchy noise to numeric gene values with probability `mutation_rate`.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        scale: float = 1.0,
        mutation_rate: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0.0:
            raise ValueError(f"scale must be a positive float (> 0.0), got {scale}.")
        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
            raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")
        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

        self.scale = float(scale)
        self.mutation_rate = float(mutation_rate)
        self.bounds = bounds
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)

        mutated_items: list[float] = []
        for val in leaves:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"CauchyMutation requires numeric genome elements, got {type(val).__name__}.")
            if random.random() < self.mutation_rate:
                u = random.random()
                cauchy_noise = self.scale * math.tan(math.pi * (u - 0.5))
                new_val = _clamp(float(val) + cauchy_noise, self.bounds)
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return unflatten_hierarchy(mutated_items, treedef)
