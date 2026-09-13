from __future__ import annotations

import math
import random
from typing import Optional, Tuple
from neutral_selection.representation.genome import Genome, Segment
from .base import MutationStrategy


def _clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


def _clamp(val: float, bounds: Optional[Tuple[float, float]]) -> float:
    """Clamps a numeric value within optional (lower, upper) bounds."""
    if bounds is None:
        return val
    low, high = bounds
    return max(low, min(high, val))


class GaussianMutation(MutationStrategy):
    """
    Gaussian mutation strategy (Evolution Strategies).

    Adds zero-mean Gaussian noise N(0, sigma^2) to numeric gene values with probability `mutation_rate`.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        mutation_rate: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
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

        self.sigma = float(sigma)
        self.mutation_rate = float(mutation_rate)
        self.bounds = bounds

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        mutated_items: list[float] = []
        for val in genome:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"GaussianMutation requires numeric genome elements, got {type(val).__name__}.")
            if random.random() < self.mutation_rate:
                new_val = _clamp(float(val) + random.gauss(0.0, self.sigma), self.bounds)
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return _clone_genome_structure(genome, mutated_items)


class UniformRealMutation(MutationStrategy):
    """
    Uniform real-valued mutation strategy.

    Perturbs numeric genes by adding uniform noise in [-delta, delta] with probability `mutation_rate`.
    """

    def __init__(
        self,
        delta: float = 1.0,
        mutation_rate: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
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

        self.delta = float(delta)
        self.mutation_rate = float(mutation_rate)
        self.bounds = bounds

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        mutated_items: list[float] = []
        for val in genome:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"UniformRealMutation requires numeric genome elements, got {type(val).__name__}.")
            if random.random() < self.mutation_rate:
                noise = random.uniform(-self.delta, self.delta)
                new_val = _clamp(float(val) + noise, self.bounds)
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return _clone_genome_structure(genome, mutated_items)


class PolynomialMutation(MutationStrategy):
    """
    Polynomial mutation strategy (Deb & Agrawal, 1995; Deb, 2001 - NSGA-II).

    Applies bounded polynomial distribution perturbation. Standard in real-parameter genetic algorithms.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        eta_m: float = 20.0,
        mutation_rate: Optional[float] = None,
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

        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.eta_m = float(eta_m)

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        n = len(genome)
        p_m = (1.0 / n) if self.mutation_rate is None else self.mutation_rate
        low, high = self.bounds
        delta_max = high - low

        mutated_items: list[float] = []
        for val in genome:
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

        return _clone_genome_structure(genome, mutated_items)


class CauchyMutation(MutationStrategy):
    """
    Cauchy mutation strategy (Fast Evolutionary Programming - Yao & Liu, 1996).

    Adds heavy-tailed Cauchy noise to numeric gene values with probability `mutation_rate`.
    """

    def __init__(
        self,
        scale: float = 1.0,
        mutation_rate: float = 1.0,
        bounds: Optional[Tuple[float, float]] = None,
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

        self.scale = float(scale)
        self.mutation_rate = float(mutation_rate)
        self.bounds = bounds

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        mutated_items: list[float] = []
        for val in genome:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"CauchyMutation requires numeric genome elements, got {type(val).__name__}.")
            if random.random() < self.mutation_rate:
                u = random.random()
                cauchy_noise = self.scale * math.tan(math.pi * (u - 0.5))
                new_val = _clamp(float(val) + cauchy_noise, self.bounds)
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return _clone_genome_structure(genome, mutated_items)
