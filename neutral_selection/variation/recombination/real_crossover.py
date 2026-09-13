from __future__ import annotations

import random
from typing import Optional, Tuple
from neutral_selection.representation.genome import Genome, Segment
from .base import RecombinationStrategy


def clone_genome_structure(original: Genome, new_items: list) -> Genome:
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


def _validate_real_parents(parent_a: Genome, parent_b: Genome) -> None:
    """Validates that parent genomes are non-empty numeric sequences of equal length."""
    if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
        raise TypeError("parent_a and parent_b must be Genome instances.")
    if len(parent_a) != len(parent_b):
        raise ValueError("Genomes must have the same length for crossover.")


class ArithmeticCrossover(RecombinationStrategy):
    """
    Arithmetic Crossover strategy (Michalewicz, 1992).

    Computes linear combinations of parent numeric vectors:
        child1 = alpha * parent_a + (1 - alpha) * parent_b
        child2 = (1 - alpha) * parent_a + alpha * parent_b
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise TypeError(f"alpha must be a float, got {type(alpha).__name__}.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")
        self.alpha = float(alpha)

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        _validate_real_parents(parent_a, parent_b)

        child1_items = []
        child2_items = []

        a = self.alpha
        for x1, x2 in zip(parent_a, parent_b):
            if not isinstance(x1, (int, float)) or not isinstance(x2, (int, float)) or isinstance(x1, bool) or isinstance(x2, bool):
                raise TypeError("ArithmeticCrossover requires numeric genome elements.")
            c1 = a * float(x1) + (1.0 - a) * float(x2)
            c2 = (1.0 - a) * float(x1) + a * float(x2)
            child1_items.append(c1)
            child2_items.append(c2)

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )


class BlendCrossover(RecombinationStrategy):
    """
    Blend Crossover strategy (BLX-alpha - Eshelman & Schaffer, 1993).

    For each gene, samples offspring values uniformly from:
        [c_min - alpha * d, c_max + alpha * d]
    where d = |parent_a - parent_b|, c_min = min(parent_a, parent_b), c_max = max(parent_a, parent_b).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0.0:
            raise ValueError(f"alpha must be a non-negative float (>= 0.0), got {alpha}.")
        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")

        self.alpha = float(alpha)
        self.bounds = bounds

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        _validate_real_parents(parent_a, parent_b)

        child1_items = []
        child2_items = []

        for x1, x2 in zip(parent_a, parent_b):
            if not isinstance(x1, (int, float)) or not isinstance(x2, (int, float)) or isinstance(x1, bool) or isinstance(x2, bool):
                raise TypeError("BlendCrossover requires numeric genome elements.")

            v1, v2 = float(x1), float(x2)
            c_min = min(v1, v2)
            c_max = max(v1, v2)
            d = c_max - c_min
            low = c_min - self.alpha * d
            high = c_max + self.alpha * d

            c1 = _clamp(random.uniform(low, high), self.bounds)
            c2 = _clamp(random.uniform(low, high), self.bounds)
            child1_items.append(c1)
            child2_items.append(c2)

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )


class SimulatedBinaryCrossover(RecombinationStrategy):
    """
    Simulated Binary Crossover strategy (SBX - Deb & Agrawal, 1995; Deb & Beyer, 2001).

    Simulates the search behavior of single-point binary crossover in continuous search spaces.
    Standard crossover operator in NSGA-II.
    """

    def __init__(
        self,
        eta_c: float = 2.0,
        swap_prob: float = 0.5,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not isinstance(eta_c, (int, float)) or isinstance(eta_c, bool) or eta_c < 0.0:
            raise ValueError(f"eta_c must be a non-negative float (>= 0.0), got {eta_c}.")
        if not isinstance(swap_prob, (int, float)) or isinstance(swap_prob, bool) or not (0.0 <= swap_prob <= 1.0):
            raise ValueError(f"swap_prob must be in [0.0, 1.0], got {swap_prob}.")
        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")

        self.eta_c = float(eta_c)
        self.swap_prob = float(swap_prob)
        self.bounds = bounds

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        _validate_real_parents(parent_a, parent_b)

        child1_items = []
        child2_items = []

        for x1, x2 in zip(parent_a, parent_b):
            if not isinstance(x1, (int, float)) or not isinstance(x2, (int, float)) or isinstance(x1, bool) or isinstance(x2, bool):
                raise TypeError("SimulatedBinaryCrossover requires numeric genome elements.")

            v1, v2 = float(x1), float(x2)
            if random.random() <= self.swap_prob and abs(v1 - v2) > 1e-14:
                u = random.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (self.eta_c + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta_c + 1.0))

                c1 = 0.5 * ((1.0 + beta) * v1 + (1.0 - beta) * v2)
                c2 = 0.5 * ((1.0 - beta) * v1 + (1.0 + beta) * v2)
            else:
                c1, c2 = v1, v2

            child1_items.append(_clamp(c1, self.bounds))
            child2_items.append(_clamp(c2, self.bounds))

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )
