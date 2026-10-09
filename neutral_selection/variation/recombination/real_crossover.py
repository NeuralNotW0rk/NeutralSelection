from __future__ import annotations

import random
import dataclasses
import sys
from typing import Optional, Tuple, Any
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.representation.hierarchy import (
    align_strands,
    flatten_strand,
    float_strand,
    is_tensor_strand,
    unflatten_hierarchy,
)
from .base import RecombinationStrategy
from neutral_selection.registry import register_crossover


def _validate_composite_parent(parent: Any) -> None:
    if isinstance(parent, (str, bytes, bytearray, int, float, bool)) or not (
        isinstance(parent, (Genome, list, tuple))
        or dataclasses.is_dataclass(parent)
        or hasattr(parent, "shape")
        or hasattr(parent, "__hierarchical_flatten__")
    ):
        raise TypeError(f"Parent must be a Genome, sequence, dataclass or tensor, got {type(parent).__name__}")


def _clamp(val: Any, bounds: Optional[Tuple[float, float]]) -> Any:
    """Clamps a numeric value (or every element of a tensor strand) within optional (lower, upper) bounds."""
    if bounds is None:
        return val
    low, high = bounds
    if is_tensor_strand(val):
        return val.clamp(low, high)
    return max(low, min(high, val))


@register_crossover("arithmetic")
class ArithmeticCrossover(RecombinationStrategy):
    """
    Arithmetic Crossover strategy (Michalewicz, 1992).

    Computes linear combinations of parent numeric vectors:
        child1 = alpha * parent_a + (1 - alpha) * parent_b
        child2 = (1 - alpha) * parent_a + alpha * parent_b
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise TypeError(f"alpha must be a float, got {type(alpha).__name__}.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.alpha = float(alpha)
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        leaves_a, treedef_a = flatten_strand(parent_a, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_b, treedef_b = flatten_strand(parent_b, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_a, leaves_b = align_strands(leaves_a, leaves_b)

        if len(leaves_a) != len(leaves_b) or treedef_a.total_leaves != treedef_b.total_leaves:
            raise ValueError("Genomes must have matching leaf structures for arithmetic crossover")

        if is_tensor_strand(leaves_a):
            x1 = float_strand(leaves_a, "ArithmeticCrossover")
            x2 = float_strand(leaves_b, "ArithmeticCrossover")
            a = self.alpha
            return (
                unflatten_hierarchy(a * x1 + (1.0 - a) * x2, treedef_a),
                unflatten_hierarchy((1.0 - a) * x1 + a * x2, treedef_b),
            )

        child1_items = []
        child2_items = []

        a = self.alpha
        for x1, x2 in zip(leaves_a, leaves_b):
            if not isinstance(x1, (int, float)) or not isinstance(x2, (int, float)) or isinstance(x1, bool) or isinstance(x2, bool):
                raise TypeError("ArithmeticCrossover requires numeric genome elements.")
            c1 = a * float(x1) + (1.0 - a) * float(x2)
            c2 = (1.0 - a) * float(x1) + a * float(x2)
            child1_items.append(c1)
            child2_items.append(c2)

        return (
            unflatten_hierarchy(child1_items, treedef_a),
            unflatten_hierarchy(child2_items, treedef_b),
        )


@register_crossover("blend")
class BlendCrossover(RecombinationStrategy):
    """
    Blend Crossover strategy (BLX-alpha - Eshelman & Schaffer, 1993).

    For each gene, samples offspring values uniformly from:
        [c_min - alpha * d, c_max + alpha * d]
    where d = |parent_a - parent_b|, c_min = min(parent_a, parent_b), c_max = max(parent_a, parent_b).
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        bounds: Optional[Tuple[float, float]] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0.0:
            raise ValueError(f"alpha must be a non-negative float (>= 0.0), got {alpha}.")
        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

        self.alpha = float(alpha)
        self.bounds = bounds
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        leaves_a, treedef_a = flatten_strand(parent_a, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_b, treedef_b = flatten_strand(parent_b, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_a, leaves_b = align_strands(leaves_a, leaves_b)

        if len(leaves_a) != len(leaves_b) or treedef_a.total_leaves != treedef_b.total_leaves:
            raise ValueError("Genomes must have matching leaf structures for blend crossover")

        if is_tensor_strand(leaves_a):
            torch = sys.modules["torch"]
            v1 = float_strand(leaves_a, "BlendCrossover")
            v2 = float_strand(leaves_b, "BlendCrossover")
            c_min = torch.minimum(v1, v2)
            d = torch.maximum(v1, v2) - c_min
            low = c_min - self.alpha * d
            span = d * (1.0 + 2.0 * self.alpha)
            return (
                unflatten_hierarchy(_clamp(low + span * torch.rand_like(v1), self.bounds), treedef_a),
                unflatten_hierarchy(_clamp(low + span * torch.rand_like(v1), self.bounds), treedef_b),
            )

        child1_items = []
        child2_items = []

        for x1, x2 in zip(leaves_a, leaves_b):
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
            unflatten_hierarchy(child1_items, treedef_a),
            unflatten_hierarchy(child2_items, treedef_b),
        )


@register_crossover(["simulated_binary", "sbx"])
class SimulatedBinaryCrossover(RecombinationStrategy):
    """
    Simulated Binary Crossover strategy (SBX - Deb & Agrawal, 1995; Deb & Beyer, 2001).

    Simulates the search behavior of single-point binary crossover in continuous search spaces.
    Standard crossover operator in NSGA-II.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        eta_c: float = 2.0,
        swap_prob: float = 0.5,
        bounds: Optional[Tuple[float, float]] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(eta_c, (int, float)) or isinstance(eta_c, bool) or eta_c < 0.0:
            raise ValueError(f"eta_c must be a non-negative float (>= 0.0), got {eta_c}.")
        if not isinstance(swap_prob, (int, float)) or isinstance(swap_prob, bool) or not (0.0 <= swap_prob <= 1.0):
            raise ValueError(f"swap_prob must be in [0.0, 1.0], got {swap_prob}.")
        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

        self.eta_c = float(eta_c)
        self.swap_prob = float(swap_prob)
        self.bounds = bounds
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        leaves_a, treedef_a = flatten_strand(parent_a, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_b, treedef_b = flatten_strand(parent_b, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_a, leaves_b = align_strands(leaves_a, leaves_b)

        if len(leaves_a) != len(leaves_b) or treedef_a.total_leaves != treedef_b.total_leaves:
            raise ValueError("Genomes must have matching leaf structures for simulated binary crossover")

        if is_tensor_strand(leaves_a):
            torch = sys.modules["torch"]
            v1 = float_strand(leaves_a, "SimulatedBinaryCrossover")
            v2 = float_strand(leaves_b, "SimulatedBinaryCrossover")
            crossed = (torch.rand_like(v1) <= self.swap_prob) & ((v1 - v2).abs() > 1e-14)
            u = torch.rand_like(v1)
            exponent = 1.0 / (self.eta_c + 1.0)
            beta = torch.where(u <= 0.5, (2.0 * u).pow(exponent), (1.0 / (2.0 * (1.0 - u))).pow(exponent))
            c1 = torch.where(crossed, 0.5 * ((1.0 + beta) * v1 + (1.0 - beta) * v2), v1)
            c2 = torch.where(crossed, 0.5 * ((1.0 - beta) * v1 + (1.0 + beta) * v2), v2)
            return (
                unflatten_hierarchy(_clamp(c1, self.bounds), treedef_a),
                unflatten_hierarchy(_clamp(c2, self.bounds), treedef_b),
            )

        child1_items = []
        child2_items = []

        for x1, x2 in zip(leaves_a, leaves_b):
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
            unflatten_hierarchy(child1_items, treedef_a),
            unflatten_hierarchy(child2_items, treedef_b),
        )

