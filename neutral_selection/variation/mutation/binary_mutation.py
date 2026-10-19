from __future__ import annotations

import random
import sys
from typing import Optional, Tuple
from neutral_selection.representation.genome import Genome
from neutral_selection.representation.hierarchy import (
    Granularity,
    resolve_granularity,
    flatten_strand,
    float_strand,
    is_tensor_strand,
    unflatten_hierarchy,
)
from .base import MutationStrategy
from neutral_selection.registry import register_mutation


def _mutation_mask(x: object, rate: float) -> object:
    """Draws an elementwise mask selecting each strand element with probability `rate`."""
    return sys.modules["torch"].rand(x.shape, device=x.device) < rate


@register_mutation(["bit_flip", "bitflip", "binary"])
class BitFlipMutation(MutationStrategy):
    """
    Bit flip mutation strategy (Holland, 1975).

    Flips boolean or binary (0/1) gene values across the genome with probability `mutation_rate`
    (defaults to 1 / number of leaves).
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        mutation_rate: Optional[float] = None,
        atomic_types: tuple[type, ...] = (),
        granularity: Granularity = None,
    ) -> None:
        if mutation_rate is not None:
            if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
                raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
            if not (0.0 <= mutation_rate <= 1.0):
                raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")
            self.mutation_rate: Optional[float] = float(mutation_rate)
        else:
            self.mutation_rate = None
        self.granularity = resolve_granularity(granularity)
        self.atomic_types = atomic_types

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        leaves, treedef = flatten_strand(genome, granularity=self.granularity, atomic_types=self.atomic_types)

        n = len(leaves)
        p_m = (1.0 / n) if (self.mutation_rate is None and n > 0) else (self.mutation_rate or 0.0)

        if is_tensor_strand(leaves):
            torch = sys.modules["torch"]
            if leaves.dtype == torch.bool:
                flipped = ~leaves
            elif leaves.is_floating_point() or leaves.is_complex() or not bool(((leaves == 0) | (leaves == 1)).all()):
                raise TypeError(f"BitFlipMutation expects bool or 0/1 integer tensors, got {leaves.dtype}.")
            else:
                flipped = 1 - leaves
            return unflatten_hierarchy(torch.where(_mutation_mask(leaves, p_m), flipped, leaves), treedef)

        mutated_items: list = []
        for val in leaves:
            if random.random() < p_m:
                if isinstance(val, bool):
                    new_val = not val
                elif val == 0 or val == 1:
                    new_val = 1 if val == 0 else 0
                else:
                    raise TypeError(f"BitFlipMutation expects bool or 0/1 integers, got {val} ({type(val).__name__}).")
            else:
                new_val = val
            mutated_items.append(new_val)

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["boundary", "boundary_mutation"])
class BoundaryMutation(MutationStrategy):
    """
    Boundary mutation strategy (Michalewicz, 1992).

    Randomly resets numeric genes to either their lower or upper bound with probability `mutation_rate`.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        mutation_rate: float = 1.0,
        atomic_types: tuple[type, ...] = (),
        granularity: Granularity = None,
    ) -> None:
        if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")
        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
            raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")

        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.mutation_rate = float(mutation_rate)
        self.granularity = resolve_granularity(granularity)
        self.atomic_types = atomic_types

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        leaves, treedef = flatten_strand(genome, granularity=self.granularity, atomic_types=self.atomic_types)
        low, high = self.bounds

        if is_tensor_strand(leaves):
            torch = sys.modules["torch"]
            x = float_strand(leaves, "BoundaryMutation")
            bound = torch.where(torch.rand_like(x) < 0.5, torch.full_like(x, low), torch.full_like(x, high))
            return unflatten_hierarchy(torch.where(_mutation_mask(x, self.mutation_rate), bound, x), treedef)

        mutated_items: list[float] = []
        for val in leaves:
            if random.random() < self.mutation_rate:
                new_val = random.choice([low, high])
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return unflatten_hierarchy(mutated_items, treedef)
