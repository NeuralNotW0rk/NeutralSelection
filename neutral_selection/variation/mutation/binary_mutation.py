from __future__ import annotations

import random
from typing import Optional, Tuple
from neutral_selection.representation.genome import Genome, Segment
from .base import MutationStrategy
from neutral_selection.registry import register_mutation


def _clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


@register_mutation(["bit_flip", "bitflip", "binary"])
class BitFlipMutation(MutationStrategy):
    """
    Bit flip mutation strategy (Holland, 1975).

    Flips boolean or binary (0/1) gene values across the genome with probability `mutation_rate`
    (defaults to 1 / length).
    """

    def __init__(self, mutation_rate: Optional[float] = None) -> None:
        if mutation_rate is not None:
            if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
                raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
            if not (0.0 <= mutation_rate <= 1.0):
                raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")
            self.mutation_rate: Optional[float] = float(mutation_rate)
        else:
            self.mutation_rate = None

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        n = len(genome)
        p_m = (1.0 / n) if (self.mutation_rate is None and n > 0) else (self.mutation_rate or 0.0)

        mutated_items: list = []
        for val in genome:
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

        return _clone_genome_structure(genome, mutated_items)


@register_mutation(["boundary", "boundary_mutation"])
class BoundaryMutation(MutationStrategy):
    """
    Boundary mutation strategy (Michalewicz, 1992).

    Randomly resets numeric genes to either their lower or upper bound with probability `mutation_rate`.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        mutation_rate: float = 1.0,
    ) -> None:
        if not isinstance(bounds, tuple) or len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError("bounds must be a tuple of (lower, upper) with lower <= upper.")
        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
            raise TypeError(f"mutation_rate must be a float, got {type(mutation_rate).__name__}.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {mutation_rate}.")

        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.mutation_rate = float(mutation_rate)

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        mutated_items: list[float] = []
        low, high = self.bounds
        for val in genome:
            if random.random() < self.mutation_rate:
                new_val = random.choice([low, high])
            else:
                new_val = float(val)
            mutated_items.append(new_val)

        return _clone_genome_structure(genome, mutated_items)
