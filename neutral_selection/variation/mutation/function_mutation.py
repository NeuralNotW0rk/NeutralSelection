from __future__ import annotations

import random
from typing import Any, Callable
from neutral_selection.representation.hierarchy import (
    Granularity,
    resolve_granularity,
    flatten_hierarchy,
    unflatten_hierarchy,
)
from .base import MutationStrategy
from .real_mutation import _validate_composite_genome
from neutral_selection.registry import register_mutation


@register_mutation(["function", "callable"])
class FunctionMutation(MutationStrategy):
    """
    Applies a custom callable to each leaf with probability `mutation_rate`.

    The callable receives one leaf at the operator's granularity: a scalar at "atom" (the default),
    a whole tensor at "tensor", or a whole gene object at "gene".
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        mutation_fn: Callable[[Any], Any],
        mutation_rate: float = 1.0,
        atomic_types: tuple[type, ...] = (),
        granularity: Granularity = None,
    ) -> None:
        if mutation_fn is None:
            raise ValueError("mutation_fn must be provided and cannot be None.")
        if not callable(mutation_fn):
            raise TypeError("mutation_fn must be a callable.")
        if not isinstance(mutation_rate, (int, float)) or isinstance(mutation_rate, bool):
            raise TypeError("mutation_rate must be a float or int.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0 inclusive.")
        self.mutation_fn = mutation_fn
        self.mutation_rate = float(mutation_rate)
        self.atomic_types = atomic_types
        self.granularity = resolve_granularity(granularity)

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        # Leaves stay Python objects (never a tensor strand), so the callable sees one leaf at a time
        leaves, treedef = flatten_hierarchy(genome, atomic_types=self.atomic_types, granularity=self.granularity)
        mutated_items = [
            self.mutation_fn(leaf) if random.random() < self.mutation_rate else leaf
            for leaf in leaves
        ]
        return unflatten_hierarchy(mutated_items, treedef)
