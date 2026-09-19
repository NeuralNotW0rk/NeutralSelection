from __future__ import annotations

import random
import dataclasses
from typing import Optional, Any
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


def _clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


@register_mutation(["inversion", "2opt"])
class InversionMutation(MutationStrategy):
    """
    Inversion mutation strategy (2-opt reversal).

    Selects a random subsequence within the multi-scale genome strand and reverses the order of its elements.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)
        n = len(leaves)
        if n <= 1:
            return genome

        idx1, idx2 = sorted(random.sample(range(n + 1), 2))
        mutated_items = list(leaves)
        mutated_items[idx1:idx2] = reversed(mutated_items[idx1:idx2])

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["swap", "exchange"])
class SwapMutation(MutationStrategy):
    """
    Swap mutation strategy (Exchange mutation).

    Selects two random elements within the multi-scale genome sequence and swaps their positions.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)
        n = len(leaves)
        if n <= 1:
            return genome

        idx1, idx2 = random.sample(range(n), 2)
        mutated_items = list(leaves)
        mutated_items[idx1], mutated_items[idx2] = mutated_items[idx2], mutated_items[idx1]

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["scramble", "shuffle"])
class ScrambleMutation(MutationStrategy):
    """
    Scramble mutation strategy.

    Selects a random subsequence within the multi-scale genome sequence and shuffles its elements.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)
        n = len(leaves)
        if n <= 1:
            return genome

        idx1, idx2 = sorted(random.sample(range(n + 1), 2))
        subseq = list(leaves[idx1:idx2])
        random.shuffle(subseq)

        mutated_items = list(leaves)
        mutated_items[idx1:idx2] = subseq

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["insertion", "displacement"])
class InsertionMutation(MutationStrategy):
    """
    Insertion mutation strategy (Displacement mutation).

    Selects an element at a random index, removes it, and inserts it at another random index.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)
        n = len(leaves)
        if n <= 1:
            return genome

        from_idx, to_idx = random.sample(range(n), 2)
        mutated_items = list(leaves)
        item = mutated_items.pop(from_idx)
        mutated_items.insert(to_idx, item)

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["transposition", "block_swap"])
class TranspositionMutation(MutationStrategy):
    """
    Transposition mutation strategy (Block swap mutation).

    Selects two non-overlapping contiguous slices/blocks within the multi-scale genome and exchanges them.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        block_size: Optional[int] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if block_size is not None:
            if type(block_size) is not int or block_size < 1:
                raise ValueError(f"block_size must be an integer >= 1, got {block_size}.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.block_size = block_size
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)
        n = len(leaves)
        if n < 4:
            return genome

        # Sample 4 cut points to define two disjoint intervals
        cuts = sorted(random.sample(range(n + 1), 4))
        p1, p2, p3, p4 = cuts

        block1 = list(leaves[p1:p2])
        block2 = list(leaves[p3:p4])

        mutated_items = (
            list(leaves[:p1])
            + block2
            + list(leaves[p2:p3])
            + block1
            + list(leaves[p4:])
        )

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["duplication", "duplicate"])
class DuplicationMutation(MutationStrategy):
    """
    Duplication mutation strategy.

    Selects a random element or subsequence and duplicates it at another location in the genome.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        max_length: Optional[int] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if max_length is not None:
            if type(max_length) is not int or max_length < 1:
                raise ValueError(f"max_length must be an integer >= 1, got {max_length}.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.max_length = max_length
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)
        n = len(leaves)
        if n == 0:
            return genome

        if self.max_length is not None and n >= self.max_length:
            return genome

        idx1, idx2 = sorted(random.sample(range(n + 1), 2))
        if idx1 == idx2:
            idx2 = min(n, idx1 + 1)
        subseq = list(leaves[idx1:idx2])

        insert_pos = random.randint(0, n)
        mutated_items = list(leaves[:insert_pos]) + subseq + list(leaves[insert_pos:])

        return unflatten_hierarchy(mutated_items, treedef)


@register_mutation(["deletion", "delete"])
class DeletionMutation(MutationStrategy):
    """
    Deletion mutation strategy.

    Deletes a random element or subsequence from the genome, preserving a minimum genome length.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        min_length: int = 1,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if type(min_length) is not int or min_length < 0:
            raise ValueError(f"min_length must be an integer >= 0, got {min_length}.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.min_length = min_length
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, genome: Any) -> Any:
        _validate_composite_genome(genome)
        leaves, treedef = flatten_hierarchy(genome, max_depth=self.max_depth, atomic_types=self.atomic_types)
        n = len(leaves)
        if n <= self.min_length:
            return genome

        max_deletable = n - self.min_length
        idx1 = random.randint(0, n - 1)
        max_idx2 = min(n, idx1 + max_deletable)
        idx2 = random.randint(idx1 + 1, max_idx2)

        mutated_items = list(leaves[:idx1]) + list(leaves[idx2:])
        return unflatten_hierarchy(mutated_items, treedef)
