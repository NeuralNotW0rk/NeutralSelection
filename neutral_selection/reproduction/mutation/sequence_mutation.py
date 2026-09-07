from __future__ import annotations

import random
from typing import Optional
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.reproduction.mutation.base import MutationStrategy


def _clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


class InversionMutation(MutationStrategy):
    """
    Inversion mutation strategy (2-opt reversal).

    Selects a random subsequence within the genome and reverses the order of its elements.
    """

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        if not hasattr(genome, "__len__") or not hasattr(genome, "__getitem__"):
            raise TypeError("genome must support sequence indexing and len for inversion mutation.")

        n = len(genome)
        if n <= 1:
            return genome

        idx1, idx2 = sorted(random.sample(range(n + 1), 2))
        mutated_items = list(genome)
        mutated_items[idx1:idx2] = reversed(mutated_items[idx1:idx2])

        return _clone_genome_structure(genome, mutated_items)


class SwapMutation(MutationStrategy):
    """
    Swap mutation strategy (Exchange mutation).

    Selects two random elements within the genome sequence and swaps their positions.
    """

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        if not hasattr(genome, "__len__") or not hasattr(genome, "__getitem__"):
            raise TypeError("genome must support sequence indexing and len for swap mutation.")

        n = len(genome)
        if n <= 1:
            return genome

        idx1, idx2 = random.sample(range(n), 2)
        mutated_items = list(genome)
        mutated_items[idx1], mutated_items[idx2] = mutated_items[idx2], mutated_items[idx1]

        return _clone_genome_structure(genome, mutated_items)


class ScrambleMutation(MutationStrategy):
    """
    Scramble mutation strategy.

    Selects a random subsequence within the genome sequence and shuffles its elements.
    """

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        if not hasattr(genome, "__len__") or not hasattr(genome, "__getitem__"):
            raise TypeError("genome must support sequence indexing and len for scramble mutation.")

        n = len(genome)
        if n <= 1:
            return genome

        idx1, idx2 = sorted(random.sample(range(n + 1), 2))
        subseq = list(genome[idx1:idx2])
        random.shuffle(subseq)

        mutated_items = list(genome)
        mutated_items[idx1:idx2] = subseq

        return _clone_genome_structure(genome, mutated_items)


class InsertionMutation(MutationStrategy):
    """
    Insertion mutation strategy (Displacement mutation).

    Selects an element at a random index, removes it, and inserts it at another random index.
    """

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        if not hasattr(genome, "__len__") or not hasattr(genome, "__getitem__"):
            raise TypeError("genome must support sequence indexing and len for insertion mutation.")

        n = len(genome)
        if n <= 1:
            return genome

        from_idx, to_idx = random.sample(range(n), 2)
        mutated_items = list(genome)
        item = mutated_items.pop(from_idx)
        mutated_items.insert(to_idx, item)

        return _clone_genome_structure(genome, mutated_items)


class TranspositionMutation(MutationStrategy):
    """
    Transposition mutation strategy (Block swap mutation).

    Selects two non-overlapping contiguous slices/blocks within the genome and exchanges them.
    """

    def __init__(self, block_size: Optional[int] = None) -> None:
        if block_size is not None:
            if type(block_size) is not int or block_size < 1:
                raise ValueError(f"block_size must be an integer >= 1, got {block_size}.")
        self.block_size = block_size

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        if not hasattr(genome, "__len__") or not hasattr(genome, "__getitem__"):
            raise TypeError("genome must support sequence indexing and len for transposition mutation.")

        n = len(genome)
        if n < 4:
            return genome

        # Sample 4 cut points to define two disjoint intervals
        cuts = sorted(random.sample(range(n + 1), 4))
        p1, p2, p3, p4 = cuts

        block1 = list(genome[p1:p2])
        block2 = list(genome[p3:p4])

        mutated_items = (
            list(genome[:p1])
            + block2
            + list(genome[p2:p3])
            + block1
            + list(genome[p4:])
        )

        return _clone_genome_structure(genome, mutated_items)


class DuplicationMutation(MutationStrategy):
    """
    Duplication mutation strategy.

    Selects a random element or subsequence and duplicates it at another location in the genome.
    """

    def __init__(self, max_length: Optional[int] = None) -> None:
        if max_length is not None:
            if type(max_length) is not int or max_length < 1:
                raise ValueError(f"max_length must be an integer >= 1, got {max_length}.")
        self.max_length = max_length

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        if not hasattr(genome, "__len__") or not hasattr(genome, "__getitem__"):
            raise TypeError("genome must support sequence indexing and len for duplication mutation.")

        n = len(genome)
        if n == 0:
            return genome

        if self.max_length is not None and n >= self.max_length:
            return genome

        idx1, idx2 = sorted(random.sample(range(n + 1), 2))
        if idx1 == idx2:
            idx2 = min(n, idx1 + 1)
        subseq = list(genome[idx1:idx2])

        insert_pos = random.randint(0, n)
        mutated_items = list(genome[:insert_pos]) + subseq + list(genome[insert_pos:])

        return _clone_genome_structure(genome, mutated_items)


class DeletionMutation(MutationStrategy):
    """
    Deletion mutation strategy.

    Deletes a random element or subsequence from the genome, preserving a minimum genome length.
    """

    def __init__(self, min_length: int = 1) -> None:
        if type(min_length) is not int or min_length < 0:
            raise ValueError(f"min_length must be an integer >= 0, got {min_length}.")
        self.min_length = min_length

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")
        if not hasattr(genome, "__len__") or not hasattr(genome, "__getitem__"):
            raise TypeError("genome must support sequence indexing and len for deletion mutation.")

        n = len(genome)
        if n <= self.min_length:
            return genome

        max_deletable = n - self.min_length
        idx1 = random.randint(0, n - 1)
        max_idx2 = min(n, idx1 + max_deletable)
        idx2 = random.randint(idx1 + 1, max_idx2)

        mutated_items = list(genome[:idx1]) + list(genome[idx2:])
        return _clone_genome_structure(genome, mutated_items)
