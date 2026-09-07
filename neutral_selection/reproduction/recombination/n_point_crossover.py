from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple, Union
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.reproduction.recombination.base import RecombinationStrategy


def clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


def n_point_crossover(parent_a: Genome, parent_b: Genome, cut_points: list[int]) -> tuple[Genome, Genome]:
    """
    Polymorphically slices and swaps parent genomes at the specified cut points.
    Acts as the atomic helper operation for structural sequence crossovers.
    """
    if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
        raise TypeError("parent_a and parent_b must be Genome instances.")
    if len(parent_a) != len(parent_b):
        raise ValueError("Genomes must have the same length for crossover")
    if any(not isinstance(c, int) or c < 0 for c in cut_points):
        raise ValueError("Cut points must be non-negative integers")

    sorted_cuts = sorted(list(set(cut_points)))

    child1_items = []
    child2_items = []

    last_cut = 0
    use_a = True

    for cut in sorted_cuts:
        if use_a:
            child1_items.extend(parent_a[last_cut:cut])
            child2_items.extend(parent_b[last_cut:cut])
        else:
            child1_items.extend(parent_b[last_cut:cut])
            child2_items.extend(parent_a[last_cut:cut])
        last_cut = cut
        use_a = not use_a

    if use_a:
        child1_items.extend(parent_a[last_cut:])
        child2_items.extend(parent_b[last_cut:])
    else:
        child1_items.extend(parent_b[last_cut:])
        child2_items.extend(parent_a[last_cut:])

    return (
        clone_genome_structure(parent_a, child1_items),
        clone_genome_structure(parent_b, child2_items),
    )


class OnePointCrossover(RecombinationStrategy):
    """
    One-Point Crossover strategy (Holland, 1975).

    Slices parents at a single cut point and exchanges tails to create two offspring.
    If `cut_point` is None, a cut point is chosen uniformly at random in [1, len(parent) - 1].
    """

    def __init__(self, cut_point: Optional[int] = None) -> None:
        if cut_point is not None:
            if type(cut_point) is not int or cut_point < 0:
                raise ValueError(f"cut_point must be a non-negative integer, got {cut_point}.")
        self.cut_point = cut_point

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
            raise TypeError("parent_a and parent_b must be Genome instances.")
        if len(parent_a) != len(parent_b):
            raise ValueError("Genomes must have the same length for crossover.")

        n = len(parent_a)
        if n <= 1:
            return parent_a, parent_b

        cut = self.cut_point if self.cut_point is not None else random.randint(1, n - 1)
        return n_point_crossover(parent_a, parent_b, [cut])


class TwoPointCrossover(RecombinationStrategy):
    """
    Two-Point Crossover strategy (De Jong, 1975).

    Slices parents at two cut points and exchanges the middle segment.
    If `cut_points` is None, two cut points are chosen uniformly at random.
    """

    def __init__(self, cut_points: Optional[Tuple[int, int]] = None) -> None:
        if cut_points is not None:
            if (
                not isinstance(cut_points, tuple)
                or len(cut_points) != 2
                or any(type(c) is not int or c < 0 for c in cut_points)
            ):
                raise ValueError("cut_points must be a tuple of two non-negative integers.")
        self.cut_points = cut_points

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
            raise TypeError("parent_a and parent_b must be Genome instances.")
        if len(parent_a) != len(parent_b):
            raise ValueError("Genomes must have the same length for crossover.")

        n = len(parent_a)
        if n <= 2:
            return parent_a, parent_b

        if self.cut_points is not None:
            cuts = list(self.cut_points)
        else:
            cuts = sorted(random.sample(range(1, n), 2))

        return n_point_crossover(parent_a, parent_b, cuts)


class NPointCrossover(RecombinationStrategy):
    """A structural crossover strategy that applies N-point sequence crossover at deterministic cut points."""

    def __init__(self, cut_points: list[int]) -> None:
        if not isinstance(cut_points, list):
            raise TypeError("cut_points must be a list of integers")
        if any(not isinstance(c, int) or c < 0 for c in cut_points):
            raise ValueError("Cut points must be non-negative integers")
        self.cut_points = cut_points

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        return n_point_crossover(parent_a, parent_b, self.cut_points)


class RandomNPointCrossover(RecombinationStrategy):
    """Applies structural N-point crossover to parent genomes with randomly selected cut points."""

    def __init__(self, num_cut_points: int = 1) -> None:
        if not isinstance(num_cut_points, int) or num_cut_points < 1:
            raise ValueError("num_cut_points must be an integer >= 1")
        self.num_cut_points = num_cut_points

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
            raise TypeError("parent_a and parent_b must be Genome instances.")
        if len(parent_a) != len(parent_b):
            raise ValueError("Genomes must have the same length for crossover")

        n = len(parent_a)
        if n <= 1:
            return parent_a, parent_b

        num_cuts = min(self.num_cut_points, n - 1)
        cut_points = sorted(random.sample(range(1, n), num_cuts))

        return n_point_crossover(parent_a, parent_b, cut_points)


class UniformCrossover(RecombinationStrategy):
    """
    Uniform Crossover strategy (Syswerda, 1989).

    Each gene in child 1 is inherited from parent A with probability `swap_prob` and from parent B
    otherwise (and vice-versa for child 2).
    """

    def __init__(self, swap_prob: float = 0.5) -> None:
        if not isinstance(swap_prob, (int, float)) or isinstance(swap_prob, bool):
            raise TypeError(f"swap_prob must be a float, got {type(swap_prob).__name__}.")
        if not (0.0 <= swap_prob <= 1.0):
            raise ValueError(f"swap_prob must be in [0.0, 1.0], got {swap_prob}.")
        self.swap_prob = float(swap_prob)

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
            raise TypeError("parent_a and parent_b must be Genome instances.")
        if len(parent_a) != len(parent_b):
            raise ValueError("Genomes must have the same length for crossover.")

        child1_items = []
        child2_items = []

        for a, b in zip(parent_a, parent_b):
            if random.random() < self.swap_prob:
                child1_items.append(a)
                child2_items.append(b)
            else:
                child1_items.append(b)
                child2_items.append(a)

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )


class ShuffleCrossover(RecombinationStrategy):
    """
    Shuffle Crossover strategy (Eshelman, Caruana & Schaffer, 1989).

    Shuffles gene positions identically in both parents, applies an inner crossover strategy
    (defaults to OnePointCrossover), and un-shuffles back to the original index positions.
    """

    def __init__(self, crossover_strategy: Optional[RecombinationStrategy] = None) -> None:
        if crossover_strategy is not None and not isinstance(crossover_strategy, RecombinationStrategy):
            raise TypeError("crossover_strategy must be an instance of RecombinationStrategy.")
        self.crossover_strategy = crossover_strategy or OnePointCrossover()

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
            raise TypeError("parent_a and parent_b must be Genome instances.")
        if len(parent_a) != len(parent_b):
            raise ValueError("Genomes must have the same length for crossover.")

        n = len(parent_a)
        if n <= 1:
            return parent_a, parent_b

        # Create identical random permutation of indices
        indices = list(range(n))
        random.shuffle(indices)

        # Shuffle parents
        shuffled_a = [parent_a[i] for i in indices]
        shuffled_b = [parent_b[i] for i in indices]

        g_a = clone_genome_structure(parent_a, shuffled_a)
        g_b = clone_genome_structure(parent_b, shuffled_b)

        # Apply crossover
        res_a, res_b = self.crossover_strategy(g_a, g_b)  # type: ignore

        # Un-shuffle back to original order
        child1_items = [None] * n
        child2_items = [None] * n
        for orig_idx, shuff_val_a, shuff_val_b in zip(indices, res_a, res_b):
            child1_items[orig_idx] = shuff_val_a
            child2_items[orig_idx] = shuff_val_b

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )
