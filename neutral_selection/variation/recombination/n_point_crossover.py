import random
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.variation.recombination.base import RecombinationStrategy


def clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """
    Creates a new genome instance matching the type and metadata of the original.
    """
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
        clone_genome_structure(parent_b, child2_items)
    )


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
