import random
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.variation.mutation.base import MutationStrategy


def _clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


class InversionMutation(MutationStrategy):
    """
    A sequence-level mutation strategy that selects a random subsequence
    within the genome and reverses the order of its elements.
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
    A sequence-level mutation strategy that selects two random elements
    within the genome sequence and swaps their positions.
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
    A sequence-level mutation strategy that selects a random subsequence
    within the genome sequence and shuffles the elements within that subsequence.
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
