import random
from typing import Callable, Any
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.variation.mutation.base import MutationStrategy


class UniformMutation(MutationStrategy):
    """
    A structure-agnostic mutation strategy that maps a mutation function across elements recursively.
    Supports flat genomes, nested genomes, and segment hierarchies.
    """

    def __init__(self, mutation_rate: float, mutation_fn: Callable[[Any], Any]) -> None:
        if mutation_fn is None:
            raise ValueError("mutation_fn must be provided and cannot be None.")
        if not callable(mutation_fn):
            raise TypeError("mutation_fn must be a callable.")
        if not isinstance(mutation_rate, (int, float)):
            raise TypeError("mutation_rate must be a float or int.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0 inclusive.")
        self.mutation_rate = float(mutation_rate)
        self.mutation_fn = mutation_fn

    def __call__(self, genome: Genome) -> Genome:
        if not isinstance(genome, Genome):
            raise TypeError("genome must be an instance of Genome.")

        mutated_items = []
        for item in genome:
            if isinstance(item, Genome):
                # Recursively mutate nested sub-genomes/segments
                mutated_item = self(item)
            else:
                mutated_item = self.mutation_fn(item) if random.random() < self.mutation_rate else item
            mutated_items.append(mutated_item)

        if isinstance(genome, Segment):
            return genome.__class__(genome.key, mutated_items)
        return genome.__class__(mutated_items)
