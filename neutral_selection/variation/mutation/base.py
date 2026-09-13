from typing import Any
from neutral_selection.representation.genome import Genome
from neutral_selection.representation.individual import Individual


class MutationStrategy:
    """Base class for all mutation strategies in the library."""

    def __call__(self, genome: Genome) -> Genome:
        raise NotImplementedError("Subclasses must implement __call__")


def mutate(
    target: Any,
    strategy: MutationStrategy
) -> Any:
    """
    Applies a mutation strategy to a target.
    If target is an Individual, returns a new Individual wrapping the mutated Genome.
    If target is a Genome, returns the mutated Genome directly.
    """
    if strategy is None:
        raise ValueError("strategy must be provided and cannot be None.")
    if not isinstance(strategy, MutationStrategy):
        raise TypeError("strategy must be an instance of MutationStrategy.")

    is_individual = isinstance(target, Individual)
    is_genome = isinstance(target, Genome)

    if not is_individual and not is_genome:
        raise TypeError("target must be an instance of Genome or Individual.")

    genotype = target.genotype if is_individual else target

    # Execute strategy
    result = strategy(genotype)

    if not isinstance(result, Genome):
        raise TypeError(f"Mutation strategy returned {type(result).__name__}, expected Genome.")

    if is_individual:
        return Individual(genotype=result)
    else:
        return result
