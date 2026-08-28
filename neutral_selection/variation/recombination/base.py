from typing import Any, Union
from neutral_selection.representation.genome import Genome
from neutral_selection.representation.individual import Individual


class RecombinationStrategy:
    """Base class for all recombination strategies in the library."""

    def __call__(self, parent_a: Genome, parent_b: Genome) -> Union[Genome, tuple[Genome, ...]]:
        raise NotImplementedError("Subclasses must implement __call__")


def recombine(
    parent_a: Any,
    parent_b: Any,
    strategy: RecombinationStrategy
) -> Any:
    """
    Applies a recombination strategy to two parents.
    If parents are Individuals, returns a list of child Individuals.
    If parents are Genomes, returns the resulting child Genome or tuple of Genomes.
    """
    is_individual = isinstance(parent_a, Individual) and isinstance(parent_b, Individual)

    genotype_a = parent_a.genotype if is_individual else parent_a
    genotype_b = parent_b.genotype if is_individual else parent_b

    # Execute strategy
    result = strategy(genotype_a, genotype_b)

    if is_individual:
        # Wrap results in Individuals
        if isinstance(result, tuple):
            return [Individual(genotype=g) for g in result]
        elif isinstance(result, list):
            return [Individual(genotype=g) for g in result]
        else:
            return [Individual(genotype=result)]
    else:
        return result
