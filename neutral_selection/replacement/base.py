from __future__ import annotations

from typing import Sequence, Union, Optional
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.variation.selection.base import (
    _extract_individuals,
    _validate_fitnesses,
    _validate_k,
)


class ReplacementStrategy:
    """Base class for all environmental replacement strategies in the library."""

    def replace(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring: Union[Population, Sequence[Individual]],
        target_size: Optional[int] = None,
    ) -> list[Individual]:
        """
        Selects target_size individuals from parents and offspring to form the next generation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement replace()")

    def __call__(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring: Union[Population, Sequence[Individual]],
        target_size: Optional[int] = None,
    ) -> list[Individual]:
        """Convenience callable alias for replace()."""
        return self.replace(parents, offspring, target_size=target_size)


def replace(
    parents: Union[Population, Sequence[Individual]],
    offspring: Union[Population, Sequence[Individual]],
    strategy: ReplacementStrategy,
    target_size: Optional[int] = None,
) -> list[Individual]:
    """
    Applies a replacement strategy to parents and offspring.
    Returns a list of surviving Individuals for the next generation.
    """
    if strategy is None:
        raise ValueError("strategy must be provided and cannot be None.")
    if not isinstance(strategy, ReplacementStrategy):
        raise TypeError(f"strategy must be an instance of ReplacementStrategy, got {type(strategy).__name__}.")

    return strategy.replace(parents, offspring, target_size=target_size)
