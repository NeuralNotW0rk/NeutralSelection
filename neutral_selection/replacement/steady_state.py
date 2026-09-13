from __future__ import annotations

from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import ReplacementStrategy, _extract_individuals, _validate_fitnesses, _validate_k


class SteadyStateReplacement(ReplacementStrategy):
    """
    Steady-State Replacement strategy.

    Retains the best parents and replaces the worst individuals in the parent population with
    the new offspring, maintaining constant population size.
    """

    def __init__(self, minimize: bool = False) -> None:
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")
        self.minimize = minimize

    def replace(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring: Union[Population, Sequence[Individual]],
        target_size: Optional[int] = None,
    ) -> list[Individual]:
        parent_inds = _extract_individuals(parents)
        offspring_inds = _extract_individuals(offspring)

        if target_size is None:
            target_size = len(parent_inds)
        _validate_k(target_size)

        _validate_fitnesses(parent_inds)
        sorted_parents = sorted(
            parent_inds,
            key=lambda ind: ind.fitness,  # type: ignore[return-value]
            reverse=not self.minimize,
        )

        slots_for_parents = max(0, target_size - len(offspring_inds))
        surviving_parents = sorted_parents[:slots_for_parents]

        # If offspring exceeds target_size, take top offspring
        if len(offspring_inds) > target_size:
            _validate_fitnesses(offspring_inds)
            sorted_offspring = sorted(
                offspring_inds,
                key=lambda ind: ind.fitness,  # type: ignore[return-value]
                reverse=not self.minimize,
            )
            return sorted_offspring[:target_size]

        return surviving_parents + offspring_inds
