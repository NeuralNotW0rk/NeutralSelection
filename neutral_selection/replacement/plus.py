from __future__ import annotations

from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import ReplacementStrategy, _extract_individuals, _validate_fitnesses, _validate_k


class PlusReplacement(ReplacementStrategy):
    """
    (mu + lambda) Replacement strategy (Evolution Strategies).

    Merges parents (size mu) and offspring (size lambda) into a single candidate pool (size mu + lambda)
    and selects the top `target_size` fittest individuals to form the next generation.
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

        combined = parent_inds + offspring_inds
        if len(combined) < target_size:
            raise ValueError(
                f"Combined pool size ({len(combined)}) is smaller than target_size ({target_size})."
            )

        _validate_fitnesses(combined)
        sorted_combined = sorted(
            combined,
            key=lambda ind: ind.fitness,  # type: ignore[return-value]
            reverse=not self.minimize,
        )

        return sorted_combined[:target_size]
