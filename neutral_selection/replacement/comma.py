from __future__ import annotations

from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import ReplacementStrategy, _extract_individuals, _validate_fitnesses, _validate_k
from neutral_selection.registry import register_replacement


@register_replacement(["comma", "mu_comma_lambda", "comma_replacement"])
class CommaReplacement(ReplacementStrategy):
    """
    (mu, lambda) Replacement strategy (Evolution Strategies).

    Completely replaces parents; selects `target_size` survivors strictly from the offspring pool (lambda).
    Requires lambda >= target_size (usually lambda > mu).
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

        if len(offspring_inds) < target_size:
            raise ValueError(
                f"CommaReplacement requires offspring count ({len(offspring_inds)}) >= target_size ({target_size})."
            )

        _validate_fitnesses(offspring_inds)
        sorted_offspring = sorted(
            offspring_inds,
            key=lambda ind: ind.fitness,  # type: ignore[return-value]
            reverse=not self.minimize,
        )

        return sorted_offspring[:target_size]
