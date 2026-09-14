from __future__ import annotations

from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import ReplacementStrategy, _extract_individuals, _validate_fitnesses, _validate_k
from neutral_selection.registry import register_replacement


@register_replacement(["generational", "generational_replacement"])
class GenerationalReplacement(ReplacementStrategy):
    """
    Generational Replacement strategy with Elitism.

    Forms the next generation primarily from newly generated offspring, while preserving the
    top `num_elites` (or top `elite_ratio` fraction) fittest individuals from the parent generation unchanged.
    """

    def __init__(
        self,
        num_elites: int = 0,
        elite_ratio: Optional[float] = None,
        minimize: bool = False,
    ) -> None:
        if num_elites < 0:
            raise ValueError(f"num_elites must be >= 0, got {num_elites}.")
        if elite_ratio is not None:
            if not isinstance(elite_ratio, (int, float)) or isinstance(elite_ratio, bool):
                raise TypeError(f"elite_ratio must be a float or int, got {type(elite_ratio).__name__}.")
            if not (0.0 <= elite_ratio <= 1.0):
                raise ValueError(f"elite_ratio must be between 0.0 and 1.0 inclusive, got {elite_ratio}.")
            self.elite_ratio: Optional[float] = float(elite_ratio)
        else:
            self.elite_ratio = None

        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.num_elites = num_elites
        self.minimize = minimize

    def replace(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring: Union[Population, Sequence[Individual]],
        target_size: Optional[int] = None,
    ) -> list[Individual]:
        parent_inds = _extract_individuals(parents)
        offspring_inds = _extract_individuals(offspring, allow_empty=True)

        if target_size is None:
            target_size = len(parent_inds)
        _validate_k(target_size)

        if self.elite_ratio is not None:
            elite_count = min(target_size, max(0, round(len(parent_inds) * self.elite_ratio)))
        else:
            elite_count = min(target_size, self.num_elites)

        elites: list[Individual] = []
        if elite_count > 0:
            _validate_fitnesses(parent_inds)
            sorted_parents = sorted(
                parent_inds,
                key=lambda ind: ind.fitness,  # type: ignore[return-value]
                reverse=not self.minimize,
            )
            elites = sorted_parents[:elite_count]

        needed_offspring = target_size - elite_count
        if len(offspring_inds) < needed_offspring:
            raise ValueError(
                f"Not enough offspring ({len(offspring_inds)}) to fill remaining {needed_offspring} survivor slots "
                f"(target_size={target_size}, elite_count={elite_count})."
            )

        # If offspring have fitnesses evaluated, sort by fitness; otherwise take in order
        all_offspring_rated = all(ind.fitness is not None for ind in offspring_inds)
        if all_offspring_rated and needed_offspring > 0:
            _validate_fitnesses(offspring_inds)
            sorted_offspring = sorted(
                offspring_inds,
                key=lambda ind: ind.fitness,  # type: ignore[return-value]
                reverse=not self.minimize,
            )
            selected_offspring = sorted_offspring[:needed_offspring]
        else:
            selected_offspring = offspring_inds[:needed_offspring]

        return elites + selected_offspring
