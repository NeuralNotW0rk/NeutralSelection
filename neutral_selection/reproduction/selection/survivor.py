from __future__ import annotations

from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SurvivorStrategy, _extract_individuals, _validate_fitnesses, _validate_k


class GenerationalReplacement(SurvivorStrategy):
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

    def select_survivors(
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
        if all_offspring_rated:
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


class PlusReplacement(SurvivorStrategy):
    """
    (mu + lambda) Replacement strategy (Evolution Strategies).

    Merges parents (size mu) and offspring (size lambda) into a single candidate pool (size mu + lambda)
    and selects the top `target_size` fittest individuals to form the next generation.
    """

    def __init__(self, minimize: bool = False) -> None:
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")
        self.minimize = minimize

    def select_survivors(
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


class CommaReplacement(SurvivorStrategy):
    """
    (mu, lambda) Replacement strategy (Evolution Strategies).

    Completely replaces parents; selects `target_size` survivors strictly from the offspring pool (lambda).
    Requires lambda >= target_size (usually lambda > mu).
    """

    def __init__(self, minimize: bool = False) -> None:
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")
        self.minimize = minimize

    def select_survivors(
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


class SteadyStateReplacement(SurvivorStrategy):
    """
    Steady-State Replacement strategy.

    Retains the best parents and replaces the worst individuals in the parent population with
    the new offspring, maintaining constant population size.
    """

    def __init__(self, minimize: bool = False) -> None:
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")
        self.minimize = minimize

    def select_survivors(
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
