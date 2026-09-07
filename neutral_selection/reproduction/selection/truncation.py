from __future__ import annotations

import random
from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SelectionStrategy, _extract_individuals, _validate_fitnesses, _validate_k


class TruncationSelection(SelectionStrategy):
    """
    Truncation Selection strategy (Mühlenbein & Schlierkamp-Voosen, 1993).

    Sorts the population by fitness and restricts the candidate pool to the top `top_k` individuals
    or top `top_ratio` percentage of the population. Individuals are sampled uniformly from this pool.

    Widely used in Evolution Strategies (ES) and Breeder Genetic Algorithms (BGA).
    """

    def __init__(
        self,
        top_k: Optional[int] = None,
        top_ratio: Optional[float] = None,
        with_replacement: bool = True,
        minimize: bool = False,
    ) -> None:
        if top_k is None and top_ratio is None:
            raise ValueError("Either top_k or top_ratio must be provided for TruncationSelection.")
        if top_k is not None and top_ratio is not None:
            raise ValueError("Cannot specify both top_k and top_ratio simultaneously.")

        if top_k is not None:
            if type(top_k) is not int or top_k < 1:
                raise ValueError(f"top_k must be an integer >= 1, got {top_k}.")
            self.top_k: Optional[int] = top_k
        else:
            self.top_k = None

        if top_ratio is not None:
            if not isinstance(top_ratio, (int, float)) or isinstance(top_ratio, bool):
                raise TypeError(f"top_ratio must be a float or int, got {type(top_ratio).__name__}.")
            if not (0.0 < top_ratio <= 1.0):
                raise ValueError(f"top_ratio must be in (0.0, 1.0], got {top_ratio}.")
            self.top_ratio: Optional[float] = float(top_ratio)
        else:
            self.top_ratio = None

        if not isinstance(with_replacement, bool):
            raise TypeError(f"with_replacement must be a boolean, got {type(with_replacement).__name__}.")
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.with_replacement = with_replacement
        self.minimize = minimize

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        inds = _extract_individuals(population)
        _validate_fitnesses(inds)
        _validate_k(k)

        sorted_inds = sorted(
            inds,
            key=lambda ind: ind.fitness,  # type: ignore[return-value]
            reverse=not self.minimize,
        )

        n = len(sorted_inds)
        if self.top_k is not None:
            pool_size = min(n, self.top_k)
        else:
            assert self.top_ratio is not None
            pool_size = max(1, round(n * self.top_ratio))

        pool = sorted_inds[:pool_size]

        if self.with_replacement:
            return [random.choice(pool) for _ in range(k)]
        else:
            if k > pool_size:
                raise ValueError(
                    f"Cannot select {k} unique individuals without replacement from truncation pool of size {pool_size}."
                )
            return random.sample(pool, k)


class ElitistSelection(SelectionStrategy):
    """
    Elitist Selection strategy.

    Deterministically selects the top `k` (or top `num_elites` / `elite_ratio`) fittest unique
    individuals directly from the population without replacement.

    This ensures preservation of top-performing genetic material into subsequent generations or mating pools.
    """

    def __init__(
        self,
        num_elites: Optional[int] = None,
        elite_ratio: Optional[float] = None,
        minimize: bool = False,
    ) -> None:
        if num_elites is not None and elite_ratio is not None:
            raise ValueError("Cannot specify both num_elites and elite_ratio simultaneously.")

        if num_elites is not None:
            if type(num_elites) is not int or num_elites < 1:
                raise ValueError(f"num_elites must be an integer >= 1, got {num_elites}.")
            self.num_elites: Optional[int] = num_elites
        else:
            self.num_elites = None

        if elite_ratio is not None:
            if not isinstance(elite_ratio, (int, float)) or isinstance(elite_ratio, bool):
                raise TypeError(f"elite_ratio must be a float or int, got {type(elite_ratio).__name__}.")
            if not (0.0 < elite_ratio <= 1.0):
                raise ValueError(f"elite_ratio must be in (0.0, 1.0], got {elite_ratio}.")
            self.elite_ratio: Optional[float] = float(elite_ratio)
        else:
            self.elite_ratio = None

        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")
        self.minimize = minimize

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        inds = _extract_individuals(population)
        _validate_fitnesses(inds)

        sorted_inds = sorted(
            inds,
            key=lambda ind: ind.fitness,  # type: ignore[return-value]
            reverse=not self.minimize,
        )

        n = len(sorted_inds)
        if self.num_elites is not None:
            count = min(n, self.num_elites)
        elif self.elite_ratio is not None:
            count = max(1, round(n * self.elite_ratio))
        else:
            _validate_k(k, max_k=n)
            count = k

        return sorted_inds[:count]
