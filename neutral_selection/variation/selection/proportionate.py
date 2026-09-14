from __future__ import annotations

import bisect
import random
from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SelectionStrategy, _extract_individuals, _validate_fitnesses, _validate_k
from neutral_selection.registry import register_selection


def _prepare_proportionate_weights(
    raw_fitnesses: list[float],
    fitness_offset: Optional[float],
    minimize: bool,
) -> list[float]:
    """
    Transforms and validates raw fitnesses into non-negative selection weights.
    Raises ValueError if weights are negative or total weight is non-positive.
    """
    if minimize:
        max_f = max(raw_fitnesses)
        offset = 0.0 if fitness_offset is None else float(fitness_offset)
        weights = [max_f - f + offset for f in raw_fitnesses]
    else:
        if fitness_offset is not None:
            weights = [f + float(fitness_offset) for f in raw_fitnesses]
        else:
            weights = list(raw_fitnesses)

    min_w = min(weights)
    if min_w < 0:
        raise ValueError(
            f"Negative fitness/weight encountered ({min_w}) during proportionate selection. "
            "All weights must be non-negative. Provide a sufficient fitness_offset or use "
            "LinearRankSelection / TournamentSelection instead."
        )

    total = sum(weights)
    if total <= 0:
        raise ValueError(
            f"Total fitness sum ({total}) must be strictly positive for proportionate selection."
        )

    return weights


@register_selection(["roulette", "roulette_wheel", "proportionate"])
class RouletteWheelSelection(SelectionStrategy):
    """
    Fitness-Proportionate / Roulette Wheel Selection strategy.

    Selects individuals with probability strictly proportional to their evaluated fitness:
        p(i) = f(i) / sum(f(j))

    Requires non-negative fitness values. An optional `fitness_offset` can be supplied to shift
    values if negative fitnesses are present.
    """

    def __init__(
        self,
        fitness_offset: Optional[float] = None,
        minimize: bool = False,
    ) -> None:
        if fitness_offset is not None:
            if not isinstance(fitness_offset, (int, float)) or isinstance(fitness_offset, bool):
                raise TypeError(f"fitness_offset must be a float or int, got {type(fitness_offset).__name__}.")
            self.fitness_offset: Optional[float] = float(fitness_offset)
        else:
            self.fitness_offset = None

        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.minimize = minimize

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        inds = _extract_individuals(population)
        raw_fitnesses = _validate_fitnesses(inds)
        _validate_k(k)

        weights = _prepare_proportionate_weights(
            raw_fitnesses=raw_fitnesses,
            fitness_offset=self.fitness_offset,
            minimize=self.minimize,
        )

        total_w = sum(weights)
        cum_weights: list[float] = []
        acc = 0.0
        for w in weights:
            acc += w / total_w
            cum_weights.append(acc)

        selected: list[Individual] = []
        for _ in range(k):
            r = random.random()
            idx = bisect.bisect_right(cum_weights, r)
            idx = min(idx, len(inds) - 1)
            selected.append(inds[idx])

        return selected


@register_selection(["stochastic_universal_sampling", "sus"])
class StochasticUniversalSamplingSelection(SelectionStrategy):
    """
    Stochastic Universal Sampling (SUS) Selection strategy (Baker, 1987).

    An optimal variant of roulette wheel selection that places `k` equally-spaced pointers
    along a 1D selection line (distance 1/k apart). A single random spin in [0, 1/k) determines
    all selected individuals simultaneously.

    SUS exhibits zero bias and minimal spread, preventing premature convergence caused by
    statistical noise in standard roulette wheel selection.
    """

    def __init__(
        self,
        fitness_offset: Optional[float] = None,
        minimize: bool = False,
    ) -> None:
        if fitness_offset is not None:
            if not isinstance(fitness_offset, (int, float)) or isinstance(fitness_offset, bool):
                raise TypeError(f"fitness_offset must be a float or int, got {type(fitness_offset).__name__}.")
            self.fitness_offset: Optional[float] = float(fitness_offset)
        else:
            self.fitness_offset = None

        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.minimize = minimize

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        inds = _extract_individuals(population)
        raw_fitnesses = _validate_fitnesses(inds)
        _validate_k(k)

        weights = _prepare_proportionate_weights(
            raw_fitnesses=raw_fitnesses,
            fitness_offset=self.fitness_offset,
            minimize=self.minimize,
        )

        total_w = sum(weights)
        step_size = total_w / k
        start_point = random.uniform(0.0, step_size)

        pointers = [start_point + i * step_size for i in range(k)]

        selected: list[Individual] = []
        cur_sum = weights[0]
        ind_idx = 0
        n = len(inds)

        for p in pointers:
            while p > cur_sum and ind_idx < n - 1:
                ind_idx += 1
                cur_sum += weights[ind_idx]
            selected.append(inds[ind_idx])

        return selected
