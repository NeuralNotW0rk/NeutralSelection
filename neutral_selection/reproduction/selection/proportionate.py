from __future__ import annotations

import bisect
import random
from typing import Optional, Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SelectionStrategy, _extract_individuals, _validate_fitnesses, _validate_k


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
            "Total population fitness/weight is 0 (or non-positive). "
            "Cannot perform proportionate selection with zero total weight."
        )

    return weights


class RouletteWheelSelection(SelectionStrategy):
    """
    Fitness Proportionate Selection (Roulette Wheel Selection).

    The probability of selecting an individual is directly proportional to its fitness.
    For minimization problems, fitnesses are inverted relative to the maximum population fitness.
    An optional `fitness_offset` can be specified to shift values to ensure strictly positive weights.
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

        weights = _prepare_proportionate_weights(raw_fitnesses, self.fitness_offset, self.minimize)
        total = sum(weights)

        cum_weights: list[float] = []
        acc = 0.0
        for w in weights:
            acc += w
            cum_weights.append(acc)

        selected: list[Individual] = []
        for _ in range(k):
            r = random.uniform(0.0, total)
            idx = bisect.bisect_right(cum_weights, r)
            idx = min(idx, len(inds) - 1)
            selected.append(inds[idx])

        return selected


class StochasticUniversalSamplingSelection(SelectionStrategy):
    """
    Stochastic Universal Sampling (SUS) selection strategy (Baker, 1987).

    SUS operates like a single-spin roulette wheel with `k` equally-spaced pointers.
    This guarantees zero sampling bias and minimal spread, ensuring that individuals
    with expected selection count E are selected either floor(E) or ceil(E) times.
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

        weights = _prepare_proportionate_weights(raw_fitnesses, self.fitness_offset, self.minimize)
        total = sum(weights)

        step = total / k
        start = random.uniform(0.0, step)
        pointers = [start + i * step for i in range(k)]

        selected: list[Individual] = []
        current_idx = 0
        cum_weight = weights[0]

        for pointer in pointers:
            while pointer >= cum_weight and current_idx < len(inds) - 1:
                current_idx += 1
                cum_weight += weights[current_idx]
            selected.append(inds[current_idx])

        return selected
