from __future__ import annotations

import bisect
import math
import random
from typing import Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SelectionStrategy, _extract_individuals, _validate_fitnesses, _validate_k


class BoltzmannSelection(SelectionStrategy):
    """
    Boltzmann / Softmax Selection strategy (Mahfoud, 1995; De la Maza & Tidor, 1993).

    Assigns selection probabilities via a Boltzmann/Gibbs distribution:
        p(i) = exp(s_i / T) / sum(exp(s_j / T))
    where s_i is fitness (or -fitness for minimization) and T > 0 is the temperature parameter.

    Allows simulating annealing in genetic algorithms:
    - High temperature (T >> 1): nearly uniform selection, encouraging wide exploration.
    - Low temperature (T -> 0): greedy selection, strongly favoring the highest-fitness individuals.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        minimize: bool = False,
    ) -> None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError(f"temperature must be a float or int, got {type(temperature).__name__}.")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive (> 0.0), got {temperature}.")
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.temperature = float(temperature)
        self.minimize = minimize

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        inds = _extract_individuals(population)
        raw_fitnesses = _validate_fitnesses(inds)
        _validate_k(k)

        t = self.temperature
        if self.minimize:
            scores = [-f / t for f in raw_fitnesses]
        else:
            scores = [f / t for f in raw_fitnesses]

        # Log-sum-exp stabilization to avoid overflow/underflow
        max_s = max(scores)
        weights = [math.exp(s - max_s) for s in scores]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        cum_probs: list[float] = []
        acc = 0.0
        for p in probs:
            acc += p
            cum_probs.append(acc)

        selected: list[Individual] = []
        for _ in range(k):
            r = random.random()
            idx = bisect.bisect_right(cum_probs, r)
            idx = min(idx, len(inds) - 1)
            selected.append(inds[idx])

        return selected
