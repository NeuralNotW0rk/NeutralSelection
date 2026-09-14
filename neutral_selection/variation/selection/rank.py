from __future__ import annotations

import bisect
import random
from typing import Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SelectionStrategy, _extract_individuals, _validate_fitnesses, _validate_k
from neutral_selection.registry import register_selection


@register_selection(["linear_rank", "linear"])
class LinearRankSelection(SelectionStrategy):
    """
    Linear Rank Selection strategy (Baker, 1985; Whitley, 1989).

    Sorts individuals by fitness and assigns selection probabilities linearly based on their rank:
        p(i) = (1 / N) * ((2 - s) + 2 * (s - 1) * (i / (N - 1)))
    where i in [0, N-1] is the 0-indexed rank from worst to best, and s in [1.0, 2.0] is the
    selection pressure parameter (1.0 = uniform random, 2.0 = maximum linear bias toward top rank).
    """

    def __init__(
        self,
        selection_pressure: float = 1.5,
        minimize: bool = False,
    ) -> None:
        if not isinstance(selection_pressure, (int, float)) or isinstance(selection_pressure, bool):
            raise TypeError(
                f"selection_pressure must be a float or int, got {type(selection_pressure).__name__}."
            )
        if not (1.0 <= selection_pressure <= 2.0):
            raise ValueError(
                f"selection_pressure must be between 1.0 and 2.0 inclusive, got {selection_pressure}."
            )
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.selection_pressure = float(selection_pressure)
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
            reverse=self.minimize,
        )

        n = len(sorted_inds)
        if n == 1:
            return [sorted_inds[0] for _ in range(k)]

        s = self.selection_pressure
        probs = [(1.0 / n) * ((2.0 - s) + 2.0 * (s - 1.0) * (i / (n - 1))) for i in range(n)]

        cum_probs: list[float] = []
        acc = 0.0
        for p in probs:
            acc += p
            cum_probs.append(acc)

        selected: list[Individual] = []
        for _ in range(k):
            r = random.random()
            idx = bisect.bisect_right(cum_probs, r)
            idx = min(idx, n - 1)
            selected.append(sorted_inds[idx])

        return selected


@register_selection(["exponential_rank", "exponential"])
class ExponentialRankSelection(SelectionStrategy):
    """
    Exponential Rank Selection strategy (Blickle & Thiele, 1995).

    Assigns selection probabilities exponentially according to rank:
        w(i) = c ** (N - 1 - i)
    where i in [0, N-1] is the rank from worst to best, and c in (0.0, 1.0) is the base parameter.
    """

    def __init__(
        self,
        c: float = 0.9,
        minimize: bool = False,
    ) -> None:
        if not isinstance(c, (int, float)) or isinstance(c, bool):
            raise TypeError(f"c must be a float or int, got {type(c).__name__}.")
        if not (0.0 < c < 1.0):
            raise ValueError(f"c must be strictly between 0.0 and 1.0, got {c}.")
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.c = float(c)
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
            reverse=self.minimize,
        )

        n = len(sorted_inds)
        if n == 1:
            return [sorted_inds[0] for _ in range(k)]

        weights = [self.c ** (n - 1 - i) for i in range(n)]
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
            idx = min(idx, n - 1)
            selected.append(sorted_inds[idx])

        return selected
