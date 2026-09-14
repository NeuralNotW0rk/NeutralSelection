from __future__ import annotations

import random
from typing import Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SelectionStrategy, _extract_individuals, _validate_k
from neutral_selection.registry import register_selection


@register_selection("random")
class RandomSelection(SelectionStrategy):
    """
    Random / Uniform Selection strategy.

    Samples individuals uniformly at random from the population regardless of fitness.
    Serves as an essential mechanism for modeling neutral genetic drift (Kimura's Neutral Theory),
    exploratory diversity, and baseline evolutionary benchmarking.
    """

    def __init__(self, with_replacement: bool = True) -> None:
        if not isinstance(with_replacement, bool):
            raise TypeError(f"with_replacement must be a boolean, got {type(with_replacement).__name__}.")
        self.with_replacement = with_replacement

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        inds = _extract_individuals(population)

        if self.with_replacement:
            _validate_k(k)
            return [random.choice(inds) for _ in range(k)]
        else:
            _validate_k(k, max_k=len(inds))
            return random.sample(inds, k)
