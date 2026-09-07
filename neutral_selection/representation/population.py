from __future__ import annotations
from typing import Iterable, Any, Optional
from neutral_selection.representation.individual import Individual


class Population:
    """Manages a collection of Individuals and provides convenience methods."""

    def __init__(self, individuals: list[Individual]) -> None:
        if not isinstance(individuals, list):
            raise TypeError("individuals must be a list")
        self.individuals: list[Individual] = list(individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, index: int) -> Individual:
        return self.individuals[index]

    def __iter__(self) -> Iterable[Individual]:
        return iter(self.individuals)

    @property
    def best_individual(self) -> Individual | None:
        """Returns the individual with the highest fitness."""
        rated = [ind for ind in self.individuals if ind.fitness is not None]
        if not rated:
            return None
        return max(rated, key=lambda ind: ind.fitness)
