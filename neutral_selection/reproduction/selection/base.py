from __future__ import annotations

from typing import Any, Sequence, Union, Tuple, List, Optional
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population


def _extract_individuals(population: Any) -> list[Individual]:
    """
    Extracts and validates a list of Individuals from a Population or sequence.
    Raises descriptive errors if invalid or empty.
    """
    if population is None:
        raise ValueError("population must be provided and cannot be None.")

    if isinstance(population, Population):
        inds = list(population.individuals)
    elif isinstance(population, (list, tuple)):
        inds = list(population)
    else:
        raise TypeError(
            f"population must be an instance of Population or Sequence of Individuals, got {type(population).__name__}."
        )

    if not inds:
        raise ValueError("Population is empty. Cannot perform selection on an empty population.")

    for i, ind in enumerate(inds):
        if not isinstance(ind, Individual):
            raise TypeError(
                f"Element at index {i} is not an Individual (got {type(ind).__name__})."
            )

    return inds


def _validate_fitnesses(individuals: list[Individual]) -> list[float]:
    """
    Validates that all individuals in the list have numeric fitness evaluated.
    Returns the list of float fitness values.
    """
    fitnesses: list[float] = []
    for i, ind in enumerate(individuals):
        if ind.fitness is None:
            raise ValueError(
                f"Individual at index {i} has unassigned (None) fitness. "
                "All individuals must have fitness evaluated before selection."
            )
        if not isinstance(ind.fitness, (int, float)) or isinstance(ind.fitness, bool):
            raise TypeError(
                f"Individual at index {i} has invalid fitness type: {type(ind.fitness).__name__} (expected float or int)."
            )
        fitnesses.append(float(ind.fitness))
    return fitnesses


def _validate_k(k: Any, max_k: Optional[int] = None, allow_zero: bool = False) -> int:
    """Validates the number of individuals to select (k)."""
    if type(k) is not int:
        raise TypeError(f"k must be an integer, got {type(k).__name__}.")
    min_k = 0 if allow_zero else 1
    if k < min_k:
        raise ValueError(f"k must be an integer >= {min_k}, got {k}.")
    if max_k is not None and k > max_k:
        raise ValueError(f"k cannot exceed population size ({max_k}), got {k}.")
    return k


class SelectionStrategy:
    """Base class for all parent/mating selection strategies in the library."""

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        """
        Selects k individuals from the provided population.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement select()")

    def __call__(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        """Convenience callable alias for select()."""
        return self.select(population, k=k)

    def select_one(
        self,
        population: Union[Population, Sequence[Individual]],
    ) -> Individual:
        """Convenience method to select a single individual."""
        selected = self.select(population, k=1)
        if not selected:
            raise ValueError("Selection strategy returned an empty list.")
        return selected[0]

    def select_pairs(
        self,
        population: Union[Population, Sequence[Individual]],
        num_pairs: int = 1,
    ) -> list[tuple[Individual, Individual]]:
        """
        Selects pairs of individuals (e.g. for two-parent recombination).
        Returns a list of (parent_a, parent_b) tuples.
        """
        if type(num_pairs) is not int or num_pairs < 1:
            raise ValueError(f"num_pairs must be an integer >= 1, got {num_pairs}.")

        selected = self.select(population, k=num_pairs * 2)
        if len(selected) < num_pairs * 2:
            raise ValueError(
                f"Expected {num_pairs * 2} individuals for {num_pairs} pairs, but received {len(selected)}."
            )

        return [(selected[i], selected[i + 1]) for i in range(0, num_pairs * 2, 2)]


class SurvivorStrategy:
    """Base class for survivor selection / replacement strategies in the library."""

    def select_survivors(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring: Union[Population, Sequence[Individual]],
        target_size: Optional[int] = None,
    ) -> list[Individual]:
        """
        Selects target_size survivors from parents and offspring to form the next generation.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement select_survivors()")

    def __call__(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring: Union[Population, Sequence[Individual]],
        target_size: Optional[int] = None,
    ) -> list[Individual]:
        """Convenience callable alias for select_survivors()."""
        return self.select_survivors(parents, offspring, target_size=target_size)


def select(
    population: Union[Population, Sequence[Individual]],
    strategy: SelectionStrategy,
    k: int = 1,
) -> list[Individual]:
    """
    Applies a selection strategy to a population or sequence of individuals.
    Returns a list of selected Individuals.
    """
    if strategy is None:
        raise ValueError("strategy must be provided and cannot be None.")
    if not isinstance(strategy, SelectionStrategy):
        raise TypeError(f"strategy must be an instance of SelectionStrategy, got {type(strategy).__name__}.")

    return strategy.select(population, k=k)


def select_survivors(
    parents: Union[Population, Sequence[Individual]],
    offspring: Union[Population, Sequence[Individual]],
    strategy: SurvivorStrategy,
    target_size: Optional[int] = None,
) -> list[Individual]:
    """
    Applies a survivor replacement strategy to parents and offspring.
    Returns a list of surviving Individuals for the next generation.
    """
    if strategy is None:
        raise ValueError("strategy must be provided and cannot be None.")
    if not isinstance(strategy, SurvivorStrategy):
        raise TypeError(f"strategy must be an instance of SurvivorStrategy, got {type(strategy).__name__}.")

    return strategy.select_survivors(parents, offspring, target_size=target_size)
