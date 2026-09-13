from __future__ import annotations

import statistics
from typing import (
    Callable,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Union,
    overload,
    TYPE_CHECKING,
)
from neutral_selection.representation.individual import Individual

if TYPE_CHECKING:
    from neutral_selection.variation.selection.base import SelectionStrategy
    from neutral_selection.variation.recombination.base import RecombinationStrategy
    from neutral_selection.variation.mutation.base import MutationStrategy
    from neutral_selection.replacement.base import ReplacementStrategy
    from neutral_selection.pipeline import GenerationPipeline


class Population:
    """Manages a collection of Individuals and provides coordination methods for selection, variation, and generation advancement."""

    def __init__(self, individuals: Sequence[Individual]) -> None:
        if not isinstance(individuals, (list, tuple)):
            raise TypeError(f"individuals must be a list or tuple of Individuals, got {type(individuals).__name__}")
        for i, ind in enumerate(individuals):
            if not isinstance(ind, Individual):
                raise TypeError(f"Element at index {i} is not an Individual (got {type(ind).__name__}).")
        self.individuals: list[Individual] = list(individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    @overload
    def __getitem__(self, index: int) -> Individual: ...

    @overload
    def __getitem__(self, index: slice) -> list[Individual]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Individual, list[Individual]]:
        return self.individuals[index]

    def __setitem__(self, index: int, value: Individual) -> None:
        if not isinstance(value, Individual):
            raise TypeError(f"Assigned value must be an Individual, got {type(value).__name__}")
        self.individuals[index] = value

    def __contains__(self, item: object) -> bool:
        return item in self.individuals

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def append(self, individual: Individual) -> None:
        """Appends an Individual to the population."""
        if not isinstance(individual, Individual):
            raise TypeError(f"individual must be an Individual, got {type(individual).__name__}")
        self.individuals.append(individual)

    def extend(self, individuals: Iterable[Individual]) -> None:
        """Extends the population with an iterable of Individuals."""
        for ind in individuals:
            self.append(ind)

    @property
    def fitnesses(self) -> list[Optional[float]]:
        """Returns the list of fitness values for all individuals in the population."""
        return [ind.fitness for ind in self.individuals]

    @property
    def best_individual(self) -> Optional[Individual]:
        """Returns the individual with the highest evaluated fitness, or None if unrated/empty."""
        rated = [ind for ind in self.individuals if ind.fitness is not None]
        if not rated:
            return None
        return max(rated, key=lambda ind: ind.fitness)  # type: ignore[arg-type, return-value]

    @property
    def worst_individual(self) -> Optional[Individual]:
        """Returns the individual with the lowest evaluated fitness, or None if unrated/empty."""
        rated = [ind for ind in self.individuals if ind.fitness is not None]
        if not rated:
            return None
        return min(rated, key=lambda ind: ind.fitness)  # type: ignore[arg-type, return-value]

    @property
    def mean_fitness(self) -> Optional[float]:
        """Returns the arithmetic mean of all rated individuals, or None if no evaluated fitnesses."""
        rated_vals = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if not rated_vals:
            return None
        return float(statistics.mean(rated_vals))

    @property
    def max_fitness(self) -> Optional[float]:
        """Returns the maximum fitness in the population, or None if unrated/empty."""
        best = self.best_individual
        return best.fitness if best is not None else None

    @property
    def min_fitness(self) -> Optional[float]:
        """Returns the minimum fitness in the population, or None if unrated/empty."""
        worst = self.worst_individual
        return worst.fitness if worst is not None else None

    def select(self, strategy: SelectionStrategy, k: int = 1) -> list[Individual]:
        """Selects k individuals from this population using the specified SelectionStrategy."""
        from neutral_selection.variation.selection.base import select as _select_fn
        return _select_fn(population=self, strategy=strategy, k=k)

    def replace(
        self,
        offspring: Union[Population, Sequence[Individual]],
        strategy: ReplacementStrategy,
        target_size: Optional[int] = None,
    ) -> Population:
        """Selects surviving individuals from this parent population and an offspring pool using a ReplacementStrategy."""
        from neutral_selection.replacement.base import replace as _replace_fn
        surviving = _replace_fn(parents=self, offspring=offspring, strategy=strategy, target_size=target_size)
        return Population(surviving)

    def step(
        self,
        pipeline: Optional[GenerationPipeline] = None,
        selection_strategy: Optional[SelectionStrategy] = None,
        crossover_strategy: Optional[RecombinationStrategy] = None,
        mutation_strategy: Optional[MutationStrategy] = None,
        replacement_strategy: Optional[ReplacementStrategy] = None,
        offspring_count: Optional[int] = None,
        target_size: Optional[int] = None,
        crossover_prob: float = 1.0,
        mutation_prob: float = 1.0,
        elitism: int = 0,
        elite_ratio: Optional[float] = None,
        evaluate_fn: Optional[Callable[[Individual], float]] = None,
        parent_ids: Optional[Sequence[str]] = None,
    ) -> Population:
        """
        Advances this Population by one generation using either an existing GenerationPipeline or individual strategies.
        """
        if pipeline is not None:
            return pipeline.step(
                parents=self,
                offspring_count=offspring_count,
                target_size=target_size,
                parent_ids=parent_ids,
            )

        if selection_strategy is None:
            raise ValueError("Must provide either a GenerationPipeline instance or at least a selection_strategy.")

        from neutral_selection.pipeline import step as _step_fn
        return _step_fn(
            parents=self,
            selection_strategy=selection_strategy,
            crossover_strategy=crossover_strategy,
            mutation_strategy=mutation_strategy,
            replacement_strategy=replacement_strategy,
            offspring_count=offspring_count,
            target_size=target_size,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            elitism=elitism,
            elite_ratio=elite_ratio,
            evaluate_fn=evaluate_fn,
            parent_ids=parent_ids,
        )

    def clone(self, deep: bool = True) -> Population:
        """Creates a shallow or deep copy of this Population."""
        return Population([ind.clone(deep=deep) for ind in self.individuals])
