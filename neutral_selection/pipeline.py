from __future__ import annotations

import copy
import random
from typing import Optional, Sequence, Union, Callable

from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.representation.lineage import Lineage
from neutral_selection.variation.selection.base import (
    SelectionStrategy,
    _extract_individuals,
    _validate_k,
)
from neutral_selection.variation.recombination.base import RecombinationStrategy, recombine
from neutral_selection.variation.mutation.base import MutationStrategy, mutate
from neutral_selection.replacement.base import ReplacementStrategy
from neutral_selection.replacement.generational import GenerationalReplacement


class GenerationPipeline:
    """
    Coordinates parent selection, variation (recombination and mutation), evaluation, and survivor replacement
    to advance an evolutionary population by one generation.

    The generational step is executed in two decoupled phases:
    1. Variation Phase: Selects parents and produces candidate offspring via crossover, mutation, and optional evaluation.
    2. Replacement Phase: Delegates next-generation survivor selection to the configured ReplacementStrategy
       (e.g., GenerationalReplacement with elitism, PlusReplacement, CommaReplacement, SteadyStateReplacement).

    Attributes:
        selection_strategy: SelectionStrategy used to select parent mating pairs.
        crossover_strategy: Optional RecombinationStrategy used to recombine parent genotypes.
        mutation_strategy: Optional MutationStrategy used to mutate offspring genotypes.
        replacement_strategy: ReplacementStrategy governing parent/offspring replacement into the next generation.
        crossover_prob: Probability of applying crossover to a selected parent pair (0.0 to 1.0).
        mutation_prob: Probability of applying mutation to a child genotype (0.0 to 1.0).
        evaluate_fn: Optional fitness evaluation callable (Individual -> float) applied to new offspring.
    """

    def __init__(
        self,
        selection_strategy: SelectionStrategy,
        crossover_strategy: Optional[RecombinationStrategy] = None,
        mutation_strategy: Optional[MutationStrategy] = None,
        replacement_strategy: Optional[ReplacementStrategy] = None,
        crossover_prob: float = 1.0,
        mutation_prob: float = 1.0,
        elitism: int = 0,
        elite_ratio: Optional[float] = None,
        evaluate_fn: Optional[Callable[[Individual], float]] = None,
    ) -> None:
        if selection_strategy is None:
            raise ValueError("selection_strategy must be provided and cannot be None.")
        if not isinstance(selection_strategy, SelectionStrategy):
            raise TypeError(
                f"selection_strategy must be an instance of SelectionStrategy, got {type(selection_strategy).__name__}."
            )

        if crossover_strategy is not None and not isinstance(crossover_strategy, RecombinationStrategy):
            raise TypeError(
                f"crossover_strategy must be an instance of RecombinationStrategy or None, got {type(crossover_strategy).__name__}."
            )

        if mutation_strategy is not None and not isinstance(mutation_strategy, MutationStrategy):
            raise TypeError(
                f"mutation_strategy must be an instance of MutationStrategy or None, got {type(mutation_strategy).__name__}."
            )

        if not isinstance(crossover_prob, (int, float)) or isinstance(crossover_prob, bool):
            raise TypeError(f"crossover_prob must be a float or int, got {type(crossover_prob).__name__}.")
        if not (0.0 <= float(crossover_prob) <= 1.0):
            raise ValueError(f"crossover_prob must be between 0.0 and 1.0, got {crossover_prob}.")

        if not isinstance(mutation_prob, (int, float)) or isinstance(mutation_prob, bool):
            raise TypeError(f"mutation_prob must be a float or int, got {type(mutation_prob).__name__}.")
        if not (0.0 <= float(mutation_prob) <= 1.0):
            raise ValueError(f"mutation_prob must be between 0.0 and 1.0, got {mutation_prob}.")

        if evaluate_fn is not None and not callable(evaluate_fn):
            raise TypeError("evaluate_fn must be a callable or None.")

        # Resolve replacement strategy
        if replacement_strategy is not None:
            if not isinstance(replacement_strategy, ReplacementStrategy):
                raise TypeError(
                    f"replacement_strategy must be an instance of ReplacementStrategy or None, got {type(replacement_strategy).__name__}."
                )
            if elitism > 0 or elite_ratio is not None:
                raise ValueError(
                    "Cannot specify both a custom replacement_strategy and elitism/elite_ratio. "
                    "Configure elitism directly inside GenerationalReplacement."
                )
            self.replacement_strategy: ReplacementStrategy = replacement_strategy
        else:
            self.replacement_strategy = GenerationalReplacement(
                num_elites=elitism,
                elite_ratio=elite_ratio,
            )

        self.selection_strategy: SelectionStrategy = selection_strategy
        self.crossover_strategy: Optional[RecombinationStrategy] = crossover_strategy
        self.mutation_strategy: Optional[MutationStrategy] = mutation_strategy
        self.crossover_prob: float = float(crossover_prob)
        self.mutation_prob: float = float(mutation_prob)
        self.evaluate_fn: Optional[Callable[[Individual], float]] = evaluate_fn

    @property
    def elitism(self) -> int:
        """Returns the number of elites if GenerationalReplacement is used, else 0."""
        if isinstance(self.replacement_strategy, GenerationalReplacement):
            return self.replacement_strategy.num_elites
        return 0

    def step(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring_count: Optional[int] = None,
        target_size: Optional[int] = None,
        parent_ids: Optional[Sequence[str]] = None,
    ) -> Population:
        """
        Advances the parent population by one generation.

        Args:
            parents: Parent Population or sequence of parent Individuals.
            offspring_count: Optional number of candidate offspring to generate during variation.
            target_size: Total survivor count forming the next generation. Defaults to len(parents).
            parent_ids: Optional list of ID strings corresponding to `parents` in order for lineage tracking.

        Returns:
            A new Population containing the surviving individuals.
        """
        parent_inds = _extract_individuals(parents)
        pop_size = len(parent_inds)

        if target_size is None:
            target_size = offspring_count if offspring_count is not None else pop_size
        _validate_k(target_size)

        if self.elitism > pop_size:
            raise ValueError(f"elitism ({self.elitism}) cannot exceed parent population size ({pop_size}).")

        # Determine how many candidate offspring need to be generated during variation
        if offspring_count is not None:
            _validate_k(offspring_count, allow_zero=True)
            needed_offspring = offspring_count
        elif isinstance(self.replacement_strategy, GenerationalReplacement):
            if self.replacement_strategy.elite_ratio is not None:
                elite_count = min(target_size, max(0, round(pop_size * self.replacement_strategy.elite_ratio)))
            else:
                elite_count = min(target_size, self.replacement_strategy.num_elites)
            needed_offspring = max(0, target_size - elite_count)
        else:
            needed_offspring = target_size

        if parent_ids is not None:
            if not isinstance(parent_ids, (list, tuple)):
                raise TypeError(f"parent_ids must be a sequence of strings, got {type(parent_ids).__name__}.")
            if len(parent_ids) != pop_size:
                raise ValueError(
                    f"Length of parent_ids ({len(parent_ids)}) does not match parent count ({pop_size})."
                )
            id_map: dict[int, str] = {id(ind): str(parent_ids[i]) for i, ind in enumerate(parent_inds)}
        else:
            id_map = {
                id(ind): str(ind.metadata.get("id", f"parent_{i}"))
                for i, ind in enumerate(parent_inds)
            }

        # Parent ID reverse map for quick identity checking
        parent_id_lookup: dict[int, Individual] = {id(ind): ind for ind in parent_inds}

        # Phase 1: Variation (Generate Offspring)
        offspring: list[Individual] = []
        while len(offspring) < needed_offspring:
            selected = self.selection_strategy.select(parent_inds, k=2)
            p1 = selected[0]
            p2 = selected[1] if len(selected) > 1 else selected[0]

            p1_id = id_map.get(id(p1), "unknown_parent_1")
            p2_id = id_map.get(id(p2), "unknown_parent_2")

            crossover_applied = False
            if self.crossover_strategy is not None and random.random() < self.crossover_prob:
                recomb_result = recombine(p1.genotype, p2.genotype, strategy=self.crossover_strategy)
                crossover_applied = True
                if isinstance(recomb_result, (tuple, list)):
                    child1_genome = recomb_result[0]
                    child2_genome = recomb_result[1] if len(recomb_result) > 1 else None
                else:
                    child1_genome = recomb_result
                    child2_genome = None
            else:
                child1_genome = copy.deepcopy(p1.genotype)
                child2_genome = copy.deepcopy(p2.genotype)

            # Mutation
            mutated1 = False
            if self.mutation_strategy is not None and random.random() < self.mutation_prob:
                child1_genome = mutate(child1_genome, strategy=self.mutation_strategy)
                mutated1 = True

            lineage1_parents = [p1_id, p2_id] if crossover_applied else [p1_id]
            child1 = Individual(
                genotype=child1_genome,
                lineage=Lineage(
                    parent_ids=lineage1_parents,
                    crossover_applied=crossover_applied,
                    mutated=mutated1,
                ),
            )
            if self.evaluate_fn is not None:
                child1.fitness = float(self.evaluate_fn(child1))
            offspring.append(child1)

            if child2_genome is not None and len(offspring) < needed_offspring:
                mutated2 = False
                if self.mutation_strategy is not None and random.random() < self.mutation_prob:
                    child2_genome = mutate(child2_genome, strategy=self.mutation_strategy)
                    mutated2 = True

                lineage2_parents = [p1_id, p2_id] if crossover_applied else [p2_id]
                child2 = Individual(
                    genotype=child2_genome,
                    lineage=Lineage(
                        parent_ids=lineage2_parents,
                        crossover_applied=crossover_applied,
                        mutated=mutated2,
                    ),
                )
                if self.evaluate_fn is not None:
                    child2.fitness = float(self.evaluate_fn(child2))
                offspring.append(child2)

        # Phase 2: Replacement (Survivor Selection)
        selected_survivors = self.replacement_strategy.replace(
            parents=parent_inds,
            offspring=offspring,
            target_size=target_size,
        )

        # Post-process: ensure surviving parents (elites) are cloned and have lineage marked
        final_survivors: list[Individual] = []
        for ind in selected_survivors:
            if id(ind) in parent_id_lookup:
                # This is a parent carried over as an elite
                elite_child = ind.clone(deep=True)
                parent_id = id_map.get(id(ind), "parent")
                elite_child.lineage = Lineage(
                    parent_ids=[parent_id],
                    crossover_applied=False,
                    mutated=False,
                )
                final_survivors.append(elite_child)
            else:
                final_survivors.append(ind)

        return Population(final_survivors)

    def __call__(
        self,
        parents: Union[Population, Sequence[Individual]],
        offspring_count: Optional[int] = None,
        target_size: Optional[int] = None,
        parent_ids: Optional[Sequence[str]] = None,
    ) -> Population:
        """Callable convenience alias for step()."""
        return self.step(
            parents=parents,
            offspring_count=offspring_count,
            target_size=target_size,
            parent_ids=parent_ids,
        )


def step(
    parents: Union[Population, Sequence[Individual]],
    selection_strategy: SelectionStrategy,
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
    Functional helper to advance an evolutionary population by one generation.

    Args:
        parents: Parent Population or sequence of parent Individuals.
        selection_strategy: SelectionStrategy used to select parent mating pairs.
        crossover_strategy: Optional RecombinationStrategy to combine parent pairs.
        mutation_strategy: Optional MutationStrategy to mutate child genotypes.
        replacement_strategy: Optional ReplacementStrategy for parent/offspring replacement into the next generation.
        offspring_count: Number of offspring to produce during variation.
        target_size: Total survivor count forming next generation. Defaults to len(parents).
        crossover_prob: Probability of applying crossover (0.0 to 1.0).
        mutation_prob: Probability of applying mutation (0.0 to 1.0).
        elitism: Number of top-fitness parents to carry over unchanged (used when replacement_strategy is None).
        elite_ratio: Optional fraction (0.0 to 1.0) of parent population to carry over.
        evaluate_fn: Optional fitness evaluation callable (Individual -> float).
        parent_ids: Optional list of ID strings for lineage tracking.

    Returns:
        A new Population of surviving individuals for the next generation.
    """
    pipeline = GenerationPipeline(
        selection_strategy=selection_strategy,
        crossover_strategy=crossover_strategy,
        mutation_strategy=mutation_strategy,
        replacement_strategy=replacement_strategy,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        elitism=elitism,
        elite_ratio=elite_ratio,
        evaluate_fn=evaluate_fn,
    )
    return pipeline.step(
        parents=parents,
        offspring_count=offspring_count,
        target_size=target_size,
        parent_ids=parent_ids,
    )
