from __future__ import annotations

import random
from typing import Sequence, Tuple
from neutral_selection.representation.genome import Genome, Segment
from .base import RecombinationStrategy


def clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


def _validate_permutation_parents(parent_a: Genome, parent_b: Genome) -> int:
    """Validates that parents are Genome instances of equal length."""
    if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
        raise TypeError("parent_a and parent_b must be Genome instances.")
    if len(parent_a) != len(parent_b):
        raise ValueError("Genomes must have the same length for crossover.")
    return len(parent_a)


class OrderCrossover(RecombinationStrategy):
    """
    Order Crossover strategy (OX1 - Davis, 1985).

    Preserves the relative order of elements from parents. Standard for permutation and ordering
    problems (such as TSP and scheduling).
    """

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        n = _validate_permutation_parents(parent_a, parent_b)
        if n <= 1:
            return parent_a, parent_b

        p1, p2 = sorted(random.sample(range(n + 1), 2))
        if p1 == p2:
            p2 = min(n, p1 + 1)

        def _make_ox_child(p_primary: Sequence, p_secondary: Sequence) -> list:
            child = [None] * n
            child[p1:p2] = p_primary[p1:p2]
            primary_set = set(child[p1:p2])

            # Fill remainder starting from p2 in circular fashion
            fill_items = [item for item in (list(p_secondary[p2:]) + list(p_secondary[:p2])) if item not in primary_set]
            fill_idx = 0
            for i in range(p2, n):
                child[i] = fill_items[fill_idx]
                fill_idx += 1
            for i in range(0, p1):
                child[i] = fill_items[fill_idx]
                fill_idx += 1
            return child

        child1_items = _make_ox_child(parent_a, parent_b)
        child2_items = _make_ox_child(parent_b, parent_a)

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )


class PartiallyMatchedCrossover(RecombinationStrategy):
    """
    Partially Matched Crossover strategy (PMX - Goldberg & Lingle, 1985).

    Preserves absolute gene positions from parents while establishing a positional mapping
    to resolve duplicate gene conflicts.
    """

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        n = _validate_permutation_parents(parent_a, parent_b)
        if n <= 1:
            return parent_a, parent_b

        p1, p2 = sorted(random.sample(range(n + 1), 2))
        if p1 == p2:
            p2 = min(n, p1 + 1)

        def _make_pmx_child(p_primary: Sequence, p_secondary: Sequence) -> list:
            child = [None] * n
            child[p1:p2] = p_primary[p1:p2]

            # Build mapping from segment
            mapping = {}
            for i in range(p1, p2):
                mapping[p_primary[i]] = p_secondary[i]

            # Fill outside segment
            for i in list(range(0, p1)) + list(range(p2, n)):
                val = p_secondary[i]
                while val in mapping:
                    val = mapping[val]
                child[i] = val

            return child

        child1_items = _make_pmx_child(parent_a, parent_b)
        child2_items = _make_pmx_child(parent_b, parent_a)

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )


class CycleCrossover(RecombinationStrategy):
    """
    Cycle Crossover strategy (CX - Oliver, Smith & Holland, 1987).

    Discovers disjoint cycles between parents and preserves exact positional information from parents.
    """

    def __call__(self, parent_a: Genome, parent_b: Genome) -> tuple[Genome, Genome]:
        n = _validate_permutation_parents(parent_a, parent_b)
        if n <= 1:
            return parent_a, parent_b

        cycles: list[list[int]] = []
        visited = set()

        for start_idx in range(n):
            if start_idx in visited:
                continue

            cycle: list[int] = []
            curr_idx = start_idx
            while curr_idx not in visited:
                visited.add(curr_idx)
                cycle.append(curr_idx)
                val_b = parent_b[curr_idx]
                try:
                    curr_idx = [i for i, x in enumerate(parent_a) if x == val_b][0]
                except IndexError:
                    break

            if cycle:
                cycles.append(cycle)

        child1_items = [None] * n
        child2_items = [None] * n

        for cycle_idx, cycle in enumerate(cycles):
            for idx in cycle:
                if cycle_idx % 2 == 0:
                    child1_items[idx] = parent_a[idx]
                    child2_items[idx] = parent_b[idx]
                else:
                    child1_items[idx] = parent_b[idx]
                    child2_items[idx] = parent_a[idx]

        return (
            clone_genome_structure(parent_a, child1_items),
            clone_genome_structure(parent_b, child2_items),
        )
