from __future__ import annotations

import random
import dataclasses
import sys
from typing import Optional, Sequence, Tuple, Union, Any
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.representation.hierarchy import (
    align_strands,
    concat_strands,
    flatten_hierarchy,
    flatten_strand,
    is_tensor_strand,
    unflatten_hierarchy,
    TreeDef,
)
from .base import RecombinationStrategy
from neutral_selection.registry import register_crossover


def _validate_composite_parent(parent: Any) -> None:
    if isinstance(parent, (str, bytes, bytearray, int, float, bool)) or not (
        isinstance(parent, (Genome, list, tuple))
        or dataclasses.is_dataclass(parent)
        or hasattr(parent, "shape")
        or hasattr(parent, "__hierarchical_flatten__")
    ):
        raise TypeError(f"Parent must be a Genome, sequence, dataclass or tensor, got {type(parent).__name__}")


def clone_genome_structure(original: Genome, new_items: list) -> Genome:
    """Creates a new genome instance matching the type and metadata of the original."""
    if isinstance(original, Segment):
        return original.__class__(original.key, new_items)
    return original.__class__(new_items)


def n_point_crossover(
    parent_a: Any,
    parent_b: Any,
    cut_points: list[int],
    max_depth: Optional[int] = None,
    atomic_types: tuple[type, ...] = (),
) -> tuple[Any, Any]:
    """
    Polymorphically slices and swaps parent genomes across a flattened multi-tier hierarchy at the specified cut points.
    Acts as the atomic helper operation for structural sequence crossovers.
    """
    _validate_composite_parent(parent_a)
    _validate_composite_parent(parent_b)
    return _splice_at_cuts(*_flatten_parents(parent_a, parent_b, max_depth, atomic_types), cut_points)


def _flatten_parents(
    parent_a: Any,
    parent_b: Any,
    max_depth: Optional[int],
    atomic_types: tuple[type, ...],
) -> tuple[Any, TreeDef, Any, TreeDef]:
    """Flattens both parents into leaf sequences of a common form (tensor strands or lists)."""
    leaves_a, treedef_a = flatten_strand(parent_a, max_depth=max_depth, atomic_types=atomic_types)
    leaves_b, treedef_b = flatten_strand(parent_b, max_depth=max_depth, atomic_types=atomic_types)
    leaves_a, leaves_b = align_strands(leaves_a, leaves_b)
    return leaves_a, treedef_a, leaves_b, treedef_b


def _splice_at_cuts(
    leaves_a: Any,
    treedef_a: TreeDef,
    leaves_b: Any,
    treedef_b: TreeDef,
    cut_points: Sequence[int],
) -> tuple[Any, Any]:
    """Swaps flattened parent segments at the given cut points and rebuilds both children."""
    if len(leaves_a) != len(leaves_b) or treedef_a.total_leaves != treedef_b.total_leaves:
        raise ValueError(
            f"Genomes must have matching leaf structures for crossover: "
            f"{len(leaves_a)} vs {len(leaves_b)}"
        )
    if any(not isinstance(c, int) or c < 0 for c in cut_points):
        raise ValueError("Cut points must be non-negative integers")

    sorted_cuts = sorted(list(set(cut_points)))

    child1_parts = []
    child2_parts = []

    last_cut = 0
    use_a = True

    # The trailing None takes each parent's tail after the final cut
    for cut in sorted_cuts + [None]:
        if use_a:
            child1_parts.append(leaves_a[last_cut:cut])
            child2_parts.append(leaves_b[last_cut:cut])
        else:
            child1_parts.append(leaves_b[last_cut:cut])
            child2_parts.append(leaves_a[last_cut:cut])
        last_cut = cut
        use_a = not use_a

    return (
        unflatten_hierarchy(concat_strands(child1_parts), treedef_a),
        unflatten_hierarchy(concat_strands(child2_parts), treedef_b),
    )


@register_crossover("one_point")
class OnePointCrossover(RecombinationStrategy):
    """
    One-Point Crossover strategy (Holland, 1975).

    Slices parents at a single cut point and exchanges tails to create two offspring.
    Supports multi-scale hierarchical structures natively via flattening/unflattening.
    """

    def __init__(
        self,
        cut_point: Optional[int] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if cut_point is not None:
            if type(cut_point) is not int or cut_point < 0:
                raise ValueError(f"cut_point must be a non-negative integer, got {cut_point}.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.cut_point = cut_point
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        flat = _flatten_parents(parent_a, parent_b, self.max_depth, self.atomic_types)
        n = len(flat[0])
        if n <= 1:
            return parent_a, parent_b

        cut = self.cut_point if self.cut_point is not None else random.randint(1, n - 1)
        return _splice_at_cuts(*flat, [cut])


@register_crossover("two_point")
class TwoPointCrossover(RecombinationStrategy):
    """
    Two-Point Crossover strategy (De Jong, 1975).

    Slices parents at two cut points and exchanges the middle segment.
    Supports multi-scale hierarchical structures natively via flattening/unflattening.
    """

    def __init__(
        self,
        cut_points: Optional[Tuple[int, int]] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if cut_points is not None:
            if (
                not isinstance(cut_points, tuple)
                or len(cut_points) != 2
                or any(type(c) is not int or c < 0 for c in cut_points)
            ):
                raise ValueError("cut_points must be a tuple of two non-negative integers.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.cut_points = cut_points
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        flat = _flatten_parents(parent_a, parent_b, self.max_depth, self.atomic_types)
        n = len(flat[0])
        if n <= 2:
            return parent_a, parent_b

        if self.cut_points is not None:
            cuts = list(self.cut_points)
        else:
            cuts = sorted(random.sample(range(1, n), 2))

        return _splice_at_cuts(*flat, cuts)


@register_crossover("fixed_n_point")
class NPointCrossover(RecombinationStrategy):
    """A structural crossover strategy that applies N-point sequence crossover at deterministic cut points."""

    def __init__(
        self,
        cut_points: list[int],
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(cut_points, list):
            raise TypeError("cut_points must be a list of integers")
        if any(not isinstance(c, int) or c < 0 for c in cut_points):
            raise ValueError("Cut points must be non-negative integers")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.cut_points = cut_points
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        return n_point_crossover(
            parent_a, parent_b, self.cut_points, max_depth=self.max_depth, atomic_types=self.atomic_types
        )


@register_crossover(["random_n_point", "n_point"])
class RandomNPointCrossover(RecombinationStrategy):
    """Applies structural N-point crossover to parent genomes with randomly selected cut points."""

    def __init__(
        self,
        num_cut_points: int = 1,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(num_cut_points, int) or num_cut_points < 1:
            raise ValueError("num_cut_points must be an integer >= 1")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.num_cut_points = num_cut_points
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        flat = _flatten_parents(parent_a, parent_b, self.max_depth, self.atomic_types)
        n = len(flat[0])
        if n <= 1:
            return parent_a, parent_b

        num_cuts = min(self.num_cut_points, n - 1)
        cut_points = sorted(random.sample(range(1, n), num_cuts))

        return _splice_at_cuts(*flat, cut_points)


@register_crossover("uniform")
class UniformCrossover(RecombinationStrategy):
    """
    Uniform Crossover strategy (Syswerda, 1989).

    Each element in child 1 is inherited from parent A with probability `swap_prob` and from parent B
    otherwise (and vice-versa for child 2).
    """

    def __init__(
        self,
        swap_prob: float = 0.5,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if not isinstance(swap_prob, (int, float)) or isinstance(swap_prob, bool):
            raise TypeError(f"swap_prob must be a float, got {type(swap_prob).__name__}.")
        if not (0.0 <= swap_prob <= 1.0):
            raise ValueError(f"swap_prob must be in [0.0, 1.0], got {swap_prob}.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.swap_prob = float(swap_prob)
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        leaves_a, treedef_a = flatten_strand(parent_a, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_b, treedef_b = flatten_strand(parent_b, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_a, leaves_b = align_strands(leaves_a, leaves_b)

        if len(leaves_a) != len(leaves_b) or treedef_a.total_leaves != treedef_b.total_leaves:
            raise ValueError("Genomes must have matching leaf structures for uniform crossover")

        if is_tensor_strand(leaves_a):
            # Draws one swap decision per element from torch's RNG instead of Python's random
            torch = sys.modules["torch"]
            from_a = torch.rand(len(leaves_a), device=leaves_a.device) < self.swap_prob
            return (
                unflatten_hierarchy(torch.where(from_a, leaves_a, leaves_b), treedef_a),
                unflatten_hierarchy(torch.where(from_a, leaves_b, leaves_a), treedef_b),
            )

        child1_items = []
        child2_items = []

        for a, b in zip(leaves_a, leaves_b):
            if random.random() < self.swap_prob:
                child1_items.append(a)
                child2_items.append(b)
            else:
                child1_items.append(b)
                child2_items.append(a)

        return (
            unflatten_hierarchy(child1_items, treedef_a),
            unflatten_hierarchy(child2_items, treedef_b),
        )


@register_crossover("shuffle")
class ShuffleCrossover(RecombinationStrategy):
    """
    Shuffle Crossover strategy (Eshelman, Caruana & Schaffer, 1989).

    Shuffles gene positions identically in both parents, applies an inner crossover strategy
    (defaults to OnePointCrossover), and un-shuffles back to the original index positions.
    """

    def __init__(
        self,
        crossover_strategy: Optional[RecombinationStrategy] = None,
        max_depth: Optional[int] = None,
        atomic_types: tuple[type, ...] = (),
    ) -> None:
        if crossover_strategy is not None and not isinstance(crossover_strategy, RecombinationStrategy):
            raise TypeError("crossover_strategy must be an instance of RecombinationStrategy.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")
        self.crossover_strategy = crossover_strategy or OnePointCrossover()
        self.max_depth = max_depth
        self.atomic_types = atomic_types

    def __call__(self, parent_a: Any, parent_b: Any) -> tuple[Any, Any]:
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        leaves_a, treedef_a = flatten_hierarchy(parent_a, max_depth=self.max_depth, atomic_types=self.atomic_types)
        leaves_b, treedef_b = flatten_hierarchy(parent_b, max_depth=self.max_depth, atomic_types=self.atomic_types)

        if len(leaves_a) != len(leaves_b) or treedef_a.total_leaves != treedef_b.total_leaves:
            raise ValueError("Genomes must have matching leaf structures for shuffle crossover")

        n = len(leaves_a)
        if n <= 1:
            return parent_a, parent_b

        indices = list(range(n))
        random.shuffle(indices)

        shuffled_a = Genome([leaves_a[i] for i in indices])
        shuffled_b = Genome([leaves_b[i] for i in indices])

        res_a, res_b = self.crossover_strategy(shuffled_a, shuffled_b)

        child1_items = [None] * n
        child2_items = [None] * n
        for orig_idx, shuff_val_a, shuff_val_b in zip(indices, res_a, res_b):
            child1_items[orig_idx] = shuff_val_a
            child2_items[orig_idx] = shuff_val_b

        return (
            unflatten_hierarchy(child1_items, treedef_a),
            unflatten_hierarchy(child2_items, treedef_b),
        )
