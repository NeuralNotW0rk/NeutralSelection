from __future__ import annotations

from typing import Any, Callable, Sequence, Union
from neutral_selection.representation.hierarchy import (
    Granularity,
    NodeDef,
    TreeDef,
    resolve_granularity,
    flatten_hierarchy,
    unflatten_hierarchy,
)
from .base import RecombinationStrategy
from .n_point_crossover import _validate_composite_parent
from neutral_selection.registry import register_crossover


def _same_structure(a: NodeDef, b: NodeDef) -> bool:
    """Checks that two flattened parents share a tree shape (leaf values and leaf types may differ)."""
    if a.metadata.get("is_leaf") and b.metadata.get("is_leaf"):
        return True
    return (
        a.node_type is b.node_type
        and a.num_leaves == b.num_leaves
        and a.metadata.get("key") == b.metadata.get("key")
        and len(a.children_defs) == len(b.children_defs)
        and all(_same_structure(x, y) for x, y in zip(a.children_defs, b.children_defs))
    )


@register_crossover(["function", "callable"])
class FunctionCrossover(RecombinationStrategy):
    """
    Combines each aligned pair of parent leaves with a custom callable `crossover_fn(leaf_a, leaf_b, blend_factor)`.

    The callable receives leaves at the operator's granularity: scalars at "atom" (the default), whole
    tensors at "tensor", or whole gene objects at "gene". With num_children=2 (the default) returns two
    children, built from crossover_fn(a, b) and crossover_fn(b, a) at every leaf; with num_children=1
    returns only the first. Use 1 for a symmetric callable, such as a 50/50 blend, whose two children
    would otherwise be identical.
    Supports multi-tier hierarchical structures natively.
    """

    def __init__(
        self,
        crossover_fn: Callable[[Any, Any, float], Any],
        blend_factor: float = 0.5,
        atomic_types: tuple[type, ...] = (),
        granularity: Granularity = None,
        num_children: int = 2,
    ) -> None:
        if crossover_fn is None:
            raise ValueError("crossover_fn must be provided and cannot be None.")
        if not callable(crossover_fn):
            raise TypeError("crossover_fn must be a callable.")
        if not isinstance(blend_factor, (int, float)) or isinstance(blend_factor, bool):
            raise TypeError("blend_factor must be a float or int.")
        if num_children not in (1, 2):
            raise ValueError(f"num_children must be 1 or 2, got {num_children!r}.")
        self.crossover_fn = crossover_fn
        self.blend_factor = float(blend_factor)
        self.atomic_types = atomic_types
        self.granularity = resolve_granularity(granularity)
        self.num_children = num_children

    def __call__(self, parent_a: Any, parent_b: Any) -> Union[Any, tuple[Any, Any]]:
        leaves_a, treedef_a, leaves_b, treedef_b = self._flatten_parents(parent_a, parent_b)
        if self.num_children == 1:
            return self._combine(leaves_a, leaves_b, treedef_a)
        return (
            self._combine(leaves_a, leaves_b, treedef_a),
            self._combine(leaves_b, leaves_a, treedef_b),
        )

    def _flatten_parents(self, parent_a: Any, parent_b: Any) -> tuple[list[Any], TreeDef, list[Any], TreeDef]:
        """Flattens both parents to Python leaves and checks they align leaf for leaf."""
        _validate_composite_parent(parent_a)
        _validate_composite_parent(parent_b)
        leaves_a, treedef_a = flatten_hierarchy(parent_a, atomic_types=self.atomic_types, granularity=self.granularity)
        leaves_b, treedef_b = flatten_hierarchy(parent_b, atomic_types=self.atomic_types, granularity=self.granularity)
        if not _same_structure(treedef_a.root_def, treedef_b.root_def):
            raise ValueError("Parents must share the same structure (lengths and segment keys) for crossover.")
        return leaves_a, treedef_a, leaves_b, treedef_b

    def _combine(self, leaves_x: Sequence[Any], leaves_y: Sequence[Any], treedef: TreeDef) -> Any:
        """Builds one child by applying crossover_fn to each aligned leaf pair, in that argument order."""
        combined = [self.crossover_fn(x, y, self.blend_factor) for x, y in zip(leaves_x, leaves_y)]
        return unflatten_hierarchy(combined, treedef)
