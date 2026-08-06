from typing import TypeVar, Any, Generic
from neutral_selection.representation.genome import CrossoverFn

T = TypeVar("T")
K = TypeVar("K")


class ElementwiseCrossover(Generic[T]):
    """A structure-agnostic crossover strategy that blends elements of parent genomes using zip_map."""

    def __init__(self, blend_factor: float, crossover_fn: CrossoverFn[T]) -> None:
        if crossover_fn is None:
            raise ValueError("crossover_fn must be provided and cannot be None.")
        if not callable(crossover_fn):
            raise TypeError("crossover_fn must be a callable.")
        if not isinstance(blend_factor, (int, float)):
            raise TypeError("blend_factor must be a float or int.")
        self.blend_factor = float(blend_factor)
        self.crossover_fn = crossover_fn

    def __call__(self, parent_a: Any, parent_b: Any) -> Any:
        if not hasattr(parent_a, "zip_map"):
            raise TypeError("parent_a does not support element-wise crossover (missing zip_map).")
        return parent_a.zip_map(parent_b, self.crossover_fn, self.blend_factor)


class NPointCrossover:
    """A structural crossover strategy that applies N-point sequence crossover to genomes."""

    def __init__(self, cut_points: list[int]) -> None:
        if not isinstance(cut_points, list):
            raise TypeError("cut_points must be a list")
        if any(not isinstance(c, int) for c in cut_points):
            raise TypeError("All cut points must be integers")
        if any(c < 0 for c in cut_points):
            raise ValueError("Cut points must be non-negative integers")
        self.cut_points = cut_points

    def __call__(self, parent_a: Any, parent_b: Any) -> Any:
        if not hasattr(parent_a, "crossover"):
            raise TypeError("parent_a does not support structural N-point crossover.")
        return parent_a.crossover(parent_b, self.cut_points)


class SegmentSwapCrossover(Generic[K]):
    """A structural crossover strategy that swaps entire segments of SegmentedGenomes wholesale."""

    def __init__(self, active_keys: set[K]) -> None:
        if not isinstance(active_keys, set):
            raise TypeError("active_keys must be a set")
        self.active_keys = active_keys

    def __call__(self, parent_a: Any, parent_b: Any) -> Any:
        if not hasattr(parent_a, "crossover_segments"):
            raise TypeError("parent_a does not support segment-swap crossover.")
        return parent_a.crossover_segments(parent_b, self.active_keys)
