from typing import Callable, Any, Optional
from neutral_selection.representation.genome import Genome, Segment
from .base import RecombinationStrategy
from neutral_selection.registry import register_crossover


@register_crossover("elementwise")
class ElementwiseCrossover(RecombinationStrategy):
    """
    A structure-agnostic crossover strategy that blends elements of parent genomes recursively.
    Supports flat genomes, nested genomes, and segment hierarchies.
    """

    def __init__(
        self,
        blend_factor: float = 0.5,
        crossover_fn: Optional[Callable[[Any, Any, float], Any]] = None,
        *,
        blend_fn: Optional[Callable[[Any, Any, float], Any]] = None,
    ) -> None:
        fn = crossover_fn if crossover_fn is not None else blend_fn
        if fn is None:
            raise ValueError("crossover_fn must be provided and cannot be None.")
        if not callable(fn):
            raise TypeError("crossover_fn must be a callable.")
        if not isinstance(blend_factor, (int, float)):
            raise TypeError("blend_factor must be a float or int.")
        self.blend_factor = float(blend_factor)
        self.crossover_fn = fn

    def __call__(self, parent_a: Genome, parent_b: Genome) -> Genome:
        if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
            raise TypeError("parent_a and parent_b must be Genome instances.")
        if len(parent_a) != len(parent_b):
            raise ValueError("Genomes must have the same length for crossover.")

        blended_items = []
        for i in range(len(parent_a)):
            item_a = parent_a[i]
            item_b = parent_b[i]

            # Recursively blend if both items are genomes/sub-genomes
            if isinstance(item_a, Genome) and isinstance(item_b, Genome):
                if isinstance(item_a, Segment) and isinstance(item_b, Segment):
                    if item_a.key != item_b.key:
                        raise ValueError(f"Segment keys mismatch: '{item_a.key}' vs '{item_b.key}'")
                
                blended_item = self(item_a, item_b)
            else:
                blended_item = self.crossover_fn(item_a, item_b, self.blend_factor)
            blended_items.append(blended_item)

        if isinstance(parent_a, Segment):
            return parent_a.__class__(parent_a.key, blended_items)
        return parent_a.__class__(blended_items)
