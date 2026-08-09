from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.variation.recombination.base import RecombinationStrategy


class SegmentSwapCrossover(RecombinationStrategy):
    """
    A structural crossover strategy that swaps entire segments wholesale based on their keys.
    Expects a genome containing only Segment sub-genomes.
    """

    def __init__(self, active_keys: set) -> None:
        if not isinstance(active_keys, set):
            raise TypeError("active_keys must be a set")
        self.active_keys = active_keys

    def __call__(self, parent_a: Genome, parent_b: Genome) -> Genome:
        if len(parent_a) != len(parent_b):
            raise ValueError("Genomes must have the same length for crossover.")

        new_items = []
        for i in range(len(parent_a)):
            item_a = parent_a[i]
            item_b = parent_b[i]

            if isinstance(item_a, Segment) and isinstance(item_b, Segment):
                if item_a.key != item_b.key:
                    raise ValueError(f"Segment keys mismatch: '{item_a.key}' vs '{item_b.key}'")
                
                if item_a.key in self.active_keys:
                    new_items.append(item_b)
                else:
                    new_items.append(item_a)
            else:
                raise TypeError("SegmentSwapCrossover expects a genome containing only Segment items.")

        if isinstance(parent_a, Segment):
            return parent_a.__class__(parent_a.key, new_items)
        return parent_a.__class__(new_items)
