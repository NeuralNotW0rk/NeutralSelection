from typing import TypeVar, Any, Generic
from neutral_selection.representation.genome import MutationFn

T = TypeVar("T")


class UniformMutation(Generic[T]):
    """A structure-agnostic mutation strategy that maps mutation_fn across elements using map."""

    def __init__(self, mutation_rate: float, mutation_fn: MutationFn[T]) -> None:
        if mutation_fn is None:
            raise ValueError("mutation_fn must be provided and cannot be None.")
        if not callable(mutation_fn):
            raise TypeError("mutation_fn must be a callable.")
        if not isinstance(mutation_rate, (int, float)):
            raise TypeError("mutation_rate must be a float or int.")
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be between 0.0 and 1.0 inclusive.")
        self.mutation_rate = float(mutation_rate)
        self.mutation_fn = mutation_fn

    def __call__(self, genome: Any) -> Any:
        if not hasattr(genome, "map"):
            raise TypeError("genome does not support mutation mapping (missing map method).")

        import random

        def mutate_element(item: T) -> T:
            return self.mutation_fn(item) if random.random() < self.mutation_rate else item

        return genome.map(mutate_element)
