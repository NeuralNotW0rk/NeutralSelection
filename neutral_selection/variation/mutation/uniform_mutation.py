import warnings
from typing import Callable, Any
from .function_mutation import FunctionMutation
from neutral_selection.registry import register_mutation


@register_mutation(["uniform", "elementwise_mutation"])
class UniformMutation(FunctionMutation):
    """
    Deprecated: use FunctionMutation(mutation_fn, mutation_rate, granularity="gene").

    Applies a callable to each gene with probability `mutation_rate`, where a gene is any item that is
    not itself a Genome or Segment.
    """

    def __init__(self, mutation_rate: float, mutation_fn: Callable[[Any], Any]) -> None:
        warnings.warn(
            "UniformMutation is deprecated; use FunctionMutation(mutation_fn, mutation_rate, granularity=\"gene\").",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(mutation_fn, mutation_rate, granularity="gene")
