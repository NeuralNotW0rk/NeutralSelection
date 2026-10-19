import warnings
from typing import Callable, Any, Optional
from .function_crossover import FunctionCrossover
from neutral_selection.registry import register_crossover


@register_crossover("elementwise")
class ElementwiseCrossover(FunctionCrossover):
    """
    Deprecated: use FunctionCrossover(crossover_fn, blend_factor, granularity="gene", num_children=1).

    Blends each aligned pair of genes, where a gene is any item that is not itself a Genome or Segment.
    """

    def __init__(
        self,
        blend_factor: float = 0.5,
        crossover_fn: Optional[Callable[[Any, Any, float], Any]] = None,
        *,
        blend_fn: Optional[Callable[[Any, Any, float], Any]] = None,
    ) -> None:
        warnings.warn(
            "ElementwiseCrossover is deprecated; use FunctionCrossover(crossover_fn, blend_factor, granularity=\"gene\", num_children=1).",
            DeprecationWarning,
            stacklevel=2,
        )
        fn = crossover_fn if crossover_fn is not None else blend_fn
        super().__init__(fn, blend_factor, granularity="gene", num_children=1)
