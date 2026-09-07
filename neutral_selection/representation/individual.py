from __future__ import annotations
from typing import Callable, Any, Optional
from neutral_selection.representation.genome import Genome


class Individual:
    """Represents an individual in the population, wrapping a genotype and managing its expression into a phenotype."""

    def __init__(self, genotype: Genome) -> None:
        self.genotype: Genome = genotype
        self._phenotype: Any = None
        self.fitness: float | None = None

    @property
    def is_expressed(self) -> bool:
        """Returns True if the phenotype has been expressed and cached."""
        return self._phenotype is not None

    @property
    def phenotype(self) -> Any:
        """
        Returns the cached phenotype expression.
        Raises a ValueError if express() has not been called yet.
        """
        if self._phenotype is None:
            raise ValueError("Phenotype has not been expressed yet. Call express() first.")
        return self._phenotype

    def express(self, decode_fn: Callable[[Genome], Any]) -> Any:
        """
        Translates the genotype into the domain-specific phenotype using decode_fn.
        Caches the resulting phenotype internally.
        """
        if decode_fn is None:
            raise ValueError("decode_fn must be provided and cannot be None.")
        if not callable(decode_fn):
            raise TypeError("decode_fn must be a callable.")

        if self._phenotype is None:
            self._phenotype = decode_fn(self.genotype)
        return self._phenotype

    def clear_expression(self) -> None:
        """Clears the cached phenotype expression."""
        self._phenotype = None
