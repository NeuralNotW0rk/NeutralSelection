from __future__ import annotations

import copy
from typing import Callable, Any, Optional
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.representation.lineage import Lineage


def _clone_genome(genome: Genome, deep: bool = True) -> Genome:
    """Clones a genome instance, preserving its subtype and segment key if applicable."""
    if deep:
        return copy.deepcopy(genome)
    if isinstance(genome, Segment):
        return genome.__class__(genome.key, list(genome._items))
    return genome.__class__(list(genome._items))


class Individual:
    """Represents an individual in the population, wrapping a genotype and managing its expression into a phenotype."""

    def __init__(
        self,
        genotype: Genome,
        fitness: Optional[float] = None,
        lineage: Optional[Lineage] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not isinstance(genotype, Genome):
            raise TypeError(f"genotype must be an instance of Genome, got {type(genotype).__name__}")
        if fitness is not None:
            if not isinstance(fitness, (int, float)) or isinstance(fitness, bool):
                raise TypeError(f"fitness must be a float, int, or None, got {type(fitness).__name__}")
            fitness = float(fitness)
        if lineage is not None and not isinstance(lineage, Lineage):
            raise TypeError(f"lineage must be an instance of Lineage or None, got {type(lineage).__name__}")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict or None, got {type(metadata).__name__}")

        self.genotype: Genome = genotype
        self._phenotype: Any = None
        self.fitness: Optional[float] = fitness
        self.lineage: Optional[Lineage] = lineage
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}

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

    def clone(self, deep: bool = True) -> Individual:
        """
        Creates a copy of the Individual.
        Genotype and metadata are cloned; phenotype cache is cleared unless deep=False.
        """
        cloned_genotype = _clone_genome(self.genotype, deep=deep)
        cloned_lineage = copy.deepcopy(self.lineage) if (deep and self.lineage is not None) else self.lineage
        cloned_metadata = copy.deepcopy(self.metadata) if deep else dict(self.metadata)
        cloned_ind = Individual(
            genotype=cloned_genotype,
            fitness=self.fitness,
            lineage=cloned_lineage,
            metadata=cloned_metadata,
        )
        return cloned_ind
