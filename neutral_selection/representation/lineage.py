from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class Lineage:
    """
    Tracks the evolutionary provenance and reproduction history of an Individual.

    Attributes:
        parent_ids: Identifiers or keys of parent individuals.
        crossover_applied: Whether recombination/crossover was applied to produce this individual.
        mutated: Whether mutation was applied to produce this individual.
        metadata: Additional arbitrary domain-specific lineage attributes.
    """
    parent_ids: list[str] = field(default_factory=list)
    crossover_applied: bool = False
    mutated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.parent_ids, list):
            if isinstance(self.parent_ids, (tuple, set)):
                self.parent_ids = [str(pid) for pid in self.parent_ids]
            else:
                raise TypeError(f"parent_ids must be a list of strings, got {type(self.parent_ids).__name__}")
        else:
            self.parent_ids = [str(pid) for pid in self.parent_ids]

        if not isinstance(self.crossover_applied, bool):
            raise TypeError(f"crossover_applied must be a bool, got {type(self.crossover_applied).__name__}")

        if not isinstance(self.mutated, bool):
            raise TypeError(f"mutated must be a bool, got {type(self.mutated).__name__}")

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the lineage record into a dictionary."""
        return {
            "parent_ids": list(self.parent_ids),
            "crossover_applied": self.crossover_applied,
            "mutated": self.mutated,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Lineage:
        """Constructs a Lineage instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"data must be a dict, got {type(data).__name__}")
        return cls(
            parent_ids=list(data.get("parent_ids", [])),
            crossover_applied=bool(data.get("crossover_applied", False)),
            mutated=bool(data.get("mutated", False)),
            metadata=dict(data.get("metadata", {})),
        )
