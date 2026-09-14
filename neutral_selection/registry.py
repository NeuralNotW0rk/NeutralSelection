from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Sequence, Type, TypeVar, Union, Optional

if TYPE_CHECKING:
    from neutral_selection.variation.selection.base import SelectionStrategy
    from neutral_selection.variation.recombination.base import RecombinationStrategy
    from neutral_selection.variation.mutation.base import MutationStrategy
    from neutral_selection.replacement.base import ReplacementStrategy

T = TypeVar("T")

SELECTION_REGISTRY: dict[str, Type[Any]] = {}
CROSSOVER_REGISTRY: dict[str, Type[Any]] = {}
MUTATION_REGISTRY: dict[str, Type[Any]] = {}
REPLACEMENT_REGISTRY: dict[str, Type[Any]] = {}


def _normalize_names(name_or_names: Union[str, Sequence[str]]) -> list[str]:
    """Helper to convert a single string or sequence of strings into lowercase identifier strings."""
    if isinstance(name_or_names, str):
        return [name_or_names.lower().strip()]
    elif isinstance(name_or_names, (list, tuple, set)):
        return [str(n).lower().strip() for n in name_or_names]
    raise TypeError(f"Registry identifier must be a str or sequence of str, got {type(name_or_names).__name__}")


def register_selection(name: Union[str, Sequence[str]]) -> Callable[[Type[T]], Type[T]]:
    """A decorator to register a SelectionStrategy subclass with one or more string identifiers."""
    def decorator(cls: Type[T]) -> Type[T]:
        from neutral_selection.variation.selection.base import SelectionStrategy
        if not issubclass(cls, SelectionStrategy):
            raise TypeError(f"Class '{cls.__name__}' must inherit from SelectionStrategy.")
        for identifier in _normalize_names(name):
            if identifier in SELECTION_REGISTRY:
                raise ValueError(f"Selection strategy '{identifier}' is already registered to {SELECTION_REGISTRY[identifier].__name__}.")
            SELECTION_REGISTRY[identifier] = cls
        return cls
    return decorator


def get_selection_strategy(name: str) -> Type[SelectionStrategy]:
    """Retrieves the registered SelectionStrategy subclass for the given identifier."""
    key = str(name).lower().strip()
    if key not in SELECTION_REGISTRY:
        raise KeyError(f"No SelectionStrategy registered for '{name}'. Available: {list(SELECTION_REGISTRY.keys())}")
    return SELECTION_REGISTRY[key]


def list_selection_strategies() -> list[str]:
    """Returns a sorted list of registered selection strategy identifiers."""
    return sorted(list(SELECTION_REGISTRY.keys()))


def register_crossover(name: Union[str, Sequence[str]]) -> Callable[[Type[T]], Type[T]]:
    """A decorator to register a RecombinationStrategy subclass with one or more string identifiers."""
    def decorator(cls: Type[T]) -> Type[T]:
        from neutral_selection.variation.recombination.base import RecombinationStrategy
        if not issubclass(cls, RecombinationStrategy):
            raise TypeError(f"Class '{cls.__name__}' must inherit from RecombinationStrategy.")
        for identifier in _normalize_names(name):
            if identifier in CROSSOVER_REGISTRY:
                raise ValueError(f"Crossover strategy '{identifier}' is already registered to {CROSSOVER_REGISTRY[identifier].__name__}.")
            CROSSOVER_REGISTRY[identifier] = cls
        return cls
    return decorator


def get_crossover_strategy(name: str) -> Type[RecombinationStrategy]:
    """Retrieves the registered RecombinationStrategy subclass for the given identifier."""
    key = str(name).lower().strip()
    if key not in CROSSOVER_REGISTRY:
        raise KeyError(f"No RecombinationStrategy registered for '{name}'. Available: {list(CROSSOVER_REGISTRY.keys())}")
    return CROSSOVER_REGISTRY[key]


def list_crossover_strategies() -> list[str]:
    """Returns a sorted list of registered crossover strategy identifiers."""
    return sorted(list(CROSSOVER_REGISTRY.keys()))


def register_mutation(name: Union[str, Sequence[str]]) -> Callable[[Type[T]], Type[T]]:
    """A decorator to register a MutationStrategy subclass with one or more string identifiers."""
    def decorator(cls: Type[T]) -> Type[T]:
        from neutral_selection.variation.mutation.base import MutationStrategy
        if not issubclass(cls, MutationStrategy):
            raise TypeError(f"Class '{cls.__name__}' must inherit from MutationStrategy.")
        for identifier in _normalize_names(name):
            if identifier in MUTATION_REGISTRY:
                raise ValueError(f"Mutation strategy '{identifier}' is already registered to {MUTATION_REGISTRY[identifier].__name__}.")
            MUTATION_REGISTRY[identifier] = cls
        return cls
    return decorator


def get_mutation_strategy(name: str) -> Type[MutationStrategy]:
    """Retrieves the registered MutationStrategy subclass for the given identifier."""
    key = str(name).lower().strip()
    if key not in MUTATION_REGISTRY:
        raise KeyError(f"No MutationStrategy registered for '{name}'. Available: {list(MUTATION_REGISTRY.keys())}")
    return MUTATION_REGISTRY[key]


def list_mutation_strategies() -> list[str]:
    """Returns a sorted list of registered mutation strategy identifiers."""
    return sorted(list(MUTATION_REGISTRY.keys()))


def register_replacement(name: Union[str, Sequence[str]]) -> Callable[[Type[T]], Type[T]]:
    """A decorator to register a ReplacementStrategy subclass with one or more string identifiers."""
    def decorator(cls: Type[T]) -> Type[T]:
        from neutral_selection.replacement.base import ReplacementStrategy
        if not issubclass(cls, ReplacementStrategy):
            raise TypeError(f"Class '{cls.__name__}' must inherit from ReplacementStrategy.")
        for identifier in _normalize_names(name):
            if identifier in REPLACEMENT_REGISTRY:
                raise ValueError(f"Replacement strategy '{identifier}' is already registered to {REPLACEMENT_REGISTRY[identifier].__name__}.")
            REPLACEMENT_REGISTRY[identifier] = cls
        return cls
    return decorator


def get_replacement_strategy(name: str) -> Type[ReplacementStrategy]:
    """Retrieves the registered ReplacementStrategy subclass for the given identifier."""
    key = str(name).lower().strip()
    if key not in REPLACEMENT_REGISTRY:
        raise KeyError(f"No ReplacementStrategy registered for '{name}'. Available: {list(REPLACEMENT_REGISTRY.keys())}")
    return REPLACEMENT_REGISTRY[key]


def list_replacement_strategies() -> list[str]:
    """Returns a sorted list of registered replacement strategy identifiers."""
    return sorted(list(REPLACEMENT_REGISTRY.keys()))
