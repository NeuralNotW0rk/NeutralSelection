from __future__ import annotations

import inspect
from typing import Any, Optional, Type, TypeVar, Union

from neutral_selection.registry import (
    get_selection_strategy,
    get_crossover_strategy,
    get_mutation_strategy,
    get_replacement_strategy,
)
from neutral_selection.variation.selection import SelectionStrategy, TournamentSelection
from neutral_selection.variation.recombination import RecombinationStrategy, ElementwiseCrossover
from neutral_selection.variation.mutation import MutationStrategy
from neutral_selection.replacement import ReplacementStrategy
from neutral_selection.pipeline import EvolutionPipeline

T = TypeVar("T")


def _instantiate_from_config(
    cls: Type[T],
    config: dict[str, Any],
    param_aliases: Optional[dict[str, Union[str, list[str]]]] = None,
    defaults: Optional[dict[str, Any]] = None,
) -> T:
    """
    Instantiates a strategy class using reflection on its __init__ signature,
    extracting matching keys from config while honoring parameter aliases and defaults.
    """
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}

    alias_lookup: dict[str, list[str]] = {}
    if param_aliases:
        for target_param, aliases in param_aliases.items():
            if isinstance(aliases, str):
                alias_lookup[target_param] = [aliases]
            else:
                alias_lookup[target_param] = list(aliases)

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "args", "kwargs"):
            continue

        if param_name in config:
            kwargs[param_name] = config[param_name]
        else:
            found = False
            for alias in alias_lookup.get(param_name, []):
                if alias in config:
                    kwargs[param_name] = config[alias]
                    found = True
                    break
            if not found and defaults and param_name in defaults:
                kwargs[param_name] = defaults[param_name]

    return cls(**kwargs)


def build_selection_strategy(config: dict[str, Any]) -> SelectionStrategy:
    """
    Constructs a SelectionStrategy instance from a configuration dictionary using the Selection Registry.

    Args:
        config: Dictionary specifying strategy configuration (e.g. `{"type": "tournament", "tournament_size": 3}`).

    Returns:
        The instantiated SelectionStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "tournament")).lower().strip()
    cls = get_selection_strategy(strategy_type)

    param_aliases = {
        "top_k": ["top_k", "k"],
        "num_elites": ["num_elites", "k"],
        "selection_pressure": ["selection_pressure", "sp"],
    }

    defaults: dict[str, Any] = {}
    if cls.__name__ == "TruncationSelection":
        if "top_k" not in config and "k" not in config and "top_ratio" not in config:
            defaults["top_k"] = 2
    elif cls.__name__ == "ElitistSelection":
        if "num_elites" not in config and "k" not in config and "elite_ratio" not in config:
            defaults["num_elites"] = 1

    return _instantiate_from_config(cls, config, param_aliases=param_aliases, defaults=defaults)


def build_crossover_strategy(config: dict[str, Any]) -> RecombinationStrategy:
    """
    Constructs a RecombinationStrategy instance from a configuration dictionary using the Crossover Registry.

    Args:
        config: Dictionary specifying crossover configuration (e.g. `{"type": "uniform", "swap_prob": 0.5}`).

    Returns:
        The instantiated RecombinationStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "random_n_point")).lower().strip()
    cls = get_crossover_strategy(strategy_type)

    if issubclass(cls, ElementwiseCrossover):
        sub_crossovers_raw = config.get("sub_crossovers", {})
        sub_crossovers = {
            k: build_crossover_strategy(v) if isinstance(v, dict) else v
            for k, v in sub_crossovers_raw.items()
        }
        default_crossover_raw = config.get("default_crossover")
        default_crossover = (
            build_crossover_strategy(default_crossover_raw)
            if isinstance(default_crossover_raw, dict)
            else default_crossover_raw
        )
        return cls(sub_crossovers=sub_crossovers, default_crossover=default_crossover)

    return _instantiate_from_config(cls, config)


def build_mutation_strategy(config: dict[str, Any]) -> MutationStrategy:
    """
    Constructs a MutationStrategy instance from a configuration dictionary using the Mutation Registry.

    Args:
        config: Dictionary specifying mutation configuration (e.g. `{"type": "gaussian", "sigma": 0.05}`).

    Returns:
        The instantiated MutationStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "gaussian")).lower().strip()
    cls = get_mutation_strategy(strategy_type)

    if cls.__name__ == "UniformMutation" and "mutation_fn" not in config:
        raise ValueError("UniformMutation requires 'mutation_fn' in config.")

    param_aliases = {
        "mutation_rate": ["mutation_rate", "rate", "flip_prob"],
        "sigma": ["sigma", "noise"],
        "scale": ["scale", "gamma"],
    }

    defaults: dict[str, Any] = {}
    if cls.__name__ == "PolynomialMutation" and "bounds" not in config:
        defaults["bounds"] = (-1.0, 1.0)
    elif cls.__name__ == "BoundaryMutation" and "bounds" not in config:
        defaults["bounds"] = (0.0, 1.0)

    return _instantiate_from_config(cls, config, param_aliases=param_aliases, defaults=defaults)


def build_replacement_strategy(config: dict[str, Any]) -> ReplacementStrategy:
    """
    Constructs a ReplacementStrategy instance from a configuration dictionary using the Replacement Registry.

    Args:
        config: Dictionary specifying replacement configuration (e.g. `{"type": "generational", "num_elites": 1}`).

    Returns:
        The instantiated ReplacementStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "generational")).lower().strip()
    cls = get_replacement_strategy(strategy_type)

    param_aliases = {
        "num_elites": ["num_elites", "elitism"],
    }

    return _instantiate_from_config(cls, config, param_aliases=param_aliases)


def build_pipeline(config: dict[str, Any]) -> EvolutionPipeline:
    """
    Constructs an EvolutionPipeline instance from a nested configuration dictionary.

    Args:
        config: Dictionary with configuration keys for selection, crossover, mutation, replacement, etc.

    Returns:
        The configured EvolutionPipeline instance.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    # 1. Selection
    selection_raw = config.get("selection")
    if selection_raw is None:
        selection_strat: SelectionStrategy = TournamentSelection(tournament_size=2)
    elif isinstance(selection_raw, SelectionStrategy):
        selection_strat = selection_raw
    elif isinstance(selection_raw, dict):
        selection_strat = build_selection_strategy(selection_raw)
    else:
        raise TypeError(f"selection must be a SelectionStrategy or dict, got {type(selection_raw).__name__}")

    # 2. Crossover
    crossover_raw = config.get("crossover")
    crossover_strat: Optional[RecombinationStrategy] = None
    if isinstance(crossover_raw, RecombinationStrategy):
        crossover_strat = crossover_raw
    elif isinstance(crossover_raw, dict):
        crossover_strat = build_crossover_strategy(crossover_raw)

    # 3. Mutation
    mutation_raw = config.get("mutation")
    mutation_strat: Optional[MutationStrategy] = None
    if isinstance(mutation_raw, MutationStrategy):
        mutation_strat = mutation_raw
    elif isinstance(mutation_raw, dict):
        mutation_strat = build_mutation_strategy(mutation_raw)

    # 4. Replacement
    replacement_raw = config.get("replacement")
    replacement_strat: Optional[ReplacementStrategy] = None
    if isinstance(replacement_raw, ReplacementStrategy):
        replacement_strat = replacement_raw
    elif isinstance(replacement_raw, dict):
        replacement_strat = build_replacement_strategy(replacement_raw)

    crossover_prob = float(config.get("crossover_prob", 1.0))
    mutation_prob = float(config.get("mutation_prob", 1.0))
    evaluate_fn = config.get("evaluate_fn")

    if replacement_strat is not None:
        elitism = 0
        elite_ratio = None
    else:
        elitism = int(config.get("elitism", 0))
        elite_ratio = config.get("elite_ratio")

    return EvolutionPipeline(
        selection_strategy=selection_strat,
        crossover_strategy=crossover_strat,
        mutation_strategy=mutation_strat,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        elitism=elitism,
        elite_ratio=elite_ratio,
        evaluate_fn=evaluate_fn,
        replacement_strategy=replacement_strat,
    )
