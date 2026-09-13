from __future__ import annotations

from typing import Any, Optional

from neutral_selection.variation.selection import (
    SelectionStrategy,
    TournamentSelection,
    RouletteWheelSelection,
    StochasticUniversalSamplingSelection,
    LinearRankSelection,
    ExponentialRankSelection,
    TruncationSelection,
    ElitistSelection,
    RandomSelection,
    BoltzmannSelection,
)
from neutral_selection.variation.recombination import (
    RecombinationStrategy,
    RandomNPointCrossover,
    NPointCrossover,
    OnePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    ShuffleCrossover,
    ArithmeticCrossover,
    BlendCrossover,
    SimulatedBinaryCrossover,
    OrderCrossover,
    PartiallyMatchedCrossover,
    CycleCrossover,
    ElementwiseCrossover,
)
from neutral_selection.variation.mutation import (
    MutationStrategy,
    UniformMutation,
    InversionMutation,
    SwapMutation,
    ScrambleMutation,
    InsertionMutation,
    TranspositionMutation,
    DuplicationMutation,
    DeletionMutation,
    GaussianMutation,
    UniformRealMutation,
    PolynomialMutation,
    CauchyMutation,
    BitFlipMutation,
    BoundaryMutation,
)
from neutral_selection.replacement import (
    ReplacementStrategy,
    GenerationalReplacement,
    PlusReplacement,
    CommaReplacement,
    SteadyStateReplacement,
)
from neutral_selection.pipeline import GenerationPipeline


def build_selection_strategy(config: dict[str, Any]) -> SelectionStrategy:
    """
    Constructs a SelectionStrategy instance from a configuration dictionary.

    Args:
        config: Dictionary specifying strategy configuration (e.g. `{"type": "tournament", "tournament_size": 3}`).

    Returns:
        The instantiated SelectionStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "tournament")).lower().strip()
    minimize = bool(config.get("minimize", False))

    if strategy_type == "tournament":
        tournament_size = config.get("tournament_size", 2)
        return TournamentSelection(tournament_size=tournament_size, minimize=minimize)
    elif strategy_type == "truncation":
        top_k = config.get("top_k", config.get("k"))
        top_ratio = config.get("top_ratio")
        if top_k is None and top_ratio is None:
            top_k = 2
        return TruncationSelection(top_k=top_k, top_ratio=top_ratio, minimize=minimize)
    elif strategy_type in ("roulette", "roulette_wheel", "proportionate"):
        return RouletteWheelSelection(minimize=minimize)
    elif strategy_type in ("stochastic_universal_sampling", "sus"):
        return StochasticUniversalSamplingSelection(minimize=minimize)
    elif strategy_type in ("linear_rank", "linear"):
        sp = float(config.get("selection_pressure", 1.5))
        return LinearRankSelection(selection_pressure=sp, minimize=minimize)
    elif strategy_type in ("exponential_rank", "exponential"):
        c = float(config.get("c", 0.9))
        return ExponentialRankSelection(c=c, minimize=minimize)
    elif strategy_type in ("elitist", "elite"):
        num_elites = config.get("num_elites", config.get("k", 1))
        elite_ratio = config.get("elite_ratio")
        return ElitistSelection(num_elites=num_elites, elite_ratio=elite_ratio, minimize=minimize)
    elif strategy_type == "random":
        return RandomSelection()
    elif strategy_type == "boltzmann":
        temperature = float(config.get("temperature", 1.0))
        return BoltzmannSelection(temperature=temperature, minimize=minimize)
    else:
        raise ValueError(f"Unknown selection strategy type: '{strategy_type}'")


def build_crossover_strategy(config: dict[str, Any]) -> RecombinationStrategy:
    """
    Constructs a RecombinationStrategy instance from a configuration dictionary.

    Args:
        config: Dictionary specifying crossover configuration (e.g. `{"type": "uniform", "swap_prob": 0.5}`).

    Returns:
        The instantiated RecombinationStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "random_n_point")).lower().strip()

    if strategy_type in ("random_n_point", "n_point"):
        num_cut_points = config.get("num_cut_points", 1)
        return RandomNPointCrossover(num_cut_points=num_cut_points)
    elif strategy_type == "one_point":
        return OnePointCrossover()
    elif strategy_type == "two_point":
        return TwoPointCrossover()
    elif strategy_type == "uniform":
        swap_prob = float(config.get("swap_prob", 0.5))
        return UniformCrossover(swap_prob=swap_prob)
    elif strategy_type == "shuffle":
        return ShuffleCrossover()
    elif strategy_type == "arithmetic":
        alpha = float(config.get("alpha", 0.5))
        return ArithmeticCrossover(alpha=alpha)
    elif strategy_type == "blend":
        alpha = float(config.get("alpha", 0.5))
        return BlendCrossover(alpha=alpha)
    elif strategy_type in ("simulated_binary", "sbx"):
        eta_c = float(config.get("eta_c", 20.0))
        bounds = config.get("bounds")
        return SimulatedBinaryCrossover(eta_c=eta_c, bounds=bounds)
    elif strategy_type in ("order", "ox"):
        return OrderCrossover()
    elif strategy_type in ("pmx", "partially_matched"):
        return PartiallyMatchedCrossover()
    elif strategy_type in ("cycle", "cx"):
        return CycleCrossover()
    elif strategy_type == "elementwise":
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
        return ElementwiseCrossover(sub_crossovers=sub_crossovers, default_crossover=default_crossover)
    else:
        raise ValueError(f"Unknown crossover strategy type: '{strategy_type}'")


def build_mutation_strategy(config: dict[str, Any]) -> MutationStrategy:
    """
    Constructs a MutationStrategy instance from a configuration dictionary.

    Args:
        config: Dictionary specifying mutation configuration (e.g. `{"type": "gaussian", "sigma": 0.05}`).

    Returns:
        The instantiated MutationStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "gaussian")).lower().strip()
    rate = float(config.get("mutation_rate", config.get("rate", 1.0)))

    if strategy_type == "gaussian":
        sigma = float(config.get("sigma", config.get("noise", 0.1)))
        bounds = config.get("bounds")
        return GaussianMutation(sigma=sigma, mutation_rate=rate, bounds=bounds)
    elif strategy_type in ("uniform_real", "real_uniform"):
        delta = float(config.get("delta", 1.0))
        bounds = config.get("bounds")
        return UniformRealMutation(delta=delta, mutation_rate=rate, bounds=bounds)
    elif strategy_type == "polynomial":
        eta_m = float(config.get("eta_m", 20.0))
        bounds = config.get("bounds", (-1.0, 1.0))
        return PolynomialMutation(eta_m=eta_m, bounds=bounds, mutation_rate=rate)
    elif strategy_type == "cauchy":
        scale = float(config.get("scale", config.get("gamma", 0.1)))
        bounds = config.get("bounds")
        return CauchyMutation(scale=scale, mutation_rate=rate, bounds=bounds)
    elif strategy_type in ("bit_flip", "bitflip", "binary"):
        flip_prob = float(config.get("flip_prob", 0.05))
        return BitFlipMutation(flip_prob=flip_prob, mutation_rate=rate)
    elif strategy_type == "boundary":
        bounds = config.get("bounds", (0.0, 1.0))
        return BoundaryMutation(bounds=bounds, mutation_rate=rate)
    elif strategy_type == "uniform":
        fn = config.get("mutation_fn")
        if fn is None:
            raise ValueError("UniformMutation requires 'mutation_fn' in config.")
        return UniformMutation(mutation_rate=rate, mutation_fn=fn)
    elif strategy_type == "inversion":
        return InversionMutation(mutation_rate=rate)
    elif strategy_type == "swap":
        return SwapMutation(mutation_rate=rate)
    elif strategy_type == "scramble":
        return ScrambleMutation(mutation_rate=rate)
    elif strategy_type == "insertion":
        return InsertionMutation(mutation_rate=rate)
    elif strategy_type == "transposition":
        return TranspositionMutation(mutation_rate=rate)
    elif strategy_type == "duplication":
        return DuplicationMutation(mutation_rate=rate)
    elif strategy_type == "deletion":
        return DeletionMutation(mutation_rate=rate)
    else:
        raise ValueError(f"Unknown mutation strategy type: '{strategy_type}'")


def build_replacement_strategy(config: dict[str, Any]) -> ReplacementStrategy:
    """
    Constructs a ReplacementStrategy instance from a configuration dictionary.

    Args:
        config: Dictionary specifying replacement configuration (e.g. `{"type": "generational", "num_elites": 1}`).

    Returns:
        The instantiated ReplacementStrategy.
    """
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dictionary, got {type(config).__name__}")

    strategy_type = str(config.get("type", "generational")).lower().strip()
    minimize = bool(config.get("minimize", False))

    if strategy_type in ("generational", "generational_replacement"):
        num_elites = config.get("num_elites", config.get("elitism", 0))
        elite_ratio = config.get("elite_ratio")
        return GenerationalReplacement(num_elites=num_elites, elite_ratio=elite_ratio, minimize=minimize)
    elif strategy_type in ("plus", "mu_plus_lambda", "plus_replacement"):
        return PlusReplacement(minimize=minimize)
    elif strategy_type in ("comma", "mu_comma_lambda", "comma_replacement"):
        return CommaReplacement(minimize=minimize)
    elif strategy_type in ("steady_state", "steady_state_replacement"):
        return SteadyStateReplacement(minimize=minimize)
    else:
        raise ValueError(f"Unknown replacement strategy type: '{strategy_type}'")


def build_pipeline(config: dict[str, Any]) -> GenerationPipeline:
    """
    Constructs a GenerationPipeline instance from a nested configuration dictionary.

    Args:
        config: Dictionary with configuration keys for selection, crossover, mutation, replacement, etc.

    Returns:
        The configured GenerationPipeline instance.
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

    return GenerationPipeline(
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
