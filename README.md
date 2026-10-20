# Neutral Selection

*"If I don't survive, tell my wife, hello."*

**Neutral Selection** is a strictly typed, composable evolutionary-computation framework for Python. It provides selection, recombination, mutation and replacement operators as standalone callables. It supports hierarchical and heterogeneous genomes, and it handles PyTorch tensors natively. Its range covers classical optimization, latent-space exploration and creative generative pipelines.

> **Status:** early alpha (`0.1.0`). The API may change between releases.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Contents

- [Why Neutral Selection](#why-neutral-selection)
- [Installation](#installation)
- [Core concepts](#core-concepts)
- [Implemented strategies](#implemented-strategies)
- [Quickstart](#quickstart)
- [Scripts and tests](#scripts-and-tests)
- [Related projects](#related-projects)
- [References and recommended reading](#references-and-recommended-reading)
- [License](#license)

---

## Why Neutral Selection

Existing Python evolutionary libraries tend to fall into one of two extremes:

- **`pymoo`** is powerful for multi-objective numerical optimization, but it is bound to flat 2-D NumPy matrices (`N_individuals × N_variables`). It has no native support for nested structures, variable-length sequences, or heterogeneous genomes.
- **`DEAP`** is flexible, but it relies on 15-year-old untyped metaprogramming (`creator.create`) and global registries, and it lacks modern typing.

Neutral Selection sits between the two. It is a modular, functional and strictly typed toolkit for evolutionary algorithms. It is designed for everything from classical optimization to **hierarchical genomes, latent-space exploration, PyTorch tensor manipulation, and creative generative pipelines**.

---

## Installation

Neutral Selection is not yet published on PyPI. Install it from source:

```bash
pip install git+https://github.com/NeuralNotW0rk/NeutralSelection.git
```

or, for development:

```bash
git clone https://github.com/NeuralNotW0rk/NeutralSelection.git neutral-selection
cd neutral-selection
pip install -e .
```

**Requirements:** Python 3.10, the only version tested so far. The core library has no third-party dependencies. NumPy and PyTorch are used when your genomes contain them.

---

## Core concepts

| Concept | Role |
|---|---|
| **Composable functional primitives** | Every operator is a standalone callable (`strategy(parent_a, parent_b)`). Functional helpers (`recombine`, `mutate`, `select`, `replace`, `step`) let you drop operators into any custom loop, interactive tool, or neural pipeline. |
| **Heterogeneous, segmented genomes** | Multi-part chromosomes are first-class through `Segment`. Each operator's leaf scale is set by `granularity` (`"atom"`, `"tensor"`, `"gene"` or a depth). Different sections of a genome can have their own representation, bounds, mutation rates and crossover rules. |
| **PyTorch and generative-AI friendly** | Native support for mutating and blending PyTorch tensors, NumPy arrays, latent vectors, and named model attributes. |
| **Modern Python, strict typing** | Built for Python 3.10, with comprehensive type annotations, dataclasses, explicit error boundaries, and no metaprogramming magic. |

The [variation README](neutral_selection/variation/README.md#granularity) explains granularity in detail.

---

## Implemented strategies

### Parent selection (`neutral_selection.variation.selection`)

- `TournamentSelection`, `RouletteWheelSelection`, `StochasticUniversalSamplingSelection` (SUS), `LinearRankSelection`, `ExponentialRankSelection`, `TruncationSelection`, `ElitistSelection`, `RandomSelection`, `BoltzmannSelection`
- **Functional helper:** `select(population, strategy, k)`

### Recombination (`neutral_selection.variation.recombination`)

- **Positional and structural:** `OnePointCrossover`, `TwoPointCrossover`, `NPointCrossover`, `RandomNPointCrossover`, `UniformCrossover`, `ShuffleCrossover`
- **Permutation and order-preserving:** `OrderCrossover` (OX1), `PartiallyMatchedCrossover` (PMX), `CycleCrossover` (CX)
- **Real-valued and continuous:** `ArithmeticCrossover`, `BlendCrossover` (BLX-$\alpha$), `SimulatedBinaryCrossover` (SBX)
- **Callable:** `FunctionCrossover` (custom leaf-pair combination at any granularity). `ElementwiseCrossover` is its deprecated gene-scale alias.
- **Functional helper:** `recombine(parent_a, parent_b, strategy)`

### Mutation (`neutral_selection.variation.mutation`)

- **Sequence and permutation:** `InversionMutation` (2-opt), `SwapMutation`, `ScrambleMutation`, `InsertionMutation`, `TranspositionMutation`, `DuplicationMutation`, `DeletionMutation`
- **Real-valued and continuous:** `GaussianMutation`, `UniformRealMutation`, `PolynomialMutation` (NSGA-II), `CauchyMutation` (Fast EP)
- **Binary and discrete:** `BitFlipMutation`, `BoundaryMutation`
- **Callable:** `FunctionMutation` (custom per-leaf mutation at any granularity). `UniformMutation` is its deprecated gene-scale alias.
- **Mutator helpers:** `gaussian_noise_mutator`, `bit_flip_mutator`, `attribute_mutator`
- **Functional helper:** `mutate(genome, strategy)`

### Environmental replacement (`neutral_selection.replacement`)

- `GenerationalReplacement` (elitism, i.e. $(\mu, \lambda)$ with elite retention), `PlusReplacement` $(\mu + \lambda)$, `CommaReplacement` $(\mu, \lambda)$, `SteadyStateReplacement`
- **Functional helper:** `replace(parents, offspring, strategy, target_size=None)`

### Pipeline and declarative builders (`neutral_selection.pipeline`, `.builders`, `.registry`)

- **`EvolutionPipeline`:** coordinates parent selection, variation (crossover and mutation), evaluation, and survivor replacement.
- **Functional helper:** `step(parents, selection_strategy, ...)`
- **Builders:** `build_selection_strategy`, `build_crossover_strategy`, `build_mutation_strategy`, `build_replacement_strategy`, `build_pipeline`
- **Registries:** `@register_selection`, `@register_crossover`, `@register_mutation` and `@register_replacement` register your own strategies.

Per-operator parameters and formulas are in the [variation](neutral_selection/variation/README.md) and [replacement](neutral_selection/replacement/README.md) READMEs.

---

## Quickstart

### Sequence crossover and mutation

```python
from neutral_selection import (
    Genome,
    TwoPointCrossover,
    InversionMutation,
    recombine,
    mutate,
)

parent_a = Genome([1, 2, 3, 4, 5, 6])
parent_b = Genome([10, 20, 30, 40, 50, 60])

# Standalone functional recombination
child_a, child_b = recombine(parent_a, parent_b, TwoPointCrossover())

# Standalone mutation
mutated_child = mutate(child_a, InversionMutation())
```

### Heterogeneous, segmented genomes

```python
from neutral_selection import (
    Genome,
    Segment,
    FunctionCrossover,
    SimulatedBinaryCrossover,
    PartiallyMatchedCrossover,
    UniformCrossover,
    recombine,
)

# A composite genome: continuous hyperparameters, permutation order, and discrete flags
def create_individual():
    return Genome([
        Segment("continuous_params", [0.5, -1.2, 3.4]),
        Segment("permutation_order", [0, 1, 2, 3, 4]),
        Segment("discrete_flags", [True, False, True]),
    ])

parent_1 = create_individual()
parent_2 = create_individual()

# Map custom crossover operators to specific segments
segment_crossovers = {
    "continuous_params": SimulatedBinaryCrossover(eta_c=2.0, bounds=(-5.0, 5.0)),
    "permutation_order": PartiallyMatchedCrossover(),
    "discrete_flags": UniformCrossover(swap_prob=0.5),
}

crossover_fn = lambda s1, s2, _: segment_crossovers[s1.key](s1, s2)[0]

# granularity=1 makes each top-level Segment one leaf, so each reaches its own operator whole
child_1, child_2 = recombine(parent_1, parent_2, FunctionCrossover(crossover_fn, granularity=1))
```

### Parent selection and replacement

```python
from neutral_selection import (
    Individual,
    Population,
    TournamentSelection,
    GenerationalReplacement,
    replace,
)

pop = Population([
    Individual(genome=Genome([1, 2, 3]), fitness=12.5),
    Individual(genome=Genome([4, 5, 6]), fitness=8.2),
    Individual(genome=Genome([7, 8, 9]), fitness=19.4),
])

# Select mating pairs
tournament = TournamentSelection(tournament_size=2)
mating_pairs = tournament.select_pairs(pop, num_pairs=2)

# Environmental replacement (carrying over top 1 elite)
survivor_strategy = GenerationalReplacement(num_elites=1)
next_generation = replace(
    parents=pop,
    offspring=offspring_pool,
    strategy=survivor_strategy,
)
```

### Full evolutionary step

```python
from neutral_selection import (
    EvolutionPipeline,
    TournamentSelection,
    UniformCrossover,
    GaussianMutation,
)

pipeline = EvolutionPipeline(
    selection_strategy=TournamentSelection(tournament_size=2),
    crossover_strategy=UniformCrossover(swap_prob=0.5),
    mutation_strategy=GaussianMutation(sigma=0.1),
    elitism=1,
)

next_gen = pop.step(pipeline=pipeline)
```

---

## Scripts and tests

```bash
python -m unittest discover tests
```

---

## Related projects

- [Diffracture](https://github.com/NeuralNotW0rk/Diffracture): a modular PyTorch library for model interventions (LoRA, DoRA, network bending).
- [StemmA2A](https://github.com/NeuralNotW0rk/StemmA2A): a desktop workbench that uses Neutral Selection to evolve per-layer LoRA genomes for generative audio and image models.

---

## References and recommended reading

> A. E. Eiben and J. E. Smith. *Introduction to Evolutionary Computing* (2nd ed.). Springer, 2015. The standard comprehensive reference for EA representations and mechanics.

> Kalyanmoy Deb. *Multi-Objective Optimization using Evolutionary Algorithms.* John Wiley & Sons, 2001. Covers real-parameter genetic algorithms, SBX, and polynomial mutation.

> Franz Rothlauf. *Representations for Genetic and Evolutionary Algorithms* (2nd ed.). Springer, 2006. Covers permutation representations, locality, and schema preservation.

> Sean Luke. *Essentials of Metaheuristics* (2nd ed.). Lulu / George Mason University, 2013. Algorithmic definitions and practical operator implementations.

---

## License

Neutral Selection is released under the [MIT License](LICENSE).
