# NeutralSelection

> **A modern, strictly-typed, and composable evolutionary computation framework for Python.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Why NeutralSelection?

Existing Python evolutionary libraries often fall into two extremes:
* **`pymoo`** is powerful for multi-objective numerical optimization, but is strictly bound to flat 2D NumPy matrices (`N_individuals × N_variables`). It offers no native ergonomics for nested structures, variable-length sequences, or heterogeneous genomes.
* **`DEAP`** is flexible, but relies on 15-year-old untyped metaprogramming (`creator.create`), global registries, and lacks modern typing.

**NeutralSelection** bridges this gap. It provides a modular, functional, and strictly-typed toolkit for evolutionary algorithms—designed for everything from classical optimization to **hierarchical genomes, latent space exploration, PyTorch tensor manipulation, and creative generative pipelines**.

---

## Core Value Proposition

* 🧩 **Composable Functional Primitives:** Every operator is a standalone callable (`strategy(parent_a, parent_b)`) and functional helper (`recombine`, `mutate`, `select`, `replace`, `step`). Drop operators directly into any custom loop, interactive tool, or neural pipeline.
* 🌳 **Heterogeneous & Segmented Genomes:** First-class support for multi-part chromosomes via `Segment` and `ElementwiseCrossover`. Different sections of a genome can have distinct representations, bounds, mutation rates, and crossover rules.
* ⚡ **PyTorch & Generative AI Friendly:** Native ergonomics for mutating and blending PyTorch tensors, NumPy arrays, latent vectors, and named model attributes.
* 🛡️ **Modern Python & Strict Type Safety:** Built from the ground up for Python 3.10+ with comprehensive type annotations, dataclasses, explicit error boundaries, and zero metaprogramming magic.

---

## Implemented Strategies

### 1. Variation — Parent Selection (`neutral_selection.variation.selection`)
* `TournamentSelection`, `RouletteWheelSelection`, `StochasticUniversalSamplingSelection` (SUS), `LinearRankSelection`, `ExponentialRankSelection`, `TruncationSelection`, `ElitistSelection`, `RandomSelection`, `BoltzmannSelection`
* Functional helper: `select(population, strategy, k)`

### 2. Variation — Recombination (`neutral_selection.variation.recombination`)
* **Positional & Structural:** `OnePointCrossover`, `TwoPointCrossover`, `NPointCrossover`, `RandomNPointCrossover`, `UniformCrossover`, `ShuffleCrossover`
* **Permutation & Order-Preserving:** `OrderCrossover` (OX1), `PartiallyMatchedCrossover` (PMX), `CycleCrossover` (CX)
* **Real-Valued & Continuous:** `ArithmeticCrossover`, `BlendCrossover` (BLX-$\alpha$), `SimulatedBinaryCrossover` (SBX)
* **Structure-Agnostic:** `ElementwiseCrossover` (recursive multi-segment blending)
* Functional helper: `recombine(parent_a, parent_b, strategy)`

### 3. Variation — Mutation (`neutral_selection.variation.mutation`)
* **Sequence & Permutation:** `InversionMutation` (2-opt), `SwapMutation`, `ScrambleMutation`, `InsertionMutation`, `TranspositionMutation`, `DuplicationMutation`, `DeletionMutation`
* **Real-Valued & Continuous:** `GaussianMutation`, `UniformRealMutation`, `PolynomialMutation` (NSGA-II), `CauchyMutation` (Fast EP)
* **Binary & Discrete:** `BitFlipMutation`, `BoundaryMutation`
* **Structure-Agnostic Mutators:** `UniformMutation`, `gaussian_noise_mutator`, `bit_flip_mutator`, `attribute_mutator`
* Functional helper: `mutate(genome, strategy)`

### 4. Environmental Replacement (`neutral_selection.replacement`)
* `GenerationalReplacement` (Elitism / $(\mu, \lambda)$ with elite retention), `PlusReplacement` $(\mu + \lambda)$, `CommaReplacement` $(\mu, \lambda)$, `SteadyStateReplacement`
* Functional helper: `replace(parents, offspring, strategy, target_size=None)`

### 5. Generational Pipeline & Declarative Builders (`neutral_selection.pipeline`, `neutral_selection.builders`)
* `GenerationPipeline`: Coordinates parent selection, variation (crossover + mutation), evaluation, and survivor replacement.
* Functional helper: `step(parents, selection_strategy, ...)`
* Builders: `build_selection_strategy`, `build_crossover_strategy`, `build_mutation_strategy`, `build_replacement_strategy`, `build_pipeline`

---

## Quickstart

### 1. Basic Sequence Crossover & Mutation
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

### 2. Heterogeneous & Segmented Genomes
```python
from neutral_selection import (
    Genome,
    Segment,
    ElementwiseCrossover,
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

blend_fn = lambda s1, s2: segment_crossovers[s1.key](s1, s2)
child_1, child_2 = recombine(parent_1, parent_2, ElementwiseCrossover(blend_fn=blend_fn))
```

### 3. Parent Selection & Replacement
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

### 4. Full Generational Step
```python
from neutral_selection import (
    GenerationPipeline,
    TournamentSelection,
    UniformCrossover,
    GaussianMutation,
)

pipeline = GenerationPipeline(
    selection_strategy=TournamentSelection(tournament_size=2),
    crossover_strategy=UniformCrossover(swap_prob=0.5),
    mutation_strategy=GaussianMutation(sigma=0.1),
    elitism=1,
)

next_gen = pop.step(pipeline=pipeline)
```

---

## General References & Recommended Reading

* **Eiben, A. E., & Smith, J. E. (2015).** *Introduction to Evolutionary Computing* (2nd ed.). Springer. *(Standard comprehensive reference for EA representations and mechanics).*
* **Deb, K. (2001).** *Multi-Objective Optimization using Evolutionary Algorithms*. John Wiley & Sons. *(Real-parameter genetic algorithms, SBX, and polynomial mutation).*
* **Rothlauf, F. (2006).** *Representations for Genetic and Evolutionary Algorithms* (2nd ed.). Springer. *(Permutation representations, locality, and schema preservation).*
* **Luke, S. (2013).** *Essentials of Metaheuristics* (2nd ed.). Lulu / George Mason University. *(Algorithmic definitions and practical operator implementations).*
