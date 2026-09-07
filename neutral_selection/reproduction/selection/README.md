# Selection Strategies (`neutral_selection.reproduction.selection`)

This package implements established parent and survivor selection algorithms for evolutionary algorithms. Selection operates at two distinct lifecycle stages:

1. **Parent Selection (Mating Selection)**: Choosing high-fitness individuals from the current population to undergo recombination and mutation.
2. **Survivor Selection (Environmental Selection / Replacement)**: Determining which individuals from the parent population and offspring pool form the next generation.

---

## 1. Parent Selection Strategies

All parent selection strategies inherit from `SelectionStrategy` and support `.select(population, k=...)`, `.select_one(population)`, and `.select_pairs(population, num_pairs=...)`.

| Strategy | Key Parameters & Defaults | Scaling & Negative Fitness | Key Property & Recommended Usage |
| :--- | :--- | :--- | :--- |
| **`TournamentSelection`** | `tournament_size: int = 2`, `winner_prob: float = 1.0`, `minimize: bool = False` | Scale-invariant (Supports negative) | Samples $T$ contestants randomly; fittest wins with probability $p$. Universal default; easily tunable selection pressure. |
| **`RouletteWheelSelection`** | `minimize: bool = False` | Sensitive to offset (Requires non-negative) | Fitness-proportionate selection: $p_i = f_i / \sum f_j$. Standard in classical Genetic Algorithms. |
| **`StochasticUniversalSamplingSelection` (SUS)** | `minimize: bool = False` | Sensitive to offset (Requires non-negative) | Single-spin roulette wheel with $k$ equally-spaced pointers ($1/k$ step). Zero sampling bias and minimal spread. |
| **`LinearRankSelection`** | `s: float = 1.5`, `minimize: bool = False` | Scale-invariant (Supports negative) | Assigns probabilities linearly by rank with selection pressure $s \in [1.0, 2.0]$. Prevents premature stagnation from super-individuals. |
| **`ExponentialRankSelection`** | `c: float = 0.9`, `minimize: bool = False` | Scale-invariant (Supports negative) | Assigns selection probabilities exponentially by rank ($w_i = c^{N - 1 - i}$ for $c \in (0, 1)$). High, non-linear selection pressure on top ranks. |
| **`TruncationSelection`** | `k_best: Optional[int] = None`, `truncation_ratio: Optional[float] = None`, `minimize: bool = False` | Scale-invariant (Supports negative) | Samples uniformly from the top $k$ (or top $T\%$) fittest individuals. Standard in Evolution Strategies and Breeder GAs. |
| **`ElitistSelection`** | `num_elites: int = 1`, `elite_ratio: Optional[float] = None`, `minimize: bool = False` | Scale-invariant (Supports negative) | Deterministically extracts the top $k$ unique fittest individuals without replacement. |
| **`RandomSelection`** | *(None)* | Independent of fitness | Samples individuals uniformly at random regardless of fitness. Ideal for neutral walks, neutral genetic drift, and baselines. |
| **`BoltzmannSelection`** | `temperature: float = 1.0`, `minimize: bool = False` | Scale-dependent (Supports negative) | Softmax / Boltzmann probability: $p_i \propto \exp(f_i / T)$ with temperature $T > 0$. Useful for annealed selection schedules. |

---

## 2. Survivor Selection Strategies

All survivor selection strategies inherit from `SurvivorStrategy` and provide `.select_survivors(parents, offspring, target_size=...)`.

| Strategy | Formal Scheme | Key Parameters & Defaults | Description & Mechanics |
| :--- | :--- | :--- | :--- |
| **`GenerationalReplacement`** | $(\mu, \lambda)$ with Elitism | `num_elites: int = 0`, `elite_ratio: Optional[float] = None`, `minimize: bool = False` | Preserves the top $e$ elites from parents; fills remaining slots from offspring. |
| **`PlusReplacement`** | $(\mu + \lambda)$ | `minimize: bool = False` | Merges parents and offspring into a combined pool of size $(\mu + \lambda)$ and retains the top $\mu$ fittest. Guarantees monotonic fitness progress. |
| **`CommaReplacement`** | $(\mu, \lambda)$ | `minimize: bool = False` | Discards parents entirely; selects $\mu$ survivors strictly from offspring ($\lambda \ge \mu$). Helps escape local optima in continuous domains. |
| **`SteadyStateReplacement`** | Steady-State GA | `minimize: bool = False` | Retains top parents and replaces the worst individuals in the population with new offspring. |

---

## 3. Usage Examples

### Parent Selection & Mating
```python
from neutral_selection import (
    Population,
    TournamentSelection,
    LinearRankSelection,
    recombine,
    select,
)

# Selecting mating pairs for recombination
tournament = TournamentSelection(tournament_size=3)
parent_pairs = tournament.select_pairs(population, num_pairs=5)

for parent_a, parent_b in parent_pairs:
    children = recombine(parent_a, parent_b, recombination_strategy)
```

### Survivor Selection (Generational with Elitism)
```python
from neutral_selection import GenerationalReplacement, select_survivors

# Carry over top 2 elites from parents, fill remainder from offspring
survivor_strat = GenerationalReplacement(num_elites=2)
next_gen = select_survivors(
    parents=current_population,
    offspring=new_offspring,
    strategy=survivor_strat,
)
```

---

## Further Reading & General References

For comparative analyses of selection pressure, takeover times, and replacement dynamics:

* **Eiben, A. E., & Smith, J. E. (2015).** *Introduction to Evolutionary Computing* (2nd ed.). Springer Natural Computing Series. *(Comprehensive breakdown of parent/survivor selection mechanisms and selection intensity).*
* **Blickle, T., & Thiele, L. (1995).** *A comparison of selection schemes used in evolutionary algorithms*. Evolutionary Computation, 4(4), 361–394. *(Rigorous analysis of selection pressure, loss of diversity, and takeover times).*
* **Hassanat, A. et al. (2019).** *Selection Methods for Genetic Algorithms: A Comparative Study*. Mathematics, 7(9), 794. *(Empirical benchmark of modern selection operators across diverse fitness landscapes).*
* **Kimura, M. (1983).** *The Neutral Theory of Molecular Evolution*. Cambridge University Press. *(Foundational theory for neutral genetic drift and neutral selection).*
