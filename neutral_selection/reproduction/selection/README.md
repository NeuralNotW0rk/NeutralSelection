# Selection Strategies (`neutral_selection.reproduction.selection`)

This package implements established selection algorithms from the evolutionary computation literature. In evolutionary algorithms (EAs), selection operates at two distinct evolutionary stages:

1. **Parent Selection (Mating Selection)**: Choosing high-fitness individuals from the current population to undergo recombination and mutation to produce candidate offspring.
2. **Survivor Selection (Environmental Selection / Replacement)**: Determining which individuals from the parent population and the offspring pool survive to form the next generation.

---

## 1. Parent Selection Strategies

All parent selection strategies inherit from `SelectionStrategy` and provide:
- `.select(population, k=...)` / `__call__(population, k=...)`
- `.select_one(population)`
- `.select_pairs(population, num_pairs=...)`

### Comparison Table

| Strategy | Literature Reference | Mechanism / Key Property | When to Use |
| :--- | :--- | :--- | :--- |
| **`TournamentSelection`** | Goldberg & Deb (1991), Miller & Goldberg (1995) | Samples $T$ contestants randomly; fittest wins with probability $p \in (0.0, 1.0]$. | Universal default; invariant to monotonic fitness scaling, supports negative fitness, easily tunable selection pressure. |
| **`RouletteWheelSelection`** | Holland (1975), Goldberg (1989) | Probability is directly proportional to fitness: $p_i = f_i / \sum f_j$. | Classical Genetic Algorithms with well-scaled non-negative fitness functions. |
| **`StochasticUniversalSamplingSelection` (SUS)** | Baker (1987) | Single-spin roulette wheel with $k$ equally-spaced pointers ($1/k$ step). | Zero sampling bias and minimal spread; eliminates stochastic drift. |
| **`LinearRankSelection`** | Baker (1985), Whitley (1989) | Assigns probabilities linearly according to sorted fitness rank: $s \in [1.0, 2.0]$. | Prevents premature convergence caused by "super-individuals" early in the run. |
| **`ExponentialRankSelection`** | Blickle & Thiele (1995) | Assigns selection probabilities exponentially by rank: $w_i = c^{N - 1 - i}$ for $c \in (0, 1)$. | High, non-linear selection pressure on top ranks. |
| **`TruncationSelection`** | Mühlenbein & Schlierkamp-Voosen (1993) | Samples uniformly from the top $k$ or top $T\%$ fittest individuals. | Standard in Evolution Strategies (ES) and Breeder Genetic Algorithms (BGA). |
| **`ElitistSelection`** | De Jong (1975) | Deterministically extracts the top $k$ unique fittest individuals without replacement. | Elite preservation and direct top-performer extraction. |
| **`RandomSelection`** | Kimura (1968, 1983) | Samples individuals uniformly at random regardless of fitness. | Modeling neutral genetic drift and neutral baseline evolution. |
| **`BoltzmannSelection`** | Mahfoud (1995), De la Maza & Tidor (1993) | Softmax / Boltzmann probability: $p_i \propto \exp(f_i / T)$ with temperature $T > 0$. | Annealed selection schedules (exploration $\rightarrow$ exploitation). |

---

## 2. Survivor Selection Strategies

All survivor selection strategies inherit from `SurvivorStrategy` and provide:
- `.select_survivors(parents, offspring, target_size=...)` / `__call__(parents, offspring, target_size=...)`

### Comparison Table

| Strategy | Scheme | Description | References |
| :--- | :--- | :--- | :--- |
| **`GenerationalReplacement`** | $(\mu, \lambda)$ with Elitism | Preserves top $e$ elites from parents; fills remaining slots from offspring. | De Jong (1975) |
| **`PlusReplacement`** | $(\mu + \lambda)$ | Merges parents and offspring ($\mu + \lambda$) and selects the top $\mu$ survivors. | Rechenberg (1973), Schwefel (1981) |
| **`CommaReplacement`** | $(\mu, \lambda)$ | Discards parents entirely; selects $\mu$ survivors strictly from offspring ($\lambda \ge \mu$). | Schwefel (1981), Bäck (1996) |
| **`SteadyStateReplacement`** | Steady-State GA | Retains the best parents and replaces the worst individuals with offspring. | Syswerda (1989), Whitley (1989) |

---

## 3. Usage Examples

### Parent Selection & Mating
```python
from neutral_selection import (
    Population,
    TournamentSelection,
    LinearRankSelection,
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
next_gen = select_survivors(parents=current_population, offspring=new_offspring, strategy=survivor_strat)
```

---

## References

1. Baker, J. E. (1985). *Adaptive selection methods for genetic algorithms*. Proceedings of an International Conference on Genetic Algorithms and their Applications, 101–111.
2. Baker, J. E. (1987). *Reducing bias and inefficiency in the selection algorithm*. Proceedings of the Second International Conference on Genetic Algorithms, 14–21.
3. Blickle, T., & Thiele, L. (1995). *A comparison of selection schemes used in evolutionary algorithms*. Evolutionary Computation, 4(4), 361–394.
4. De Jong, K. A. (1975). *An analysis of the behavior of a class of genetic adaptive systems*. Doctoral dissertation, University of Michigan.
5. De la Maza, M., & Tidor, B. (1993). *An analysis of selection procedures with particular attention to Boltzmann selection*. Proceedings of the 5th International Conference on Genetic Algorithms, 124–131.
6. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
7. Goldberg, D. E., & Deb, K. (1991). *A comparative analysis of selection schemes used in genetic algorithms*. Foundations of Genetic Algorithms, 1, 69–93.
8. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
9. Kimura, M. (1968). *Evolutionary rate at the molecular level*. Nature, 217(5129), 624–626.
10. Kimura, M. (1983). *The Neutral Theory of Molecular Evolution*. Cambridge University Press.
11. Mahfoud, S. W. (1995). *Niching methods for genetic algorithms*. Doctoral dissertation, University of Illinois at Urbana-Champaign.
12. Miller, B. L., & Goldberg, D. E. (1995). *Genetic algorithms, tournament selection, and the effects of noise*. Complex Systems, 9(3), 193–212.
13. Mühlenbein, H., & Schlierkamp-Voosen, D. (1993). *Predictive models for the breeder genetic algorithm: I. Continuous parameter optimization*. Evolutionary Computation, 1(1), 25–49.
14. Rechenberg, I. (1973). *Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution*. Frommann-Holzboog.
15. Schwefel, H. P. (1981). *Numerical Optimization of Computer Models*. John Wiley & Sons.
16. Syswerda, G. (1989). *Uniform crossover in genetic algorithms*. Proceedings of the 3rd International Conference on Genetic Algorithms, 2–9.
17. Whitley, D. (1989). *The GENITOR algorithm and selection pressure: Why rank-based allocation of reproductive trials is best*. Proceedings of the 3rd International Conference on Genetic Algorithms, 116–121.
