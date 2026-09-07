# Mutation Strategies (`neutral_selection.reproduction.mutation`)

This package implements established mutation strategies from the evolutionary computation literature, supporting sequence/permutation representations, real-valued/continuous vectors, binary encodings, and nested/hierarchical genome structures.

All mutation strategies inherit from `MutationStrategy` and can be applied directly as callables `strategy(genome)` or via the functional helper `mutate(target, strategy)`.

---

## 1. Sequence & Permutation Mutation

| Strategy | Literature Reference | Mechanism / Key Property |
| :--- | :--- | :--- |
| **`InversionMutation`** | Holland (1975), Fogel (1988) | Reverses elements between two randomly chosen cut points (classic 2-opt inversion). |
| **`SwapMutation`** | Banzhaf (1990) | Swaps two elements at randomly selected indices. |
| **`ScrambleMutation`** | Syswerda (1991) | Randomly permutes/shuffles a contiguous subsequence. |
| **`InsertionMutation`** | Michalewicz (1992), Fogel (1990) | Removes an element at index $i$ and inserts it at index $j$. |
| **`TranspositionMutation`** | Goldberg (1989), Bäck (1996) | Selects two disjoint blocks/subsequences and exchanges their positions. |
| **`DuplicationMutation`** | Holland (1975), Koza (1992) | Duplicates a random subsequence into another location in the genome. |
| **`DeletionMutation`** | Harvey (1992), Koza (1992) | Deletes a random subsequence while preserving a minimum genome length. |

---

## 2. Real-Valued & Continuous Mutation

| Strategy | Literature Reference | Mechanism / Key Property |
| :--- | :--- | :--- |
| **`GaussianMutation`** | Rechenberg (1973), Schwefel (1981) | Adds zero-mean Gaussian noise $N(0, \sigma^2)$ per gene with mutation rate $p_m$. |
| **`UniformRealMutation`** | Michalewicz (1992) | Perturbs gene values uniformly in $[x_i - \delta, x_i + \delta]$ within optional domain bounds. |
| **`PolynomialMutation`** | Deb & Agrawal (1995), Deb (2001 - NSGA-II) | Standard continuous mutation in multi-objective EAs; uses polynomial probability distribution with distribution index $\eta_m$. |
| **`CauchyMutation`** | Yao & Liu (1996) | Heavy-tailed Cauchy noise enabling occasional long jumps to escape deep local optima (Fast EP). |

---

## 3. Binary & Discrete Mutation

| Strategy | Literature Reference | Mechanism / Key Property |
| :--- | :--- | :--- |
| **`BitFlipMutation`** | Holland (1975), Goldberg (1989) | Flips boolean/bit values across the genome with probability $p_m$ (default $1/L$). |
| **`BoundaryMutation`** | Michalewicz (1992) | Resets genes to their lower or upper domain boundary with equal probability. |

---

## 4. Structure-Agnostic & Custom Mutators

| Component | Description |
| :--- | :--- |
| **`UniformMutation`** | Recursively maps a custom `mutation_fn` across flat, nested, or segmented genomes. |
| **`gaussian_noise_mutator`** | Helper generating Gaussian noise for numeric values, NumPy arrays, and PyTorch tensors. |
| **`bit_flip_mutator`** | Helper generating bit-flip functions for boolean attributes. |
| **`attribute_mutator`** | Helper mapping specific mutators to named object attributes. |

---

## 5. Usage Examples

### Sequence Mutation
```python
from neutral_selection import Genome, mutate, InversionMutation, SwapMutation

genome = Genome([1, 2, 3, 4, 5, 6])
mutated = mutate(genome, InversionMutation())
```

### Real-Valued Mutation
```python
from neutral_selection import Genome, mutate, PolynomialMutation, GaussianMutation

genome = Genome([0.5, -0.2, 0.8])
poly_mut = PolynomialMutation(bounds=(-1.0, 1.0), eta_m=20.0)
mutated = mutate(genome, poly_mut)
```

---

## References

1. Banzhaf, W. (1990). *The "molecular" processor—new concepts for hardware implementation of combinatorial algorithms*. International Journal of Circuit Theory and Applications, 18(2), 113–128.
2. Bäck, T. (1996). *Evolutionary Algorithms in Theory and Practice*. Oxford University Press.
3. Deb, K., & Agrawal, R. B. (1995). *Simulated binary crossover for continuous search space*. Complex Systems, 9(2), 115–134.
4. Deb, K. (2001). *Multi-Objective Optimization using Evolutionary Algorithms*. John Wiley & Sons.
5. Fogel, D. B. (1988). *An evolutionary approach to the traveling salesman problem*. Biological Cybernetics, 60(2), 139–144.
6. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
7. Harvey, I. (1992). *Species adaptation genetic algorithms: A basis for a continuing SAGA*. Towards a practice of autonomous systems, 346–354.
8. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
9. Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.
10. Michalewicz, Z. (1992). *Genetic Algorithms + Data Structures = Evolution Programs*. Springer-Verlag.
11. Rechenberg, I. (1973). *Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution*. Frommann-Holzboog.
12. Schwefel, H. P. (1981). *Numerical Optimization of Computer Models*. John Wiley & Sons.
13. Syswerda, G. (1991). *Schedule optimization using genetic algorithms*. Handbook of Genetic Algorithms, 332–349.
14. Yao, X., & Liu, Y. (1996). *Fast evolutionary programming*. Evolutionary Programming V, 451–460.
