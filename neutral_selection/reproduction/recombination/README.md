# Recombination Strategies (`neutral_selection.reproduction.recombination`)

This package implements established recombination (crossover) strategies from the evolutionary computation literature. Recombination combines genetic material from two parent genomes to produce offspring containing combinations of parental traits.

All recombination strategies inherit from `RecombinationStrategy` and can be invoked directly as callables `strategy(parent_a, parent_b)` or via the functional helper `recombine(parent_a, parent_b, strategy)`.

---

## 1. Positional & Structural Crossover

| Strategy | Literature Reference | Mechanism / Key Property |
| :--- | :--- | :--- |
| **`OnePointCrossover`** | Holland (1975) | Slices parents at a single cut point and exchanges tails. |
| **`TwoPointCrossover`** | De Jong (1975), Holland (1975) | Slices parents at two cut points and exchanges the middle segment. |
| **`NPointCrossover`** | De Jong (1975) | Slices and alternates parent segments at deterministic cut points. |
| **`RandomNPointCrossover`** | Eshelman et al. (1989) | Slices parents at $N$ randomly sampled cut points. |
| **`UniformCrossover`** | Syswerda (1989) | For each gene, children inherit from Parent A with probability $p_x$ (default 0.5) and Parent B otherwise. |
| **`ShuffleCrossover`** | Eshelman, Caruana & Schaffer (1989) | Shuffles gene positions identically in both parents, applies crossover, and un-shuffles back to reduce positional bias. |

---

## 2. Permutation & Order-Preserving Crossover

| Strategy | Literature Reference | Mechanism / Key Property |
| :--- | :--- | :--- |
| **`OrderCrossover` (OX1)** | Davis (1985) | Preserves relative order; copies a slice from Parent 1 and fills remainder from Parent 2 preserving circular order. Standard for TSP & scheduling. |
| **`PartiallyMatchedCrossover` (PMX)** | Goldberg & Lingle (1985) | Standard permutation crossover; copies a slice from Parent 1 and uses lookup mappings to prevent duplicate conflicts. |
| **`CycleCrossover` (CX)** | Oliver, Smith & Holland (1987) | Discovers disjoint cycles between parents and preserves exact parent positions for each cycle. |

---

## 3. Real-Valued & Continuous Arithmetic Crossover

| Strategy | Literature Reference | Mechanism / Key Property |
| :--- | :--- | :--- |
| **`ArithmeticCrossover`** | Michalewicz (1992) | Linear combination of parent vectors: $\text{child}_1 = \alpha P_1 + (1-\alpha) P_2$ and $\text{child}_2 = (1-\alpha) P_1 + \alpha P_2$. |
| **`BlendCrossover` (BLX-$\alpha$)** | Eshelman & Schaffer (1993) | Samples gene values uniformly from $[c_{\min} - \alpha \cdot d, c_{\max} + \alpha \cdot d]$ where $d = |x_{1,i} - x_{2,i}|$. |
| **`SimulatedBinaryCrossover` (SBX)** | Deb & Agrawal (1995), Deb & Beyer (2001) | Continuous analogue of single-point binary crossover parameterized by distribution index $\eta_c$. Standard in NSGA-II. |

---

## 4. Structure-Agnostic Crossover

| Strategy | Mechanism / Key Property |
| :--- | :--- |
| **`ElementwiseCrossover`** | Recursively blends items in hierarchical or segment genomes using a custom blend function. |

---

## 5. Usage Examples

### Sequence & Positional Crossover
```python
from neutral_selection import Genome, recombine, TwoPointCrossover, UniformCrossover

parent_a = Genome([1, 2, 3, 4, 5])
parent_b = Genome([10, 20, 30, 40, 50])

child_a, child_b = recombine(parent_a, parent_b, TwoPointCrossover())
```

### Permutation Crossover (PMX & OX1)
```python
from neutral_selection import Genome, recombine, PartiallyMatchedCrossover, OrderCrossover

parent_a = Genome([1, 2, 3, 4, 5, 6, 7, 8])
parent_b = Genome([8, 7, 6, 5, 4, 3, 2, 1])

child_a, child_b = recombine(parent_a, parent_b, PartiallyMatchedCrossover())
```

### Real-Valued Continuous Crossover (SBX & BLX-alpha)
```python
from neutral_selection import Genome, recombine, SimulatedBinaryCrossover, BlendCrossover

parent_a = Genome([1.0, 2.5, -0.5])
parent_b = Genome([3.0, 1.5, 0.5])

child_a, child_b = recombine(parent_a, parent_b, SimulatedBinaryCrossover(eta_c=2.0, bounds=(-5.0, 5.0)))
```

---

## References

1. Davis, L. (1985). *Applying adaptive algorithms to epistatic domains*. Proceedings of the 9th International Joint Conference on Artificial Intelligence, 162–164.
2. De Jong, K. A. (1975). *An analysis of the behavior of a class of genetic adaptive systems*. Doctoral dissertation, University of Michigan.
3. Deb, K., & Agrawal, R. B. (1995). *Simulated binary crossover for continuous search space*. Complex Systems, 9(2), 115–134.
4. Deb, K., & Beyer, H. G. (2001). *Self-adaptive simulated binary crossover for real-parameter optimization*. Complex Systems, 13(1), 25–40.
5. Eshelman, L. J., Caruana, R. A., & Schaffer, J. D. (1989). *Biases in the crossover landscape*. Proceedings of the 3rd International Conference on Genetic Algorithms, 10–19.
6. Eshelman, L. J., & Schaffer, J. D. (1993). *Real-coded genetic algorithms and interval-schemata*. Foundations of Genetic Algorithms, 2, 187–202.
7. Goldberg, D. E., & Lingle, R. (1985). *Alleles, loci, and the traveling salesman problem*. Proceedings of an International Conference on Genetic Algorithms and their Applications, 154–159.
8. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
9. Michalewicz, Z. (1992). *Genetic Algorithms + Data Structures = Evolution Programs*. Springer-Verlag.
10. Oliver, I. M., Smith, D. J., & Holland, J. R. (1987). *A study of permutation crossover operators on the TSP*. Proceedings of the 2nd International Conference on Genetic Algorithms, 224–230.
11. Syswerda, G. (1989). *Uniform crossover in genetic algorithms*. Proceedings of the 3rd International Conference on Genetic Algorithms, 2–9.
