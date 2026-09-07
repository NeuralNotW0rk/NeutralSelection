# Mutation Strategies (`neutral_selection.reproduction.mutation`)

This package implements established mutation strategies for evolutionary algorithms, supporting sequence/permutation representations, real-valued/continuous vectors, binary/boolean encodings, and nested/hierarchical genome structures.

All mutation strategies inherit from `MutationStrategy` and can be applied directly as callables `strategy(genome)` or via the functional helper `mutate(target, strategy)`.

---

## 1. Sequence & Permutation Mutation

Designed for order-based and combinatorial encodings (e.g. TSP, scheduling, sequence alignment). Preserves the multiset of genome elements without altering gene counts (except Duplication/Deletion).

| Strategy | Key Parameters & Defaults | Length Invariant | Mechanism & Behavior |
| :--- | :--- | :--- | :--- |
| **`InversionMutation`** | *(None)* | Preserved | Reverses the order of elements between two randomly sampled indices (classic 2-opt inversion). |
| **`SwapMutation`** | *(None)* | Preserved | Exchanges the elements at two randomly selected index locations. |
| **`ScrambleMutation`** | *(None)* | Preserved | Randomly shuffles/permutes all elements within a contiguous subsequence. |
| **`InsertionMutation`** | *(None)* | Preserved | Removes an element from index $i$ and inserts it at target index $j$ (displacement). |
| **`TranspositionMutation`** | `block_size: Optional[int] = None` | Preserved | Selects two non-overlapping contiguous slices and swaps their positions. |
| **`DuplicationMutation`** | `max_length: Optional[int] = None` | Variable (increases) | Duplicates a random subsequence and inserts it at another location in the genome. |
| **`DeletionMutation`** | `min_length: int = 1` | Variable (decreases) | Deletes a random subsequence while preserving a minimum genome length. |

---

## 2. Real-Valued & Continuous Mutation

Designed for continuous floating-point vectors ($x \in \mathbb{R}^n$).

| Strategy | Key Parameters & Defaults | Bounds Handling | Perturbation Distribution |
| :--- | :--- | :--- | :--- |
| **`GaussianMutation`** | `sigma: float = 1.0`, `mutation_rate: float = 1.0`, `bounds: Optional[Tuple[float, float]] = None` | Optional $[low, high]$ clamping | Adds zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ per gene with probability $p_m$. |
| **`UniformRealMutation`** | `delta: float = 1.0`, `mutation_rate: float = 1.0`, `bounds: Optional[Tuple[float, float]] = None` | Optional $[low, high]$ clamping | Perturbs genes by adding uniform noise $\mathcal{U}(-\delta, \delta)$ with probability $p_m$. |
| **`PolynomialMutation`** | `bounds: Tuple[float, float]`, `eta_m: float = 20.0`, `mutation_rate: Optional[float] = None` (default $1/L$) | Required $[low, high]$ bounds | Applies polynomial distribution perturbation parameterized by distribution index $\eta_m$. Standard in NSGA-II. |
| **`CauchyMutation`** | `scale: float = 1.0`, `mutation_rate: float = 1.0`, `bounds: Optional[Tuple[float, float]] = None` | Optional $[low, high]$ clamping | Heavy-tailed Cauchy noise enabling occasional long jumps to escape local optima (Fast EP). |

---

## 3. Binary & Discrete Mutation

| Strategy | Key Parameters & Defaults | Applicable Types | Mechanism |
| :--- | :--- | :--- | :--- |
| **`BitFlipMutation`** | `mutation_rate: Optional[float] = None` (default $1/L$) | `bool`, `0`/`1` int | Flips binary or boolean states with probability $p_m$. |
| **`BoundaryMutation`** | `bounds: Tuple[float, float]`, `mutation_rate: float = 1.0` | Numeric floats/ints | Resets selected genes to either their lower or upper domain boundary with equal probability. |

---

## 4. Structure-Agnostic & Custom Mutators

| Component | Description |
| :--- | :--- |
| **`UniformMutation`** | Recursively maps a custom `mutation_fn` across flat, nested, or segmented genomes. |
| **`gaussian_noise_mutator`** | Helper generating Gaussian noise functions for numeric values, NumPy arrays, and PyTorch tensors. |
| **`bit_flip_mutator`** | Helper generating bit-flip functions for boolean attributes. |
| **`attribute_mutator`** | Helper mapping specific mutator functions to named object attributes. |

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

## Further Reading & General References

For foundational theory, mutation step-size control, and self-adaptation across these operators:

* **Eiben, A. E., & Smith, J. E. (2015).** *Introduction to Evolutionary Computing* (2nd ed.). Springer Natural Computing Series. *(Covers bit-flip, real-valued distributions, and permutation mutation operators).*
* **Deb, K. (2001).** *Multi-Objective Optimization using Evolutionary Algorithms*. John Wiley & Sons. *(Mathematical derivation of polynomial mutation and bounded continuous perturbation).*
* **Kramer, O. (2017).** *Genetic Algorithm Essentials* (Studies in Computational Intelligence, Vol. 679). Springer. *(Concise overview of mutation operators and variance parameters).*
* **Luke, S. (2013).** *Essentials of Metaheuristics* (2nd ed.). Lulu / George Mason University. *(Algorithmic details and parameter guidance for continuous and discrete mutators).*
