# Recombination Strategies (`neutral_selection.reproduction.recombination`)

This package implements established recombination (crossover) strategies for evolutionary algorithms, supporting sequence/positional genomes, permutation/ordering representations, real-valued continuous vectors, and nested/hierarchical structures.

All recombination strategies inherit from `RecombinationStrategy` and can be invoked directly as callables `strategy(parent_a, parent_b)` or via the functional helper `recombine(parent_a, parent_b, strategy)`.

---

## 1. Positional & Structural Crossover

Designed for fixed-length sequence encodings (binary, integer, discrete categories) where gene position corresponds to a specific locus.

| Strategy | Key Parameters & Defaults | Supported Representation | Mechanism & Guarantees |
| :--- | :--- | :--- | :--- |
| **`OnePointCrossover`** | `cut_point: Optional[int] = None` | Any sequence | Slices parents at a single cut point and exchanges tails. If `cut_point` is `None`, uniformly samples in $[1, L-1]$. |
| **`TwoPointCrossover`** | `cut_points: Optional[Tuple[int, int]] = None` | Any sequence | Slices parents at two cut points and exchanges the middle segment. Reduces endpoint bias relative to 1-point crossover. |
| **`NPointCrossover`** | `cut_points: list[int]` | Any sequence | Slices and alternates parent segments at explicit, deterministic cut points. |
| **`RandomNPointCrossover`** | `num_cut_points: int = 1` | Any sequence | Slices parents at $N$ randomly sampled, unique cut points. |
| **`UniformCrossover`** | `swap_prob: float = 0.5` | Any sequence | For each locus independently, children inherit from Parent A with probability $p_x$ and Parent B otherwise. Eliminates positional bias. |
| **`ShuffleCrossover`** | `crossover_strategy: Optional[RecombinationStrategy] = None` | Any sequence | Applies an identical random shuffle to both parents, performs the inner crossover (default `OnePointCrossover`), then un-shuffles. |

---

## 2. Permutation & Order-Preserving Crossover

Designed for combinatorial and ordering problems (e.g., Traveling Salesperson, scheduling, routing) where each gene must appear exactly once without duplicate conflicts or omissions.

| Strategy | Key Parameters & Defaults | Invariant / Guarantee | Mechanism |
| :--- | :--- | :--- | :--- |
| **`OrderCrossover` (OX1)** | *(None)* | Preserves relative order | Copies a contiguous slice from Parent 1 and fills remaining positions with elements from Parent 2 starting after the second cut point in circular order. |
| **`PartiallyMatchedCrossover` (PMX)** | *(None)* | Preserves absolute positions | Copies a contiguous slice from Parent 1 and constructs a bijective index-mapping between parents to resolve conflicts outside the slice. |
| **`CycleCrossover` (CX)** | *(None)* | Preserves exact parent positions | Decomposes parent permutations into disjoint permutation cycles; alternating cycles inherit their exact index positions from either Parent 1 or Parent 2. |

---

## 3. Real-Valued & Continuous Arithmetic Crossover

Designed for floating-point and continuous parameter spaces ($x \in \mathbb{R}^n$).

| Strategy | Key Parameters & Defaults | Bounds Handling | Mechanism & Behavior |
| :--- | :--- | :--- | :--- |
| **`ArithmeticCrossover`** | `alpha: float = 0.5` | Convex combination | Linearly blends parent vectors: $\text{child}_1 = \alpha P_1 + (1-\alpha) P_2$ and $\text{child}_2 = (1-\alpha) P_1 + \alpha P_2$. |
| **`BlendCrossover` (BLX-$\alpha$)** | `alpha: float = 0.5`, `bounds: Optional[Tuple[float, float]] = None` | Optional $[low, high]$ clamping | Samples each gene uniformly from $[c_{\min} - \alpha d, c_{\max} + \alpha d]$ where $d = |x_{1,i} - x_{2,i}|$. Allows exploration outside parent bounds. |
| **`SimulatedBinaryCrossover` (SBX)** | `eta_c: float = 2.0`, `swap_prob: float = 0.5`, `bounds: Optional[Tuple[float, float]] = None` | Optional $[low, high]$ clamping | Self-adaptive continuous analogue of single-point binary crossover parameterized by distribution index $\eta_c$. Standard operator in NSGA-II. |

---

## 4. Structure-Agnostic Crossover

| Component | Key Parameters & Defaults | Description |
| :--- | :--- | :--- |
| **`ElementwiseCrossover`** | `blend_fn: Callable[[Any, Any], Tuple[Any, Any]]` | Recursively traverses hierarchical, nested, or segment genome structures and applies a custom element-level blend function to corresponding leaves. |

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

### Real-Valued Continuous Crossover (SBX & BLX-$\alpha$)
```python
from neutral_selection import Genome, recombine, SimulatedBinaryCrossover, BlendCrossover

parent_a = Genome([1.0, 2.5, -0.5])
parent_b = Genome([3.0, 1.5, 0.5])

child_a, child_b = recombine(
    parent_a,
    parent_b,
    SimulatedBinaryCrossover(eta_c=2.0, bounds=(-5.0, 5.0)),
)
```

---

## Further Reading & General References

For comprehensive theoretical derivations, performance analyses, and schema theorems across these operators:

* **Eiben, A. E., & Smith, J. E. (2015).** *Introduction to Evolutionary Computing* (2nd ed.). Springer Natural Computing Series. *(Comprehensive overview of representation, crossover, and mutation operators).*
* **Deb, K. (2001).** *Multi-Objective Optimization using Evolutionary Algorithms*. John Wiley & Sons. *(Detailed treatment of real-coded operators, SBX, and continuous search).*
* **Rothlauf, F. (2006).** *Representations for Genetic and Evolutionary Algorithms* (2nd ed.). Springer. *(In-depth analysis of permutation representations and ordering operators).*
* **Luke, S. (2013).** *Essentials of Metaheuristics* (2nd ed.). Lulu / George Mason University. *(Freely available reference on metaheuristic algorithms and genetic operators).*
