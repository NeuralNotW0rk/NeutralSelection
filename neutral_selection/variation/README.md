# Variation (`neutral_selection.variation`)

The `variation` package implements operators responsible for creating genetic diversity in candidate populations.

---

## 1. Parent Selection (`neutral_selection.variation.selection`)

Parent selection operators choose individuals from a population for reproduction based on their relative fitness values.

| Operator | Mechanics & Parameters |
| :--- | :--- |
| `TournamentSelection` | Sub-samples $k$ individuals and selects the best with probability $p$. Parameters: `tournament_size`, `winner_prob`, `minimize`. |
| `RouletteWheelSelection` | Fitness-proportionate sampling ($p_i \propto f_i$). Parameters: `fitness_offset`, `minimize`. |
| `StochasticUniversalSamplingSelection` (SUS) | Zero-bias, minimal spread proportionate sampling with equidistant pointers (Baker, 1987). |
| `LinearRankSelection` | Ranks individuals linearly to normalize selection pressure. Parameters: `selection_pressure` ($1.0 \le \eta \le 2.0$). |
| `ExponentialRankSelection` | Non-linear ranking with base $c \in (0, 1)$. |
| `TruncationSelection` | Selects top $k$ or top ratio $\rho$ best individuals. |
| `ElitistSelection` | Deterministically selects the absolute top-fittest individuals. |
| `RandomSelection` | Uniform random selection ignoring fitness (drift benchmark). |
| `BoltzmannSelection` | Simulated annealing-style temperature-scaled selection ($p_i \propto e^{f_i / T}$). |

---

## 2. Recombination (`neutral_selection.variation.recombination`)

Crossover strategies combine genetic material from two parent chromosomes.

| Operator | Type | Description |
| :--- | :--- | :--- |
| `OnePointCrossover` | Positional | Single random or fixed split point. |
| `TwoPointCrossover` | Positional | Swaps slice between two cut points. |
| `NPointCrossover` / `RandomNPointCrossover` | Positional | Arbitrary $N$-point crossover. |
| `UniformCrossover` | Positional | Gene-by-gene independent binomial swap ($p_{\text{swap}}$). |
| `OrderCrossover` (OX1) | Permutation | Order-preserving permutation crossover. |
| `PartiallyMatchedCrossover` (PMX) | Permutation | Cycle-mapping permutation crossover (Goldberg & Lingle, 1985). |
| `CycleCrossover` (CX) | Permutation | Preserves exact absolute positions from parents. |
| `ArithmeticCrossover` | Real-Valued | Convex combination: $\mathbf{c}_1 = \alpha \mathbf{p}_1 + (1-\alpha)\mathbf{p}_2$. |
| `BlendCrossover` (BLX-$\alpha$) | Real-Valued | Interval expansion: $[x_{\min} - \alpha I, x_{\max} + \alpha I]$ (Eshelman & Schaffer, 1993). |
| `SimulatedBinaryCrossover` (SBX) | Real-Valued | Self-adaptive search power with distribution index $\eta_c$ (Deb & Agrawal, 1995). |
| `ElementwiseCrossover` | Heterogeneous | Maps custom crossover strategies across named chromosome `Segment` elements. |

---

## 3. Mutation (`neutral_selection.variation.mutation`)

Mutation strategies introduce novel variations into individual genomes.

| Operator | Domain | Parameters / Notes |
| :--- | :--- | :--- |
| `GaussianMutation` | Real-Valued | Adds zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ with optional bounds clamping. |
| `UniformRealMutation` | Real-Valued | Adds uniform noise in $[-\delta, \delta]$. |
| `PolynomialMutation` | Real-Valued | NSGA-II polynomial distribution perturbation ($\eta_m$). |
| `CauchyMutation` | Real-Valued | Fast Evolutionary Programming heavy-tailed Cauchy noise ($\gamma$). |
| `BitFlipMutation` | Binary / Boolean | Flips boolean/bit values with probability `flip_prob`. |
| `BoundaryMutation` | Bounded | Mutates genes randomly to their lower or upper bounds. |
| `InversionMutation` (2-opt) | Permutation | Reverses an interior chromosome slice. |
| `SwapMutation` | Permutation | Swaps positions of two randomly chosen genes. |
| `ScrambleMutation` | Permutation | Randomly shuffles genes within a sub-range. |
| `InsertionMutation` | Sequence | Removes a gene and re-inserts it at a new position. |
| `TranspositionMutation` | Sequence | Swaps two contiguous sub-sequences. |
| `DuplicationMutation` | Sequence | Duplicates a gene segment (variable-length genomes). |
| `DeletionMutation` | Sequence | Deletes a gene segment (variable-length genomes). |
| `UniformMutation` | Structure-Agnostic | Applies a custom callable mutator across genome elements. |
