# Variation (`neutral_selection.variation`)

The `variation` package implements operators responsible for creating genetic diversity in candidate populations.

---

## Granularity

Every operator that handles nested genomes takes a `granularity` argument that sets what counts as one leaf. Operators flatten the genome to a strand of leaves at that scale, operate on the strand, and rebuild the original structure.

| `granularity` | One leaf is | Example on a genome of LoRA genes |
|---|---|---|
| `"atom"` (default) | A single scalar: descends into genes, dataclasses and tensors | One weight |
| `"tensor"` | A whole tensor; everything above tensors is still expanded | One `lora_down` or `lora_up` matrix |
| `"gene"` | Any item that is not itself a `Genome` or `Segment`, at whatever depth it sits | One `LoRAGene` |
| `int` | Whatever sits at that depth below the root | `1`: each top-level item |

`atomic_types` still applies on top of any granularity. `max_depth=k` is the older spelling of `granularity=k`.

Positional and callable operators work at every granularity. Numeric operators (Gaussian, polynomial, arithmetic, SBX and similar) need numeric leaves, so they are normally used at `"atom"`.

---

## Parent selection (`neutral_selection.variation.selection`)

Parent selection operators choose individuals from a population for reproduction based on their relative fitness values.

| Operator | Mechanics & Parameters |
|---|---|
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

## Recombination (`neutral_selection.variation.recombination`)

Crossover strategies combine genetic material from two parent chromosomes.

| Operator | Type | Description |
|---|---|---|
| `OnePointCrossover` | Positional | Single random or fixed split point. |
| `TwoPointCrossover` | Positional | Swaps slice between two cut points. |
| `NPointCrossover` / `RandomNPointCrossover` | Positional | Arbitrary $N$-point crossover. |
| `UniformCrossover` | Positional | Leaf-by-leaf independent binomial swap ($p_{\text{swap}}$). |
| `OrderCrossover` (OX1) | Permutation | Order-preserving permutation crossover. |
| `PartiallyMatchedCrossover` (PMX) | Permutation | Cycle-mapping permutation crossover (Goldberg & Lingle, 1985). |
| `CycleCrossover` (CX) | Permutation | Preserves exact absolute positions from parents. |
| `ArithmeticCrossover` | Real-Valued | Convex combination: $\mathbf{c}_1 = \alpha \mathbf{p}_1 + (1-\alpha)\mathbf{p}_2$. |
| `BlendCrossover` (BLX-$\alpha$) | Real-Valued | Interval expansion: $[x_{\min} - \alpha I, x_{\max} + \alpha I]$ (Eshelman & Schaffer, 1993). |
| `SimulatedBinaryCrossover` (SBX) | Real-Valued | Self-adaptive search power with distribution index $\eta_c$ (Deb & Agrawal, 1995). |
| `FunctionCrossover` | Callable | Combines each aligned pair of leaves with a custom `crossover_fn(a, b, blend_factor)`; returns two children, or one with `num_children=1` (for symmetric callables). |
| `ElementwiseCrossover` | Callable (deprecated) | `FunctionCrossover` at `granularity="gene"` with `num_children=1`. |

---

## Mutation (`neutral_selection.variation.mutation`)

Mutation strategies introduce novel variations into individual genomes.

| Operator | Domain | Parameters / Notes |
|---|---|---|
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
| `FunctionMutation` | Callable | Applies a custom `mutation_fn` to each leaf with probability `mutation_rate`. |
| `UniformMutation` | Callable (deprecated) | `FunctionMutation` at `granularity="gene"`. |
