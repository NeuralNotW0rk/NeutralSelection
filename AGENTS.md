# NeutralSelection — Agent Development Guide (`AGENTS.md`)

This document defines the strategic vision, architectural principles, feature priorities, and coding standards for AI agents and developers working on the `NeutralSelection` codebase.

---

## 1. Strategic Vision & Ecosystem Niche

`NeutralSelection` is designed to be the **modern, strictly-typed, and composable evolutionary computation framework for Python**.

### Ecosystem Comparison
* **Why not `pymoo`?** `pymoo` is built exclusively around fixed 2D NumPy matrices (`N_individuals × N_variables`). It is rigid, difficult to extend to heterogeneous/nested structures, and forces users into a monolithic `minimize()` loop.
* **Why not `DEAP`?** `DEAP` relies on 15-year-old untyped metaprogramming (`creator.create`), global mutable state, and lacks modern Python typing.
* **Where `NeutralSelection` wins:**
  1. **Composable Functional Operators:** Every operator is a standalone callable (`strategy(a, b)`) and helper (`recombine`, `mutate`, `select`) usable inside any custom loop, interactive tool, or neural pipeline.
  2. **Heterogeneous & Segmented Genomes:** Native `Segment` and `ElementwiseCrossover` support for multi-part chromosomes (e.g., continuous parameters + permutation orders + discrete flags).
  3. **Generative AI & PyTorch Alignment:** First-class ergonomics for PyTorch tensors, NumPy arrays, latent vectors, and model attributes.
  4. **Strict Type Safety:** Modern Python 3.10+, dataclasses, and zero global state.

---

## 2. Feature Prioritization Roadmap

When planning or implementing new features, prioritize according to the following framework:

### High Priority
* **Quality Diversity (QD) & MAP-Elites:** Implementing multi-dimensional behavioral grid archiving (MAP-Elites, Novelty Search) for creative exploration and diverse solution generation.
* **Multi-Objective Optimization (MOO):** Non-dominated Pareto sorting (NSGA-II sorting and crowding distance) implemented as modular, composable selectors.
* **Batching & Vectorized Variation:** Vectorized mutation and crossover helpers for batch PyTorch tensors and NumPy arrays without losing object-level flexibility.
* **Neural & Generative Operators:** Helpers for parameter blending (weight interpolation, layer-wise crossover) and latent space exploration.

### Low Priority / Avoid
* **Rigid Matrix-Only Constraints:** Never constrain operators to rigid 2D matrices or assume all genomes are flat float vectors.
* **Monolithic Black-Box Loops:** Avoid building monolithic engines that obscure intermediate states; keep primitives callable and inspectable.
* **Global Registries:** Never use global state registries or metaprogramming to define individuals or operators.

---

## 3. Architectural Rules & Invariants

1. **Pure Functions & Immutability:**
   * Crossover and mutation operators must **never** mutate input `Genome` or `Individual` instances in place.
   * Always construct and return new genome instances using `_clone_genome_structure(original, new_items)` to preserve custom metadata and `Segment` keys.
2. **Dual-Interface Parity:**
   * All operators must be usable as standalone callable objects: `strategy = TournamentSelection(...)` $\rightarrow$ `strategy(population)`.
   * All operators must integrate with top-level functional helpers: `recombine()`, `mutate()`, `select()`, and `select_survivors()`.
3. **Fail-Fast Validation:**
   * Validate all hyperparameters (`mutation_rate`, `bounds`, `tournament_size`, `eta_c`, `eta_m`, etc.) during `__init__`.
   * Validate input types and dimension/length compatibility immediately during `__call__` / `select()`.
   * Raise explicit, descriptive `TypeError` or `ValueError` exceptions immediately rather than falling back silently.
4. **No Global State:**
   * Individual strategies, populations, and configurations must remain self-contained, stateless, and thread-safe.

---

## 4. Coding Standards

* **Python Version:** Target Python 3.10+.
* **Strict Type Hints:** 
  * Every function, method, and helper must have explicit parameter type annotations and an explicit return type (e.g., `-> None`, `-> tuple[Genome, Genome]`, `-> list[Individual]`).
  * Avoid `Any`. Prefer specific generics (`Sequence[T]`, `Callable[[Genome], Genome]`, `Optional[Tuple[float, float]]`) or `unknown` where appropriate.
* **Line Endings:** Always use LF (`\n`) line endings.
* **Testing Discipline:**
  * Every new operator, strategy, or mutator must be accompanied by unit tests in `tests/`.
  * Tests must verify:
    1. Correct output types and structural cloning.
    2. Mathematical invariants (e.g. permutation uniqueness, bounds clamping, probability sums).
    3. Type errors and value error boundaries (invalid rates, unequal lengths, out-of-bounds inputs).
