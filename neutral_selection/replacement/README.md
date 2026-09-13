# Environmental Replacement (`neutral_selection.replacement`)

The `replacement` package implements survivor selection strategies that determine which individuals from the parent generation ($\mu$) and candidate offspring ($\lambda$) survive into the next generation.

---

## Implemented Strategies

| Strategy | Formulation | Mechanics & Parameters |
| :--- | :--- | :--- |
| `GenerationalReplacement` | $(\mu, \lambda)$ with Elitism | Next generation consists of newly evaluated offspring, retaining the top `num_elites` (or fraction `elite_ratio`) fittest parent individuals unchanged. |
| `PlusReplacement` | $(\mu + \lambda)$ | Merges parents ($\mu$) and offspring ($\lambda$) into a single pool and selects the top `target_size` fittest individuals. Common in Evolution Strategies. |
| `CommaReplacement` | $(\mu, \lambda)$ | Completely discards parents; selects survivors strictly from the offspring pool ($\lambda \ge \mu$). |
| `SteadyStateReplacement` | Steady-State | Replaces the lowest-fitness individuals in the parent population with newly generated offspring, maintaining a constant population size. |

---

## Functional Helper

```python
from neutral_selection.replacement import replace, GenerationalReplacement

survivors = replace(
    parents=parent_population,
    offspring=offspring_pool,
    strategy=GenerationalReplacement(num_elites=2),
    target_size=10,
)
```
