from __future__ import annotations

import unittest
from neutral_selection.representation.genome import Genome
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.replacement import (
    ReplacementStrategy,
    replace,
    GenerationalReplacement,
    PlusReplacement,
    CommaReplacement,
    SteadyStateReplacement,
)


def _make_individual(val: int, fitness: float | None = None) -> Individual:
    ind = Individual(Genome([val]))
    ind.fitness = fitness
    return ind


class TestReplacementStrategies(unittest.TestCase):

    def test_generational_replacement_with_elitism(self) -> None:
        parents = [_make_individual(i, float(i * 10)) for i in range(4)]      # [0, 10, 20, 30]
        offspring = [_make_individual(10 + i, float(100 + i)) for i in range(4)] # [100, 101, 102, 103]

        strat = GenerationalReplacement(num_elites=1)
        survivors = replace(parents, offspring, strat, target_size=4)

        self.assertEqual(len(survivors), 4)
        # 1 elite parent (fitness 30.0) + 3 top offspring (103, 102, 101)
        self.assertEqual(survivors[0].fitness, 30.0)
        self.assertEqual(survivors[1].fitness, 103.0)
        self.assertEqual(survivors[2].fitness, 102.0)
        self.assertEqual(survivors[3].fitness, 101.0)

    def test_generational_replacement_with_elite_ratio(self) -> None:
        parents = [_make_individual(i, float(i * 10)) for i in range(4)]      # [0, 10, 20, 30]
        offspring = [_make_individual(10 + i, float(100 + i)) for i in range(4)] # [100, 101, 102, 103]

        strat = GenerationalReplacement(elite_ratio=0.5)
        survivors = replace(parents, offspring, strat, target_size=4)

        self.assertEqual(len(survivors), 4)
        # 2 elite parents (30.0, 20.0) + 2 top offspring (103, 102)
        self.assertEqual(survivors[0].fitness, 30.0)
        self.assertEqual(survivors[1].fitness, 20.0)
        self.assertEqual(survivors[2].fitness, 103.0)
        self.assertEqual(survivors[3].fitness, 102.0)

    def test_plus_replacement(self) -> None:
        # (mu + lambda)
        parents = [_make_individual(1, 10.0), _make_individual(2, 80.0)]
        offspring = [_make_individual(3, 50.0), _make_individual(4, 90.0)]

        strat = PlusReplacement()
        survivors = strat.replace(parents, offspring, target_size=2)

        self.assertEqual(len(survivors), 2)
        self.assertEqual(survivors[0].fitness, 90.0)
        self.assertEqual(survivors[1].fitness, 80.0)

    def test_comma_replacement(self) -> None:
        # (mu, lambda)
        parents = [_make_individual(1, 100.0), _make_individual(2, 100.0)]
        offspring = [_make_individual(3, 30.0), _make_individual(4, 50.0), _make_individual(5, 40.0)]

        strat = CommaReplacement()
        survivors = strat.replace(parents, offspring, target_size=2)

        self.assertEqual(len(survivors), 2)
        self.assertEqual(survivors[0].fitness, 50.0)
        self.assertEqual(survivors[1].fitness, 40.0)

        # Error if offspring < target_size
        with self.assertRaises(ValueError):
            strat.replace(parents, offspring, target_size=5)

    def test_steady_state_replacement(self) -> None:
        parents = [
            _make_individual(1, 10.0),
            _make_individual(2, 40.0),
            _make_individual(3, 30.0),
            _make_individual(4, 20.0),
        ]
        offspring = [_make_individual(5, 50.0), _make_individual(6, 60.0)]

        strat = SteadyStateReplacement()
        survivors = strat.replace(parents, offspring, target_size=4)

        self.assertEqual(len(survivors), 4)
        # Best 2 parents (40.0, 30.0) + 2 offspring (50.0, 60.0)
        self.assertEqual(survivors[0].fitness, 40.0)
        self.assertEqual(survivors[1].fitness, 30.0)
        self.assertEqual(survivors[2].fitness, 50.0)
        self.assertEqual(survivors[3].fitness, 60.0)

    def test_replace_fail_fast_validations(self) -> None:
        parents = [_make_individual(1, 10.0)]
        offspring = [_make_individual(2, 20.0)]

        with self.assertRaises(ValueError):
            replace(parents, offspring, None)  # type: ignore

        with self.assertRaises(TypeError):
            replace(parents, offspring, "not-a-strategy")  # type: ignore

        with self.assertRaises(ValueError):
            GenerationalReplacement(num_elites=-1)

        with self.assertRaises(ValueError):
            GenerationalReplacement(elite_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
