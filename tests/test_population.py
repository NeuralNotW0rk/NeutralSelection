from __future__ import annotations

import unittest
from neutral_selection.representation.genome import Genome
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.variation.selection import TournamentSelection
from neutral_selection.variation.recombination import UniformCrossover
from neutral_selection.variation.mutation import GaussianMutation
from neutral_selection.replacement import GenerationalReplacement


def _make_ind(val: int, fitness: float | None = None) -> Individual:
    ind = Individual(Genome([val]), fitness=fitness)
    return ind


class TestPopulation(unittest.TestCase):

    def test_population_initialization_and_fail_fast(self) -> None:
        inds = [_make_ind(1), _make_ind(2)]
        pop = Population(inds)
        self.assertEqual(len(pop), 2)
        self.assertEqual(pop[0].genotype[0], 1)
        self.assertEqual(pop[1].genotype[0], 2)

        # Tuple input
        pop_tuple = Population(tuple(inds))
        self.assertEqual(len(pop_tuple), 2)

        # Invalid container type
        with self.assertRaises(TypeError):
            Population("not-a-list")  # type: ignore

        # Invalid element type
        with self.assertRaises(TypeError):
            Population([_make_ind(1), "not-an-ind"])  # type: ignore

    def test_container_methods(self) -> None:
        i1 = _make_ind(10)
        i2 = _make_ind(20)
        pop = Population([i1, i2])

        # Slice
        self.assertEqual(len(pop[:1]), 1)
        self.assertEqual(pop[:1][0], i1)

        # Setitem
        i3 = _make_ind(30)
        pop[0] = i3
        self.assertEqual(pop[0], i3)

        with self.assertRaises(TypeError):
            pop[0] = "invalid"  # type: ignore

        # Contains
        self.assertIn(i3, pop)
        self.assertIn(i2, pop)
        self.assertNotIn(i1, pop)

        # Iter
        itered = list(pop)
        self.assertEqual(itered, [i3, i2])

        # Append & Extend
        i4 = _make_ind(40)
        pop.append(i4)
        self.assertEqual(len(pop), 3)
        self.assertEqual(pop[2], i4)

        with self.assertRaises(TypeError):
            pop.append("invalid")  # type: ignore

        i5 = _make_ind(50)
        pop.extend([i5])
        self.assertEqual(len(pop), 4)

    def test_fitness_statistics_and_properties(self) -> None:
        # Empty / unrated
        empty_pop = Population([])
        self.assertIsNone(empty_pop.best_individual)
        self.assertIsNone(empty_pop.worst_individual)
        self.assertIsNone(empty_pop.mean_fitness)
        self.assertIsNone(empty_pop.max_fitness)
        self.assertIsNone(empty_pop.min_fitness)
        self.assertEqual(empty_pop.fitnesses, [])

        unrated_pop = Population([_make_ind(1), _make_ind(2)])
        self.assertIsNone(unrated_pop.best_individual)
        self.assertIsNone(unrated_pop.worst_individual)
        self.assertIsNone(unrated_pop.mean_fitness)
        self.assertEqual(unrated_pop.fitnesses, [None, None])

        # Rated population
        i1 = _make_ind(1, fitness=10.0)
        i2 = _make_ind(2, fitness=30.0)
        i3 = _make_ind(3, fitness=20.0)
        pop = Population([i1, i2, i3])

        self.assertEqual(pop.best_individual, i2)
        self.assertEqual(pop.worst_individual, i1)
        self.assertEqual(pop.max_fitness, 30.0)
        self.assertEqual(pop.min_fitness, 10.0)
        self.assertAlmostEqual(pop.mean_fitness, 20.0)
        self.assertEqual(pop.fitnesses, [10.0, 30.0, 20.0])

    def test_population_coordination_methods(self) -> None:
        i1 = _make_ind(1, fitness=10.0)
        i2 = _make_ind(2, fitness=30.0)
        i3 = _make_ind(3, fitness=20.0)
        pop = Population([i1, i2, i3])

        # select()
        sel = pop.select(TournamentSelection(tournament_size=2), k=2)
        self.assertEqual(len(sel), 2)
        self.assertIsInstance(sel[0], Individual)

        # replace()
        offspring = Population([_make_ind(4, fitness=40.0), _make_ind(5, fitness=5.0)])
        surv_pop = pop.replace(offspring, GenerationalReplacement(num_elites=1), target_size=2)
        self.assertIsInstance(surv_pop, Population)
        self.assertEqual(len(surv_pop), 2)
        # Elite i2 (30.0) + best offspring (40.0)
        self.assertEqual(surv_pop[0].fitness, 30.0)
        self.assertEqual(surv_pop[1].fitness, 40.0)

        # step()
        stepped_pop = pop.step(
            selection_strategy=TournamentSelection(tournament_size=2),
            crossover_strategy=UniformCrossover(swap_prob=0.5),
            mutation_strategy=GaussianMutation(sigma=0.1),
            offspring_count=4,
            elitism=1,
        )
        self.assertIsInstance(stepped_pop, Population)
        self.assertEqual(len(stepped_pop), 4)
        # Elite is preserved
        self.assertEqual(stepped_pop[0].fitness, 30.0)

    def test_population_clone(self) -> None:
        i1 = _make_ind(1, fitness=10.0)
        i2 = _make_ind(2, fitness=20.0)
        pop = Population([i1, i2])

        cloned_pop = pop.clone(deep=True)
        self.assertEqual(len(cloned_pop), 2)
        self.assertEqual(cloned_pop[0].fitness, 10.0)
        self.assertIsNot(cloned_pop[0], i1)
        self.assertIsNot(cloned_pop[0].genotype, i1.genotype)


if __name__ == "__main__":
    unittest.main()
