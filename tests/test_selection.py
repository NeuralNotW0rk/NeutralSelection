from __future__ import annotations

import unittest
import unittest.mock as mock
from neutral_selection.representation.genome import Genome
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.variation.selection import (
    SelectionStrategy,
    select,
    TournamentSelection,
    RouletteWheelSelection,
    StochasticUniversalSamplingSelection,
    LinearRankSelection,
    ExponentialRankSelection,
    TruncationSelection,
    ElitistSelection,
    RandomSelection,
    BoltzmannSelection,
)


def _make_individual(val: int, fitness: float | None = None) -> Individual:
    ind = Individual(Genome([val]))
    ind.fitness = fitness
    return ind


class TestSelectionBaseAndHelpers(unittest.TestCase):

    def test_extract_individuals_and_population_input(self) -> None:
        inds = [_make_individual(i, float(i)) for i in range(4)]
        pop = Population(inds)

        strategy = RandomSelection()
        res_pop = strategy.select(pop, k=2)
        res_list = strategy.select(inds, k=2)

        self.assertEqual(len(res_pop), 2)
        self.assertEqual(len(res_list), 2)

    def test_select_one_and_select_pairs(self) -> None:
        inds = [_make_individual(i, float(i)) for i in range(4)]
        strategy = ElitistSelection(num_elites=4)

        one = strategy.select_one(inds)
        self.assertEqual(one.fitness, 3.0)

        pairs = strategy.select_pairs(inds, num_pairs=2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0].fitness, 3.0)
        self.assertEqual(pairs[0][1].fitness, 2.0)
        self.assertEqual(pairs[1][0].fitness, 1.0)
        self.assertEqual(pairs[1][1].fitness, 0.0)

    def test_functional_select_wrappers(self) -> None:
        inds = [_make_individual(i, float(i)) for i in range(3)]
        strategy = ElitistSelection()
        res = select(inds, strategy, k=2)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].fitness, 2.0)

        with self.assertRaises(ValueError):
            select(inds, None, k=2)  # type: ignore

        with self.assertRaises(TypeError):
            select(inds, "not-a-strategy", k=2)  # type: ignore

    def test_fail_fast_validations(self) -> None:
        # Empty population
        strategy = RandomSelection()
        with self.assertRaises(ValueError):
            strategy.select([], k=1)

        # Non-Individual element
        with self.assertRaises(TypeError):
            strategy.select(["not-an-ind"], k=1)  # type: ignore

        # Invalid k
        inds = [_make_individual(1, 1.0)]
        with self.assertRaises(TypeError):
            strategy.select(inds, k="invalid")  # type: ignore
        with self.assertRaises(ValueError):
            strategy.select(inds, k=0)

        # None fitness
        unrated_inds = [_make_individual(1, None)]
        elitist = ElitistSelection()
        with self.assertRaises(ValueError):
            elitist.select(unrated_inds, k=1)


class TestTournamentSelection(unittest.TestCase):

    def test_deterministic_tournament(self) -> None:
        inds = [
            _make_individual(1, 10.0),
            _make_individual(2, 50.0),
            _make_individual(3, 30.0),
            _make_individual(4, 90.0),
        ]
        strat = TournamentSelection(tournament_size=3, winner_prob=1.0)

        # Mock random.sample to return [ind1, ind2, ind3] (fitnesses 10, 50, 30) -> winner is ind2 (50.0)
        with mock.patch("random.sample", return_value=[inds[0], inds[1], inds[2]]):
            winner = strat.select_one(inds)
            self.assertEqual(winner.fitness, 50.0)

    def test_minimization_tournament(self) -> None:
        inds = [
            _make_individual(1, 10.0),
            _make_individual(2, 50.0),
            _make_individual(3, 30.0),
        ]
        strat = TournamentSelection(tournament_size=3, winner_prob=1.0, minimize=True)
        with mock.patch("random.sample", return_value=list(inds)):
            winner = strat.select_one(inds)
            self.assertEqual(winner.fitness, 10.0)

    def test_probabilistic_tournament(self) -> None:
        inds = [
            _make_individual(1, 10.0),
            _make_individual(2, 50.0),
        ]
        strat = TournamentSelection(tournament_size=2, winner_prob=0.8)

        # If random.random() < 0.8 -> best wins (50.0)
        with mock.patch("random.sample", return_value=list(inds)), mock.patch("random.random", return_value=0.5):
            winner = strat.select_one(inds)
            self.assertEqual(winner.fitness, 50.0)

        # If random.random() >= 0.8 -> remaining contestant wins (10.0)
        with mock.patch("random.sample", return_value=list(inds)), mock.patch("random.random", return_value=0.9):
            winner = strat.select_one(inds)
            self.assertEqual(winner.fitness, 10.0)

    def test_tournament_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            TournamentSelection(tournament_size=0)
        with self.assertRaises(ValueError):
            TournamentSelection(winner_prob=0.0)
        with self.assertRaises(ValueError):
            TournamentSelection(winner_prob=1.5)

        inds = [_make_individual(1, 10.0)]
        strat = TournamentSelection(tournament_size=3)
        with self.assertRaises(ValueError):
            strat.select(inds, k=1)


class TestProportionateSelection(unittest.TestCase):

    def test_roulette_wheel_maximization(self) -> None:
        inds = [
            _make_individual(1, 10.0),
            _make_individual(2, 20.0),
            _make_individual(3, 70.0),
        ]
        strat = RouletteWheelSelection()

        # Cumulative normalized weights: [0.10, 0.30, 1.00]
        with mock.patch("random.random", return_value=0.05):
            self.assertEqual(strat.select_one(inds).fitness, 10.0)
        with mock.patch("random.random", return_value=0.25):
            self.assertEqual(strat.select_one(inds).fitness, 20.0)
        with mock.patch("random.random", return_value=0.85):
            self.assertEqual(strat.select_one(inds).fitness, 70.0)

    def test_roulette_wheel_minimization(self) -> None:
        inds = [
            _make_individual(1, 10.0),
            _make_individual(2, 30.0),
            _make_individual(3, 100.0),
        ]
        # Inverted weights: max=100 -> [90, 70, 0] + offset 10 -> [100, 80, 10]
        strat = RouletteWheelSelection(fitness_offset=10.0, minimize=True)
        selected = strat.select(inds, k=5)
        self.assertEqual(len(selected), 5)

    def test_roulette_wheel_negative_fitness_error(self) -> None:
        inds = [_make_individual(1, -5.0), _make_individual(2, 10.0)]
        strat = RouletteWheelSelection()
        with self.assertRaises(ValueError):
            strat.select(inds, k=1)

        # With sufficient offset, it works
        strat_offset = RouletteWheelSelection(fitness_offset=10.0)
        res = strat_offset.select(inds, k=1)
        self.assertEqual(len(res), 1)

    def test_stochastic_universal_sampling_zero_drift(self) -> None:
        inds = [
            _make_individual(1, 10.0),  # 10%
            _make_individual(2, 20.0),  # 20%
            _make_individual(3, 70.0),  # 70%
        ]
        strat = StochasticUniversalSamplingSelection()
        # When selecting k=10, step_size = 100/10 = 10.0
        # Placing start_point at 5.0 (midpoint of first interval [0, 10]):
        # pointers: [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
        # ind1 (10): 1 pointer (5)
        # ind2 (20): 2 pointers (15, 25)
        # ind3 (70): 7 pointers (35, 45, 55, 65, 75, 85, 95)
        with mock.patch("random.uniform", return_value=5.0):
            selected = strat.select(inds, k=10)

        counts = {ind.fitness: selected.count(ind) for ind in inds}
        self.assertEqual(counts[10.0], 1)
        self.assertEqual(counts[20.0], 2)
        self.assertEqual(counts[70.0], 7)


class TestRankSelection(unittest.TestCase):

    def test_linear_rank_selection(self) -> None:
        inds = [
            _make_individual(1, 100.0),  # Rank 3
            _make_individual(2, 10.0),   # Rank 1
            _make_individual(3, 50.0),   # Rank 2
        ]
        strat = LinearRankSelection(selection_pressure=1.5)
        selected = strat.select(inds, k=5)
        self.assertEqual(len(selected), 5)

        with self.assertRaises(ValueError):
            LinearRankSelection(selection_pressure=0.9)
        with self.assertRaises(ValueError):
            LinearRankSelection(selection_pressure=2.1)

    def test_exponential_rank_selection(self) -> None:
        inds = [
            _make_individual(1, 100.0),
            _make_individual(2, 10.0),
            _make_individual(3, 50.0),
        ]
        strat = ExponentialRankSelection(c=0.5)
        selected = strat.select(inds, k=3)
        self.assertEqual(len(selected), 3)

        with self.assertRaises(ValueError):
            ExponentialRankSelection(c=0.0)
        with self.assertRaises(ValueError):
            ExponentialRankSelection(c=1.0)


class TestTruncationAndElitistSelection(unittest.TestCase):

    def test_truncation_with_top_k(self) -> None:
        inds = [_make_individual(i, float(i * 10)) for i in range(5)]
        strat = TruncationSelection(top_k=2, with_replacement=False)
        selected = strat.select(inds, k=2)
        self.assertEqual(len(selected), 2)
        for s in selected:
            self.assertIn(s.fitness, [30.0, 40.0])

        with self.assertRaises(ValueError):
            strat.select(inds, k=3)

    def test_truncation_with_top_ratio(self) -> None:
        inds = [_make_individual(i, float(i * 10)) for i in range(10)]
        strat = TruncationSelection(top_ratio=0.3, with_replacement=True)
        selected = strat.select(inds, k=5)
        self.assertEqual(len(selected), 5)

    def test_elitist_selection(self) -> None:
        inds = [_make_individual(i, float(i)) for i in range(6)]
        strat_k = ElitistSelection(num_elites=2)
        elites = strat_k.select(inds)
        self.assertEqual(len(elites), 2)
        self.assertEqual(elites[0].fitness, 5.0)
        self.assertEqual(elites[1].fitness, 4.0)

        strat_ratio = ElitistSelection(elite_ratio=0.5)
        elites_ratio = strat_ratio.select(inds)
        self.assertEqual(len(elites_ratio), 3)


class TestRandomAndBoltzmannSelection(unittest.TestCase):

    def test_random_selection(self) -> None:
        inds = [_make_individual(i) for i in range(4)]
        strat = RandomSelection(with_replacement=False)
        selected = strat.select(inds, k=3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)

        with self.assertRaises(ValueError):
            strat.select(inds, k=5)

    def test_boltzmann_selection(self) -> None:
        inds = [
            _make_individual(1, 10.0),
            _make_individual(2, 50.0),
        ]
        strat = BoltzmannSelection(temperature=0.001)
        selected = strat.select(inds, k=10)
        self.assertTrue(all(s.fitness == 50.0 for s in selected))

        with self.assertRaises(ValueError):
            BoltzmannSelection(temperature=-1.0)


if __name__ == "__main__":
    unittest.main()
