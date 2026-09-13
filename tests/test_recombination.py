from __future__ import annotations

import unittest
import unittest.mock as mock
from neutral_selection.representation.genome import Genome
from neutral_selection.variation.recombination import (
    recombine,
    OnePointCrossover,
    TwoPointCrossover,
    UniformCrossover,
    ShuffleCrossover,
    OrderCrossover,
    PartiallyMatchedCrossover,
    CycleCrossover,
    ArithmeticCrossover,
    BlendCrossover,
    SimulatedBinaryCrossover,
)


class TestPositionalCrossover(unittest.TestCase):

    def test_one_point_crossover(self) -> None:
        p1 = Genome([1, 2, 3, 4, 5])
        p2 = Genome([10, 20, 30, 40, 50])

        strat = OnePointCrossover(cut_point=2)
        c1, c2 = recombine(p1, p2, strat)
        self.assertEqual(list(c1), [1, 2, 30, 40, 50])
        self.assertEqual(list(c2), [10, 20, 3, 4, 5])

    def test_two_point_crossover(self) -> None:
        p1 = Genome([1, 2, 3, 4, 5])
        p2 = Genome([10, 20, 30, 40, 50])

        strat = TwoPointCrossover(cut_points=(1, 4))
        c1, c2 = recombine(p1, p2, strat)
        self.assertEqual(list(c1), [1, 20, 30, 40, 5])
        self.assertEqual(list(c2), [10, 2, 3, 4, 50])

    def test_uniform_crossover(self) -> None:
        p1 = Genome([1, 2, 3])
        p2 = Genome([10, 20, 30])

        strat = UniformCrossover(swap_prob=1.0)
        c1, c2 = recombine(p1, p2, strat)
        self.assertEqual(list(c1), [1, 2, 3])
        self.assertEqual(list(c2), [10, 20, 30])

        strat_swap = UniformCrossover(swap_prob=0.0)
        c1, c2 = recombine(p1, p2, strat_swap)
        self.assertEqual(list(c1), [10, 20, 30])
        self.assertEqual(list(c2), [1, 2, 3])

    def test_shuffle_crossover(self) -> None:
        p1 = Genome([1, 2, 3, 4])
        p2 = Genome([10, 20, 30, 40])
        strat = ShuffleCrossover()
        c1, c2 = recombine(p1, p2, strat)
        self.assertEqual(len(c1), 4)
        self.assertEqual(len(c2), 4)


class TestPermutationCrossover(unittest.TestCase):

    def test_order_crossover(self) -> None:
        p1 = Genome([1, 2, 3, 4, 5, 6, 7, 8])
        p2 = Genome([8, 7, 6, 5, 4, 3, 2, 1])

        strat = OrderCrossover()
        with mock.patch("random.sample", return_value=[2, 5]):
            # Primary slice 2:5 -> [3, 4, 5]
            c1, c2 = recombine(p1, p2, strat)

            # Check valid permutation (all elements present, no duplicates)
            self.assertEqual(sorted(list(c1)), [1, 2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(sorted(list(c2)), [1, 2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(list(c1)[2:5], [3, 4, 5])
            self.assertEqual(list(c2)[2:5], [6, 5, 4])

    def test_partially_matched_crossover(self) -> None:
        p1 = Genome([1, 2, 3, 4, 5, 6, 7, 8])
        p2 = Genome([8, 7, 6, 5, 4, 3, 2, 1])

        strat = PartiallyMatchedCrossover()
        with mock.patch("random.sample", return_value=[3, 6]):
            c1, c2 = recombine(p1, p2, strat)

            # Check valid permutation
            self.assertEqual(sorted(list(c1)), [1, 2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(sorted(list(c2)), [1, 2, 3, 4, 5, 6, 7, 8])
            self.assertEqual(list(c1)[3:6], [4, 5, 6])
            self.assertEqual(list(c2)[3:6], [5, 4, 3])

    def test_cycle_crossover(self) -> None:
        p1 = Genome([1, 2, 3, 4, 5, 6, 7, 8, 9])
        p2 = Genome([9, 3, 7, 8, 2, 6, 5, 1, 4])

        strat = CycleCrossover()
        c1, c2 = recombine(p1, p2, strat)

        self.assertEqual(sorted(list(c1)), list(range(1, 10)))
        self.assertEqual(sorted(list(c2)), list(range(1, 10)))


class TestRealValuedCrossover(unittest.TestCase):

    def test_arithmetic_crossover(self) -> None:
        p1 = Genome([1.0, 2.0, 3.0])
        p2 = Genome([3.0, 4.0, 5.0])

        strat = ArithmeticCrossover(alpha=0.5)
        c1, c2 = recombine(p1, p2, strat)
        self.assertEqual(list(c1), [2.0, 3.0, 4.0])
        self.assertEqual(list(c2), [2.0, 3.0, 4.0])

    def test_blend_crossover(self) -> None:
        p1 = Genome([1.0, 5.0])
        p2 = Genome([3.0, 5.0])

        strat = BlendCrossover(alpha=0.5, bounds=(0.0, 10.0))
        c1, c2 = recombine(p1, p2, strat)
        self.assertEqual(len(c1), 2)
        self.assertEqual(len(c2), 2)
        self.assertTrue(0.0 <= c1[0] <= 10.0)

    def test_simulated_binary_crossover(self) -> None:
        p1 = Genome([1.0, 2.0])
        p2 = Genome([3.0, 4.0])

        strat = SimulatedBinaryCrossover(eta_c=2.0, swap_prob=1.0, bounds=(-10.0, 10.0))
        c1, c2 = recombine(p1, p2, strat)
        self.assertEqual(len(c1), 2)
        self.assertEqual(len(c2), 2)
        self.assertTrue(all(-10.0 <= x <= 10.0 for x in c1))


if __name__ == "__main__":
    unittest.main()
