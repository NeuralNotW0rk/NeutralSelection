from __future__ import annotations

import unittest
import unittest.mock as mock
from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.variation.mutation import (
    mutate,
    InversionMutation,
    SwapMutation,
    ScrambleMutation,
    InsertionMutation,
    TranspositionMutation,
    DuplicationMutation,
    DeletionMutation,
    GaussianMutation,
    UniformRealMutation,
    PolynomialMutation,
    CauchyMutation,
    BitFlipMutation,
    BoundaryMutation,
)


class TestSequenceMutation(unittest.TestCase):

    def test_insertion_mutation(self) -> None:
        g = Genome([10, 20, 30, 40, 50])
        strat = InsertionMutation()

        with mock.patch("random.sample", return_value=[1, 3]):
            # Extracts item at index 1 (value 20), inserts at index 3 -> [10, 30, 40, 20, 50]
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [10, 30, 40, 20, 50])

    def test_transposition_mutation(self) -> None:
        g = Genome([1, 2, 3, 4, 5, 6, 7, 8])
        strat = TranspositionMutation()

        # Cuts at [1, 3, 5, 7] -> block1: [2, 3] (idx 1:3), block2: [6, 7] (idx 5:7)
        with mock.patch("random.sample", return_value=[1, 3, 5, 7]):
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [1, 6, 7, 4, 5, 2, 3, 8])

    def test_duplication_mutation(self) -> None:
        g = Genome([10, 20, 30])
        strat = DuplicationMutation()

        with mock.patch("random.sample", return_value=[1, 2]), mock.patch("random.randint", return_value=0):
            # Duplicates [20] and inserts at position 0 -> [20, 10, 20, 30]
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [20, 10, 20, 30])

    def test_deletion_mutation(self) -> None:
        g = Genome([10, 20, 30, 40, 50])
        strat = DeletionMutation(min_length=2)

        with mock.patch("random.randint", side_effect=[1, 3]):
            # Deletes slice 1:3 ([20, 30]) -> [10, 40, 50]
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [10, 40, 50])


class TestRealValuedMutation(unittest.TestCase):

    def test_gaussian_mutation(self) -> None:
        g = Genome([1.0, 2.0, 3.0])
        strat = GaussianMutation(sigma=0.5, mutation_rate=1.0, bounds=(0.0, 5.0))

        with mock.patch("random.gauss", return_value=0.5):
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [1.5, 2.5, 3.5])

        # Clamping
        strat_clamp = GaussianMutation(sigma=5.0, bounds=(0.0, 2.0))
        with mock.patch("random.gauss", return_value=10.0):
            mutated_clamp = mutate(g, strat_clamp)
            self.assertEqual(list(mutated_clamp), [2.0, 2.0, 2.0])

    def test_uniform_real_mutation(self) -> None:
        g = Genome([10.0, 20.0])
        strat = UniformRealMutation(delta=2.0, mutation_rate=1.0)

        with mock.patch("random.uniform", return_value=1.5):
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [11.5, 21.5])

    def test_polynomial_mutation(self) -> None:
        g = Genome([0.0, 0.5])
        strat = PolynomialMutation(bounds=(-1.0, 1.0), eta_m=20.0, mutation_rate=1.0)
        mutated = mutate(g, strat)
        self.assertEqual(len(mutated), 2)
        self.assertTrue(all(-1.0 <= x <= 1.0 for x in mutated))

    def test_cauchy_mutation(self) -> None:
        g = Genome([5.0, 10.0])
        strat = CauchyMutation(scale=1.0, mutation_rate=1.0, bounds=(0.0, 20.0))
        with mock.patch("random.random", return_value=0.5):
            # tan(0) = 0 -> no change
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [5.0, 10.0])


class TestBinaryMutation(unittest.TestCase):

    def test_bit_flip_mutation_bool(self) -> None:
        g = Genome([True, False, True])
        strat = BitFlipMutation(mutation_rate=1.0)
        mutated = mutate(g, strat)
        self.assertEqual(list(mutated), [False, True, False])

    def test_bit_flip_mutation_int(self) -> None:
        g = Genome([1, 0, 1])
        strat = BitFlipMutation(mutation_rate=1.0)
        mutated = mutate(g, strat)
        self.assertEqual(list(mutated), [0, 1, 0])

    def test_boundary_mutation(self) -> None:
        g = Genome([2.5, 3.5])
        strat = BoundaryMutation(bounds=(0.0, 10.0), mutation_rate=1.0)

        with mock.patch("random.choice", return_value=10.0):
            mutated = mutate(g, strat)
            self.assertEqual(list(mutated), [10.0, 10.0])


if __name__ == "__main__":
    unittest.main()
