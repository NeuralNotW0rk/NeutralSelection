import unittest
from typing import Any
from neutral_selection.representation.genome import ContiguousArrayGenome
from neutral_selection.representation.individual import Individual


class MockPhenotype:
    def __init__(self, sum_val: float, raw_vals: list[float]) -> None:
        self.sum_val = sum_val
        self.raw_vals = raw_vals


class TestIndividual(unittest.TestCase):

    def test_initialization(self) -> None:
        genotype = ContiguousArrayGenome([1.0, 2.0, 3.0])
        ind = Individual(genotype)

        self.assertEqual(ind.genotype, genotype)
        self.assertIsNone(ind.fitness)
        self.assertFalse(ind.is_expressed)

        # Accessing phenotype before expression should raise ValueError
        with self.assertRaises(ValueError):
            _ = ind.phenotype

    def test_expression_and_caching(self) -> None:
        genotype = ContiguousArrayGenome([1.0, 2.0, 3.0])
        ind = Individual(genotype)

        call_count = 0

        def mock_decoder(g: ContiguousArrayGenome[float]) -> MockPhenotype:
            nonlocal call_count
            call_count += 1
            return MockPhenotype(sum(g), list(g))

        # First expression
        pheno1 = ind.express(mock_decoder)
        self.assertTrue(ind.is_expressed)
        self.assertEqual(call_count, 1)
        self.assertEqual(pheno1.sum_val, 6.0)
        self.assertEqual(pheno1.raw_vals, [1.0, 2.0, 3.0])

        # Accessing property
        self.assertEqual(ind.phenotype, pheno1)

        # Second expression (should be cached, not calling decoder again)
        pheno2 = ind.express(mock_decoder)
        self.assertEqual(call_count, 1)
        self.assertIs(pheno2, pheno1)

    def test_clear_expression(self) -> None:
        genotype = ContiguousArrayGenome([1.0, 2.0, 3.0])
        ind = Individual(genotype)

        def mock_decoder(g: ContiguousArrayGenome[float]) -> MockPhenotype:
            return MockPhenotype(sum(g), list(g))

        ind.express(mock_decoder)
        self.assertTrue(ind.is_expressed)

        # Clear expression
        ind.clear_expression()
        self.assertFalse(ind.is_expressed)

        with self.assertRaises(ValueError):
            _ = ind.phenotype

    def test_fail_fast_validations(self) -> None:
        genotype = ContiguousArrayGenome([1.0, 2.0, 3.0])
        ind = Individual(genotype)

        # Missing decode_fn (None)
        with self.assertRaises(ValueError):
            ind.express(None)  # type: ignore

        # Non-callable decode_fn
        with self.assertRaises(TypeError):
            ind.express("not-callable")  # type: ignore


if __name__ == "__main__":
    unittest.main()
