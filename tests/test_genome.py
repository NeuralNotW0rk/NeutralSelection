import unittest
import random
from dataclasses import dataclass
from typing import Any

from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.reproduction.recombination import ElementwiseCrossover, NPointCrossover, RandomNPointCrossover
from neutral_selection.reproduction.mutation import mutate, UniformMutation, InversionMutation, SwapMutation, ScrambleMutation, gaussian_noise_mutator, bit_flip_mutator, attribute_mutator



@dataclass
class CustomPayload:
    value: float
    tag: str


class TestGenomeAndSegment(unittest.TestCase):

    def test_initialization(self) -> None:
        g = Genome([1, 2, 3])
        self.assertEqual(list(g), [1, 2, 3])

        with self.assertRaises(TypeError):
            Genome((1, 2, 3))  # type: ignore

    def test_sequence_protocol(self) -> None:
        g = Genome([10, 20, 30, 40])
        self.assertEqual(len(g), 4)
        self.assertEqual(g[0], 10)
        self.assertEqual(g[1:3], [20, 30])
        self.assertEqual([x for x in g], [10, 20, 30, 40])

        with self.assertRaises(TypeError):
            _ = g["invalid"]  # type: ignore

    def test_segment_initialization(self) -> None:
        s = Segment("block1", [1, 2])
        self.assertEqual(s.key, "block1")
        self.assertEqual(list(s), [1, 2])


class TestVariationStrategies(unittest.TestCase):

    def test_n_point_crossover_strategy(self) -> None:
        g1 = Genome([1, 2, 3, 4, 5])
        g2 = Genome([10, 20, 30, 40, 50])

        strategy = NPointCrossover([2])
        child_a, child_b = strategy(g1, g2)
        self.assertEqual(list(child_a), [1, 2, 30, 40, 50])
        self.assertEqual(list(child_b), [10, 20, 3, 4, 5])

        # Fail-Fast validation on initialization
        with self.assertRaises(TypeError):
            NPointCrossover("invalid")  # type: ignore
        with self.assertRaises(ValueError):
            NPointCrossover([-2])

        # Fail-Fast on call:
        with self.assertRaises(TypeError):
            strategy("invalid-parent", g2)  # type: ignore

    def test_random_n_point_crossover_strategy(self) -> None:
        g1 = Genome([1, 2, 3, 4, 5])
        g2 = Genome([10, 20, 30, 40, 50])

        strategy = RandomNPointCrossover(1)
        # Mock random sample to return [2]
        import unittest.mock as mock
        with mock.patch("random.sample", return_value=[2]):
            child_a, child_b = strategy(g1, g2)
        self.assertEqual(list(child_a), [1, 2, 30, 40, 50])
        self.assertEqual(list(child_b), [10, 20, 3, 4, 5])

    def test_elementwise_crossover_strategy(self) -> None:
        g1 = Genome([
            CustomPayload(1.0, "A"),
            CustomPayload(2.0, "B")
        ])
        g2 = Genome([
            CustomPayload(3.0, "X"),
            CustomPayload(7.0, "Y")
        ])

        def blend_payload(p1: CustomPayload, p2: CustomPayload, weight: float) -> CustomPayload:
            val = p1.value * (1.0 - weight) + p2.value * weight
            return CustomPayload(val, f"{p1.tag}-{p2.tag}")

        strategy = ElementwiseCrossover(0.75, blend_payload)
        child = strategy(g1, g2)
        self.assertEqual(len(child), 2)
        self.assertEqual(child[0].value, 2.5)
        self.assertEqual(child[0].tag, "A-X")

        # Fail-Fast on initialization
        with self.assertRaises(ValueError):
            ElementwiseCrossover(0.5, None)  # type: ignore
        with self.assertRaises(TypeError):
            ElementwiseCrossover("invalid", blend_payload)  # type: ignore

        # Fail-Fast on call:
        with self.assertRaises(TypeError):
            strategy("invalid-parent", g2)  # type: ignore

    def test_uniform_mutation_strategy(self) -> None:
        g = Genome([
            CustomPayload(10.0, "first"),
            CustomPayload(20.0, "second")
        ])

        def mutate_fn(p: CustomPayload) -> CustomPayload:
            return CustomPayload(p.value + 5.0, p.tag.upper())

        strategy_all = UniformMutation(1.0, mutate_fn)
        mutated_all = mutate(g, strategy_all)
        self.assertEqual(mutated_all[0].value, 15.0)
        self.assertEqual(mutated_all[0].tag, "FIRST")

        strategy_none = UniformMutation(0.0, mutate_fn)
        mutated_none = mutate(g, strategy_none)
        self.assertEqual(mutated_none[0].value, 10.0)

        # Fail-Fast on initialization
        with self.assertRaises(ValueError):
            UniformMutation(0.5, None)  # type: ignore
        with self.assertRaises(ValueError):
            UniformMutation(-0.1, mutate_fn)

        # Fail-Fast on call:
        with self.assertRaises(TypeError):
            mutate("invalid-genome", strategy_all)

    def test_inversion_mutation_strategy(self) -> None:
        g = Genome([1, 2, 3, 4, 5])
        strategy = InversionMutation()

        # Mock random sample to reverse items between index 1 and 4 -> elements [2, 3, 4] -> reversed [4, 3, 2]
        import unittest.mock as mock
        with mock.patch("random.sample", return_value=[1, 4]):
            mutated = mutate(g, strategy)

        self.assertEqual(list(mutated), [1, 4, 3, 2, 5])
        self.assertNotIsInstance(mutated, list)
        self.assertIsInstance(mutated, Genome)

        # Boundary condition: length 1 genome should remain unchanged
        g_short = Genome([42])
        self.assertEqual(list(mutate(g_short, strategy)), [42])

    def test_swap_mutation_strategy(self) -> None:
        g = Genome([1, 2, 3, 4, 5])
        strategy = SwapMutation()

        # Mock random sample to swap index 1 and 3 (values 2 and 4)
        import unittest.mock as mock
        with mock.patch("random.sample", return_value=[1, 3]):
            mutated = mutate(g, strategy)

        self.assertEqual(list(mutated), [1, 4, 3, 2, 5])

        # Boundary condition: length 1 genome should remain unchanged
        g_short = Genome([42])
        self.assertEqual(list(mutate(g_short, strategy)), [42])

    def test_scramble_mutation_strategy(self) -> None:
        g = Genome([1, 2, 3, 4, 5])
        strategy = ScrambleMutation()

        # Mock random shuffle to reverse the scramble slice (index 1 to 4 -> [2, 3, 4])
        # random.shuffle works in-place; we mock it to reverse the input list
        def mock_shuffle(x: list) -> None:
            x.reverse()

        import unittest.mock as mock
        with mock.patch("random.sample", return_value=[1, 4]), mock.patch("random.shuffle", side_effect=mock_shuffle):
            mutated = mutate(g, strategy)

        self.assertEqual(list(mutated), [1, 4, 3, 2, 5])

        # Boundary condition: length 1 genome should remain unchanged
        g_short = Genome([42])
        self.assertEqual(list(mutate(g_short, strategy)), [42])

    def test_generic_mutators(self) -> None:
        # Test gaussian_noise_mutator (float)
        float_mut = gaussian_noise_mutator(1.5)
        # Mock random.gauss to return 1.5
        import unittest.mock as mock
        with mock.patch("random.gauss", return_value=1.5) as mock_gauss:
            self.assertEqual(float_mut(10.0), 11.5)
            mock_gauss.assert_called_once_with(0.0, 1.5)

        # Test gaussian_noise_mutator with custom mean
        float_mut_mean = gaussian_noise_mutator(1.5, mean=5.0)
        with mock.patch("random.gauss", return_value=6.5) as mock_gauss_mean:
            self.assertEqual(float_mut_mean(10.0), 16.5)
            mock_gauss_mean.assert_called_once_with(5.0, 1.5)

        # Test bit_flip_mutator
        bool_mut_always = bit_flip_mutator(1.0)
        self.assertFalse(bool_mut_always(True))
        bool_mut_never = bit_flip_mutator(0.0)
        self.assertTrue(bool_mut_never(True))

        # Test attribute_mutator
        @dataclass
        class SimpleGene:
            val: float
            active: bool

            def __post_init__(self) -> None:
                pass

        gene = SimpleGene(10.0, True)
        gene_mutator = attribute_mutator({
            "val": gaussian_noise_mutator(1.0),
            "active": bit_flip_mutator(1.0)
        })

        with mock.patch("random.gauss", return_value=2.0):
            mutated_gene = gene_mutator(gene)

        self.assertEqual(mutated_gene.val, 12.0)
        self.assertFalse(mutated_gene.active)
        # Verify original was not mutated in-place
        self.assertEqual(gene.val, 10.0)
        self.assertTrue(gene.active)


if __name__ == "__main__":
    unittest.main()
