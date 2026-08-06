import unittest
from dataclasses import dataclass
from typing import Any
from neutral_selection.representation.genome import ContiguousArrayGenome, SegmentedGenome
from neutral_selection.variation.recombination import ElementwiseCrossover, NPointCrossover, SegmentSwapCrossover
from neutral_selection.variation.mutate import UniformMutation


@dataclass
class CustomPayload:
    value: float
    tag: str


class TestContiguousArrayGenome(unittest.TestCase):

    def test_initialization(self) -> None:
        g = ContiguousArrayGenome([1, 2, 3])
        self.assertEqual(list(g), [1, 2, 3])

        with self.assertRaises(TypeError):
            ContiguousArrayGenome((1, 2, 3))  # type: ignore

    def test_sequence_protocol(self) -> None:
        g = ContiguousArrayGenome([10, 20, 30, 40])
        self.assertEqual(len(g), 4)
        self.assertEqual(g[0], 10)
        self.assertEqual(g[1:3], [20, 30])
        self.assertEqual([x for x in g], [10, 20, 30, 40])

        with self.assertRaises(TypeError):
            _ = g["invalid"]  # type: ignore

    def test_map(self) -> None:
        g = ContiguousArrayGenome([1, 2, 3])
        mapped = g.map(lambda x: x * 10)
        self.assertEqual(list(mapped), [10, 20, 30])

        # Fail-Fast:
        with self.assertRaises(ValueError):
            g.map(None)  # type: ignore
        with self.assertRaises(TypeError):
            g.map("not-callable")  # type: ignore

    def test_zip_map(self) -> None:
        g1 = ContiguousArrayGenome([1, 2, 3])
        g2 = ContiguousArrayGenome([10, 20, 30])
        zipped = g1.zip_map(g2, lambda x, y, b: x * (1 - b) + y * b, 0.5)
        self.assertEqual(list(zipped), [5.5, 11.0, 16.5])

        # Fail-Fast:
        with self.assertRaises(TypeError):
            g1.zip_map("not-a-genome", lambda x, y, b: x, 0.5)  # type: ignore
        with self.assertRaises(ValueError):
            g1.zip_map(g2, None, 0.5)  # type: ignore
        with self.assertRaises(TypeError):
            g1.zip_map(g2, "not-callable", 0.5)  # type: ignore
        with self.assertRaises(TypeError):
            g1.zip_map(g2, lambda x, y, b: x, "invalid")  # type: ignore
        with self.assertRaises(ValueError):
            g1.zip_map(ContiguousArrayGenome([1]), lambda x, y, b: x, 0.5)

    def test_crossover_structural(self) -> None:
        g1 = ContiguousArrayGenome([1, 2, 3, 4, 5])
        g2 = ContiguousArrayGenome([10, 20, 30, 40, 50])

        # 1-point crossover at index 2
        child_a, child_b = g1.crossover(g2, [2])
        self.assertEqual(list(child_a), [1, 2, 30, 40, 50])
        self.assertEqual(list(child_b), [10, 20, 3, 4, 5])

        # Validation: other must be ContiguousArrayGenome
        with self.assertRaises(TypeError):
            g1.crossover([10, 20, 30, 40, 50], [2])  # type: ignore

        # Validation: cut points must be non-negative
        with self.assertRaises(ValueError):
            g1.crossover(g2, [-1])


class TestVariationStrategies(unittest.TestCase):

    def test_n_point_crossover_strategy(self) -> None:
        g1 = ContiguousArrayGenome([1, 2, 3, 4, 5])
        g2 = ContiguousArrayGenome([10, 20, 30, 40, 50])

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
            strategy("invalid-parent", g2)

    def test_elementwise_crossover_strategy(self) -> None:
        g1 = ContiguousArrayGenome([
            CustomPayload(1.0, "A"),
            CustomPayload(2.0, "B")
        ])
        g2 = ContiguousArrayGenome([
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
            strategy("invalid-parent", g2)

    def test_uniform_mutation_strategy(self) -> None:
        g = ContiguousArrayGenome([
            CustomPayload(10.0, "first"),
            CustomPayload(20.0, "second")
        ])

        def mutate_fn(p: CustomPayload) -> CustomPayload:
            return CustomPayload(p.value + 5.0, p.tag.upper())

        strategy_all = UniformMutation(1.0, mutate_fn)
        mutated_all = strategy_all(g)
        self.assertEqual(mutated_all[0].value, 15.0)
        self.assertEqual(mutated_all[0].tag, "FIRST")

        strategy_none = UniformMutation(0.0, mutate_fn)
        mutated_none = strategy_none(g)
        self.assertEqual(mutated_none[0].value, 10.0)

        # Fail-Fast on initialization
        with self.assertRaises(ValueError):
            UniformMutation(0.5, None)  # type: ignore
        with self.assertRaises(ValueError):
            UniformMutation(-0.1, mutate_fn)

        # Fail-Fast on call:
        with self.assertRaises(TypeError):
            strategy_all("invalid-genome")

    def test_segment_swap_crossover_strategy(self) -> None:
        seg1_a = ContiguousArrayGenome([1, 2])
        seg2_a = ContiguousArrayGenome([3, 4])
        genome_a = SegmentedGenome({"block1": seg1_a, "block2": seg2_a})

        seg1_b = ContiguousArrayGenome([10, 20])
        seg2_b = ContiguousArrayGenome([30, 40])
        genome_b = SegmentedGenome({"block1": seg1_b, "block2": seg2_b})

        strategy = SegmentSwapCrossover({"block1"})
        child = strategy(genome_a, genome_b)
        self.assertEqual(list(child["block1"]), [10, 20])
        self.assertEqual(list(child["block2"]), [3, 4])

        # Fail-Fast on initialization
        with self.assertRaises(TypeError):
            SegmentSwapCrossover(["block1"])  # type: ignore

        # Fail-Fast on call
        with self.assertRaises(TypeError):
            strategy("invalid-genome", genome_b)


class TestSegmentedGenome(unittest.TestCase):

    def setUp(self) -> None:
        self.seg1_a = ContiguousArrayGenome([1, 2])
        self.seg2_a = ContiguousArrayGenome([3, 4])
        self.genome_a = SegmentedGenome({"block1": self.seg1_a, "block2": self.seg2_a})

        self.seg1_b = ContiguousArrayGenome([10, 20])
        self.seg2_b = ContiguousArrayGenome([30, 40])
        self.genome_b = SegmentedGenome({"block1": self.seg1_b, "block2": self.seg2_b})

    def test_initialization(self) -> None:
        with self.assertRaises(TypeError):
            SegmentedGenome([self.seg1_a, self.seg2_a])  # type: ignore
        with self.assertRaises(TypeError):
            SegmentedGenome({"block1": [1, 2]})  # type: ignore

    def test_dict_like_navigation(self) -> None:
        self.assertEqual(self.genome_a["block1"], self.seg1_a)
        with self.assertRaises(KeyError):
            _ = self.genome_a["nonexistent"]
        self.assertEqual(set(self.genome_a.keys()), {"block1", "block2"})
        items = dict(self.genome_a.items())
        self.assertEqual(items["block1"], self.seg1_a)

    def test_map(self) -> None:
        mapped = self.genome_a.map(lambda x: x + 100)
        self.assertEqual(list(mapped["block1"]), [101, 102])
        self.assertEqual(list(mapped["block2"]), [103, 104])

        # Verify fail-fast
        with self.assertRaises(ValueError):
            self.genome_a.map(None)  # type: ignore

    def test_zip_map(self) -> None:
        def blend_fn(x: int, y: int, b: float) -> int:
            return int(x * (1 - b) + y * b)

        zipped = self.genome_a.zip_map(self.genome_b, blend_fn, 0.5)
        self.assertEqual(list(zipped["block1"]), [5, 11])
        self.assertEqual(list(zipped["block2"]), [16, 22])

        # Fail-Fast
        with self.assertRaises(TypeError):
            self.genome_a.zip_map("invalid", blend_fn, 0.5)  # type: ignore
        with self.assertRaises(ValueError):
            self.genome_a.zip_map(self.genome_b, None, 0.5)  # type: ignore

    def test_crossover_segments(self) -> None:
        child = self.genome_a.crossover_segments(self.genome_b, {"block1"})
        self.assertEqual(list(child["block1"]), [10, 20])
        self.assertEqual(list(child["block2"]), [3, 4])

        with self.assertRaises(KeyError):
            self.genome_a.crossover_segments(self.genome_b, {"nonexistent"})


if __name__ == "__main__":
    unittest.main()
