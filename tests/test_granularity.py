from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import unittest
from typing import Any

from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.representation.hierarchy import (
    flatten_hierarchy,
    flatten_strand,
    resolve_granularity,
    unflatten_hierarchy,
)
from neutral_selection.variation.mutation import (
    BitFlipMutation,
    BoundaryMutation,
    FunctionMutation,
    GaussianMutation,
    SwapMutation,
    UniformMutation,
    mutate,
)
from neutral_selection.variation.recombination import (
    ElementwiseCrossover,
    FunctionCrossover,
    TwoPointCrossover,
)
from neutral_selection.builders import build_crossover_strategy, build_mutation_strategy

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch


@dataclass
class Payload:
    """Mock gene with a numeric list field, so atom granularity has something to descend into."""
    tag: str
    values: list


def _nested_genome() -> Genome:
    return Genome([
        Segment("a", [Payload("x", [1.0, 2.0]), Payload("y", [3.0])]),
        Segment("b", [Payload("z", [4.0, 5.0, 6.0])]),
    ])


class TestResolveGranularity(unittest.TestCase):

    def test_defaults_to_atom(self) -> None:
        self.assertEqual(resolve_granularity(), "atom")

    def test_names_and_depths(self) -> None:
        self.assertEqual(resolve_granularity("GENE"), "gene")
        self.assertEqual(resolve_granularity("tensor"), "tensor")
        self.assertEqual(resolve_granularity(2), 2)

    def test_max_depth_is_an_int_granularity(self) -> None:
        self.assertEqual(resolve_granularity(max_depth=1), 1)
        self.assertEqual(resolve_granularity(1, max_depth=1), 1)
        with self.assertRaises(ValueError):
            resolve_granularity("gene", max_depth=1)

    def test_invalid_values(self) -> None:
        for bad in ("layer", -1, 1.5, True):
            with self.assertRaises(ValueError):
                resolve_granularity(bad)  # type: ignore[arg-type]


class TestFlattenGranularity(unittest.TestCase):

    def test_atom_descends_into_genes(self) -> None:
        leaves, _ = flatten_hierarchy(_nested_genome())
        self.assertEqual(leaves, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_gene_stops_at_genome_items_at_any_depth(self) -> None:
        genome = _nested_genome()
        leaves, treedef = flatten_hierarchy(genome, granularity="gene")
        self.assertEqual([p.tag for p in leaves], ["x", "y", "z"])
        rebuilt = unflatten_hierarchy(leaves, treedef)
        self.assertEqual([s.key for s in rebuilt], ["a", "b"])
        self.assertEqual(rebuilt[0][1].tag, "y")

    def test_gene_differs_from_depth_one_on_nested_genomes(self) -> None:
        leaves, _ = flatten_hierarchy(_nested_genome(), granularity=1)
        self.assertEqual([s.key for s in leaves], ["a", "b"])

    def test_gene_expands_a_plain_sequence_root(self) -> None:
        leaves, _ = flatten_hierarchy([Payload("x", [1.0]), Payload("y", [2.0])], granularity="gene")
        self.assertEqual([p.tag for p in leaves], ["x", "y"])

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_tensor_keeps_each_tensor_whole(self) -> None:
        genome = Genome([torch.zeros(2, 3), torch.ones(4)])
        leaves, treedef = flatten_strand(genome, granularity="tensor")
        self.assertEqual([tuple(t.shape) for t in leaves], [(2, 3), (4,)])
        rebuilt = unflatten_hierarchy(leaves, treedef)
        self.assertTrue(torch.equal(rebuilt[1], torch.ones(4)))


class TestOperatorGranularity(unittest.TestCase):

    def test_swap_at_gene_moves_whole_genes(self) -> None:
        genome = Genome([Payload("x", [1.0]), Payload("y", [2.0, 3.0])])
        swapped = mutate(genome, SwapMutation(granularity="gene"))
        self.assertEqual([p.tag for p in swapped], ["y", "x"])

    def test_two_point_crossover_at_gene(self) -> None:
        a = Genome([Payload("a1", [0.0]), Payload("a2", [0.0]), Payload("a3", [0.0])])
        b = Genome([Payload("b1", [1.0]), Payload("b2", [1.0]), Payload("b3", [1.0])])
        child_a, _ = TwoPointCrossover(granularity="gene")(a, b)
        self.assertTrue(all(isinstance(p, Payload) for p in child_a))

    def test_max_depth_still_accepted(self) -> None:
        strat = GaussianMutation(sigma=1.0, max_depth=1)
        self.assertEqual(strat.granularity, 1)
        with self.assertRaises(ValueError):
            GaussianMutation(sigma=1.0, max_depth=1, granularity="gene")

    def test_bit_flip_handles_nested_segments(self) -> None:
        genome = Genome([Segment("a", [True, False]), Segment("b", [1, 0])])
        flipped = mutate(genome, BitFlipMutation(mutation_rate=1.0))
        self.assertEqual(list(flipped[0]), [False, True])
        self.assertEqual(list(flipped[1]), [0, 1])

    def test_boundary_handles_nested_segments(self) -> None:
        genome = Genome([Segment("a", [0.3, 0.6])])
        reset = mutate(genome, BoundaryMutation(bounds=(0.0, 1.0), mutation_rate=1.0))
        self.assertTrue(all(v in (0.0, 1.0) for v in reset[0]))

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_boundary_on_tensor_strand(self) -> None:
        genome = Genome([torch.full((5,), 0.5)])
        reset = mutate(genome, BoundaryMutation(bounds=(-1.0, 1.0), mutation_rate=1.0))
        self.assertTrue(bool(((reset[0] == -1.0) | (reset[0] == 1.0)).all()))


class TestFunctionMutation(unittest.TestCase):

    def test_atom_passes_scalars(self) -> None:
        seen: list[Any] = []
        mutate(_nested_genome(), FunctionMutation(lambda v: seen.append(v) or v))
        self.assertEqual(seen, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_gene_passes_whole_genes(self) -> None:
        mutated = mutate(_nested_genome(), FunctionMutation(lambda p: Payload(p.tag.upper(), p.values), granularity="gene"))
        self.assertEqual([p.tag for seg in mutated for p in seg], ["X", "Y", "Z"])

    @unittest.skipUnless(HAS_TORCH, "torch not installed")
    def test_tensor_passes_whole_tensors(self) -> None:
        genome = Genome([torch.zeros(3), torch.zeros(2)])
        mutated = mutate(genome, FunctionMutation(lambda t: t + t.numel(), granularity="tensor"))
        self.assertTrue(torch.equal(mutated[0], torch.full((3,), 3.0)))
        self.assertTrue(torch.equal(mutated[1], torch.full((2,), 2.0)))

    def test_rate_zero_leaves_genome_unchanged(self) -> None:
        mutated = mutate(Genome([1.0, 2.0]), FunctionMutation(lambda v: v + 10.0, mutation_rate=0.0))
        self.assertEqual(list(mutated), [1.0, 2.0])

    def test_uniform_mutation_is_a_deprecated_gene_alias(self) -> None:
        with self.assertWarns(DeprecationWarning):
            strat = UniformMutation(1.0, lambda p: Payload(p.tag + "!", p.values))
        self.assertIsInstance(strat, FunctionMutation)
        self.assertEqual(strat.granularity, "gene")
        mutated = mutate(_nested_genome(), strat)
        self.assertEqual(mutated[1][0].tag, "z!")

    def test_builder(self) -> None:
        strat = build_mutation_strategy({"type": "function", "fn": lambda v: v, "granularity": "gene"})
        self.assertIsInstance(strat, FunctionMutation)
        self.assertEqual(strat.granularity, "gene")
        with self.assertRaises(ValueError):
            build_mutation_strategy({"type": "function"})


class TestFunctionCrossover(unittest.TestCase):

    @staticmethod
    def _mean(x: float, y: float, w: float) -> float:
        return x * (1.0 - w) + y * w

    def test_returns_a_pair_with_swapped_arguments(self) -> None:
        child_1, child_2 = FunctionCrossover(self._mean, blend_factor=0.25)(Genome([0.0, 4.0]), Genome([4.0, 0.0]))
        self.assertEqual(list(child_1), [1.0, 3.0])
        self.assertEqual(list(child_2), [3.0, 1.0])

    def test_single_child(self) -> None:
        child = FunctionCrossover(self._mean, num_children=1)(Genome([0.0, 4.0]), Genome([4.0, 0.0]))
        self.assertIsInstance(child, Genome)
        self.assertEqual(list(child), [2.0, 2.0])
        with self.assertRaises(ValueError):
            FunctionCrossover(self._mean, num_children=3)

    def test_structure_mismatch_raises(self) -> None:
        strat = FunctionCrossover(self._mean)
        with self.assertRaises(ValueError):
            strat(Genome([1.0, 2.0]), Genome([1.0]))
        with self.assertRaises(ValueError):
            strat(Genome([Segment("a", [1.0])]), Genome([Segment("b", [1.0])]))

    def test_elementwise_crossover_is_a_deprecated_single_child_alias(self) -> None:
        blend = lambda p, q, w: Payload(p.tag + q.tag, p.values)
        with self.assertWarns(DeprecationWarning):
            strat = ElementwiseCrossover(0.5, blend)
        self.assertIsInstance(strat, FunctionCrossover)
        child = strat(_nested_genome(), _nested_genome())
        self.assertIsInstance(child, Genome)
        self.assertEqual(child[0][0].tag, "xx")

    def test_builder(self) -> None:
        strat = build_crossover_strategy({"type": "function", "fn": self._mean, "granularity": 1})
        self.assertIsInstance(strat, FunctionCrossover)
        self.assertEqual(strat.blend_factor, 0.5)
        self.assertEqual(strat.granularity, 1)


if __name__ == "__main__":
    unittest.main()
