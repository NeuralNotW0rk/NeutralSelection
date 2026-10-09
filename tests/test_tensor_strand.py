from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import unittest
import unittest.mock as mock
from typing import Any

from neutral_selection.representation.genome import Genome
from neutral_selection.representation.hierarchy import (
    flatten_hierarchy,
    flatten_strand,
    is_tensor_strand,
    unflatten_hierarchy,
)
from neutral_selection.variation.recombination.n_point_crossover import (
    NPointCrossover,
    TwoPointCrossover,
    UniformCrossover,
)
from neutral_selection.variation.recombination.real_crossover import (
    ArithmeticCrossover,
    SimulatedBinaryCrossover,
)
from neutral_selection.variation.mutation.real_mutation import GaussianMutation
from neutral_selection.variation.mutation.sequence_mutation import (
    InversionMutation,
    ScrambleMutation,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch


@dataclass
class TensorGene:
    """Mock gene holding two weight tensors and a non-leaf flag."""
    address: str
    down: Any
    up: Any
    active: bool


def _make_genome(offset: float) -> Genome:
    return Genome([
        TensorGene("l1", torch.arange(6.0).reshape(2, 3) + offset, torch.arange(4.0).reshape(4, 1) + offset, True),
        TensorGene("l2", torch.arange(3.0).reshape(1, 3) + offset, torch.arange(2.0).reshape(2, 1) + offset, False),
    ])


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestTensorStrand(unittest.TestCase):

    def test_flatten_strand_matches_flatten_hierarchy(self) -> None:
        genome = _make_genome(0.0)
        strand, treedef = flatten_strand(genome)
        leaves, list_treedef = flatten_hierarchy(genome)

        self.assertTrue(is_tensor_strand(strand))
        self.assertEqual(strand.tolist(), leaves)
        self.assertEqual(treedef.total_leaves, list_treedef.total_leaves)

    def test_unflatten_strand_round_trip_owns_storage(self) -> None:
        genome = _make_genome(0.0)
        strand, treedef = flatten_strand(genome)
        rebuilt = unflatten_hierarchy(strand, treedef)

        self.assertEqual(rebuilt[1].address, "l2")
        self.assertFalse(rebuilt[1].active)
        self.assertTrue(torch.equal(rebuilt[0].down, genome[0].down))
        self.assertTrue(torch.equal(rebuilt[1].up, genome[1].up))
        # Each rebuilt tensor must own its memory (e.g. safetensors rejects shared storage)
        self.assertNotEqual(rebuilt[0].down.untyped_storage().data_ptr(), strand.untyped_storage().data_ptr())

    def test_flatten_strand_falls_back_to_list(self) -> None:
        mixed_dtype = Genome([torch.ones(2), torch.ones(2, dtype=torch.float64)])
        mixed_leaves = Genome([torch.ones(2), 3.0])

        for genome in (mixed_dtype, mixed_leaves):
            leaves, _ = flatten_strand(genome)
            self.assertEqual(leaves, flatten_hierarchy(genome)[0])

    def test_n_point_crossover_on_strand(self) -> None:
        child_a, child_b = NPointCrossover(cut_points=[3, 12])(_make_genome(0.0), _make_genome(100.0))

        # l1.down holds strand positions 0-5: cut at 3 switches child A to parent B mid-tensor
        self.assertEqual(child_a[0].down.tolist(), [[0.0, 1.0, 2.0], [103.0, 104.0, 105.0]])
        self.assertEqual(child_b[0].down.tolist(), [[100.0, 101.0, 102.0], [3.0, 4.0, 5.0]])
        # l2.down holds positions 10-12: cut at 12 switches child A back to parent A
        self.assertEqual(child_a[1].down.tolist(), [[100.0, 101.0, 2.0]])
        self.assertTrue(child_a[0].active)
        self.assertFalse(child_b[1].active)

    def test_two_point_crossover_matches_list_path(self) -> None:
        p1, p2 = _make_genome(0.0), _make_genome(100.0)
        with mock.patch("random.sample", return_value=[4, 9]):
            child_a, _ = TwoPointCrossover()(p1, p2)

        leaves_a, _ = flatten_hierarchy(p1)
        leaves_b, _ = flatten_hierarchy(p2)
        expected = leaves_a[:4] + leaves_b[4:9] + leaves_a[9:]
        self.assertEqual(flatten_hierarchy(child_a)[0], expected)

    def test_uniform_crossover_on_strand_is_complementary(self) -> None:
        p1, p2 = _make_genome(0.0), _make_genome(100.0)
        child_a, child_b = UniformCrossover(swap_prob=0.5)(p1, p2)

        leaves = [flatten_hierarchy(g)[0] for g in (p1, p2, child_a, child_b)]
        for a, b, x, y in zip(*leaves):
            self.assertIn((x, y), [(a, b), (b, a)])

    def test_sequence_mutations_on_strand(self) -> None:
        genome = _make_genome(0.0)
        leaves, _ = flatten_hierarchy(genome)

        with mock.patch("random.sample", return_value=[2, 7]):
            inverted = InversionMutation()(genome)
        self.assertEqual(flatten_hierarchy(inverted)[0], leaves[:2] + leaves[2:7][::-1] + leaves[7:])

        with mock.patch("random.sample", return_value=[2, 5]), mock.patch("random.shuffle", side_effect=lambda x: x.reverse()):
            scrambled = ScrambleMutation()(genome)
        self.assertEqual(flatten_hierarchy(scrambled)[0], leaves[:2] + leaves[2:5][::-1] + leaves[5:])

    def test_arithmetic_crossover_on_strand(self) -> None:
        p1, p2 = _make_genome(0.0), _make_genome(100.0)
        child_a, child_b = ArithmeticCrossover(alpha=0.25)(p1, p2)

        leaves_a, leaves_b = flatten_hierarchy(p1)[0], flatten_hierarchy(p2)[0]
        self.assertEqual(flatten_hierarchy(child_a)[0], [0.25 * x + 0.75 * y for x, y in zip(leaves_a, leaves_b)])
        self.assertEqual(flatten_hierarchy(child_b)[0], [0.75 * x + 0.25 * y for x, y in zip(leaves_a, leaves_b)])

    def test_gaussian_mutation_on_strand(self) -> None:
        genome = _make_genome(0.0)
        n = len(flatten_hierarchy(genome)[0])
        noise = torch.full((n,), 0.5)
        mask = torch.tensor([0.0, 1.0] * (n // 2) + [0.0] * (n % 2))

        with mock.patch("torch.randn_like", return_value=noise), mock.patch("torch.rand", return_value=mask):
            mutated = GaussianMutation(sigma=2.0, mutation_rate=0.5, bounds=(-10.0, 5.0))(genome)

        expected = [
            min(5.0, x + 1.0) if i % 2 == 0 else x
            for i, x in enumerate(flatten_hierarchy(genome)[0])
        ]
        self.assertEqual(flatten_hierarchy(mutated)[0], expected)
        self.assertTrue(mutated[0].active)

    def test_value_operators_preserve_tensor_dtypes(self) -> None:
        genome = Genome([torch.ones(4, dtype=torch.bfloat16), torch.ones(2, dtype=torch.bfloat16)])
        self.assertEqual(GaussianMutation(sigma=0.1)(genome)[0].dtype, torch.bfloat16)

        ints_a = Genome([torch.tensor([10, 20])])
        ints_b = Genome([torch.tensor([30, 41])])
        child_a, _ = ArithmeticCrossover(alpha=0.5)(ints_a, ints_b)
        self.assertEqual(child_a[0].dtype, torch.int64)
        self.assertEqual(child_a[0].tolist(), [20, 30])

    def test_value_operators_reject_bool_tensors(self) -> None:
        with self.assertRaises(TypeError):
            GaussianMutation()(Genome([torch.ones(3, dtype=torch.bool)]))
        with self.assertRaises(TypeError):
            SimulatedBinaryCrossover()(Genome([torch.ones(3, dtype=torch.bool)]), Genome([torch.zeros(3, dtype=torch.bool)]))


if __name__ == "__main__":
    unittest.main()
