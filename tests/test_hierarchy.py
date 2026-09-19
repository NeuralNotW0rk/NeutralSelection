from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import unittest
import unittest.mock as mock

from neutral_selection.representation.genome import Genome, Segment
from neutral_selection.representation.hierarchy import (
    TreeDef,
    flatten_hierarchy,
    unflatten_hierarchy,
)
from neutral_selection.variation.recombination.n_point_crossover import (
    OnePointCrossover,
    TwoPointCrossover,
    NPointCrossover,
    RandomNPointCrossover,
)
from neutral_selection.variation.mutation.sequence_mutation import (
    InversionMutation,
    ScrambleMutation,
)


@dataclass
class MockGene:
    """Mock domain dataclass containing primitive metadata and sub-elements."""
    address: str
    weights: list[float]
    active: bool


class CustomProtocolItem:
    """Mock custom domain item implementing __hierarchical_flatten__ protocol."""
    def __init__(self, name: str, values: list[float]) -> None:
        self.name = name
        self.values = values

    def __hierarchical_flatten__(self, max_depth: int | None = None, current_depth: int = 0) -> tuple[list[float], dict]:
        return list(self.values), {"name": self.name}

    @classmethod
    def __hierarchical_unflatten__(cls, leaves: list[float], metadata: dict) -> "CustomProtocolItem":
        return cls(name=metadata["name"], values=leaves)


class TestHierarchyFramework(unittest.TestCase):

    def test_flatten_unflatten_flat_genome(self) -> None:
        g = Genome([1, 2, 3, 4, 5])
        leaves, treedef = flatten_hierarchy(g)
        self.assertEqual(leaves, [1, 2, 3, 4, 5])
        self.assertEqual(treedef.total_leaves, 5)

        reconstructed = unflatten_hierarchy(leaves, treedef)
        self.assertIsInstance(reconstructed, Genome)
        self.assertEqual(list(reconstructed), [1, 2, 3, 4, 5])

    def test_flatten_unflatten_nested_segments(self) -> None:
        seg1 = Segment("block_a", [10, 20])
        seg2 = Segment("block_b", [30, 40, 50])
        root = Genome([seg1, seg2])

        leaves, treedef = flatten_hierarchy(root)
        self.assertEqual(leaves, [10, 20, 30, 40, 50])
        self.assertEqual(treedef.total_leaves, 5)

        # Mutate leaves
        mutated_leaves = [1, 2, 3, 4, 5]
        reconstructed = unflatten_hierarchy(mutated_leaves, treedef)

        self.assertIsInstance(reconstructed, Genome)
        self.assertEqual(len(reconstructed), 2)
        self.assertIsInstance(reconstructed[0], Segment)
        self.assertEqual(reconstructed[0].key, "block_a")
        self.assertEqual(list(reconstructed[0]), [1, 2])
        self.assertIsInstance(reconstructed[1], Segment)
        self.assertEqual(reconstructed[1].key, "block_b")
        self.assertEqual(list(reconstructed[1]), [3, 4, 5])

    def test_flatten_unflatten_dataclass(self) -> None:
        gene1 = MockGene(address="layer1", weights=[1.0, 2.0], active=True)
        gene2 = MockGene(address="layer2", weights=[3.0, 4.0, 5.0], active=False)
        root = Genome([gene1, gene2])

        leaves, treedef = flatten_hierarchy(root)
        self.assertEqual(leaves, [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(treedef.total_leaves, 5)

        mutated_leaves = [10.0, 20.0, 30.0, 40.0, 50.0]
        reconstructed = unflatten_hierarchy(mutated_leaves, treedef)

        self.assertIsInstance(reconstructed, Genome)
        self.assertEqual(len(reconstructed), 2)
        self.assertIsInstance(reconstructed[0], MockGene)
        self.assertEqual(reconstructed[0].address, "layer1")
        self.assertEqual(reconstructed[0].weights, [10.0, 20.0])
        self.assertTrue(reconstructed[0].active)

        self.assertIsInstance(reconstructed[1], MockGene)
        self.assertEqual(reconstructed[1].address, "layer2")
        self.assertEqual(reconstructed[1].weights, [30.0, 40.0, 50.0])
        self.assertFalse(reconstructed[1].active)

    def test_flatten_with_max_depth(self) -> None:
        seg1 = Segment("block_a", [10, 20])
        seg2 = Segment("block_b", [30, 40, 50])
        root = Genome([seg1, seg2])

        # Depth 1: treats Segments as atomic leaves
        leaves, treedef = flatten_hierarchy(root, max_depth=1)
        self.assertEqual(len(leaves), 2)
        self.assertEqual(leaves[0], seg1)
        self.assertEqual(leaves[1], seg2)

        reconstructed = unflatten_hierarchy(leaves, treedef)
        self.assertIsInstance(reconstructed, Genome)
        self.assertEqual(len(reconstructed), 2)

    def test_custom_protocol_object(self) -> None:
        item1 = CustomProtocolItem("layer1", [1.0, 2.0])
        item2 = CustomProtocolItem("layer2", [3.0, 4.0, 5.0])
        root = Genome([item1, item2])

        leaves, treedef = flatten_hierarchy(root)
        self.assertEqual(leaves, [1.0, 2.0, 3.0, 4.0, 5.0])

        inverted_leaves = list(reversed(leaves))
        reconstructed = unflatten_hierarchy(inverted_leaves, treedef)

        self.assertIsInstance(reconstructed[0], CustomProtocolItem)
        self.assertEqual(reconstructed[0].name, "layer1")
        self.assertEqual(reconstructed[0].values, [5.0, 4.0])

        self.assertIsInstance(reconstructed[1], CustomProtocolItem)
        self.assertEqual(reconstructed[1].name, "layer2")
        self.assertEqual(reconstructed[1].values, [3.0, 2.0, 1.0])

    def test_n_point_crossover_implicit_hierarchical(self) -> None:
        """Verify NPointCrossover implicitly operates across nested segments."""
        p1 = Genome([Segment("s1", [1, 2, 3]), Segment("s2", [4, 5, 6])])
        p2 = Genome([Segment("s1", [10, 20, 30]), Segment("s2", [40, 50, 60])])

        # Cut at linear index 4: p1[:4] = [1, 2, 3, 4], p2[4:] = [50, 60]
        crossover = NPointCrossover(cut_points=[4])
        child1, child2 = crossover(p1, p2)

        self.assertIsInstance(child1, Genome)
        self.assertEqual(list(child1[0]), [1, 2, 3])
        self.assertEqual(list(child1[1]), [4, 50, 60])

        self.assertEqual(list(child2[0]), [10, 20, 30])
        self.assertEqual(list(child2[1]), [40, 5, 6])

    def test_two_point_crossover_implicit_hierarchical(self) -> None:
        """Verify TwoPointCrossover implicitly operates across dataclass genome items."""
        p1 = Genome([
            MockGene("l1", [1.0, 2.0], True),
            MockGene("l2", [3.0, 4.0], True),
        ])
        p2 = Genome([
            MockGene("l1", [10.0, 20.0], False),
            MockGene("l2", [30.0, 40.0], False),
        ])

        # Cuts at [1, 3] -> middle segment [2.0, 3.0] from P1 swapped with [20.0, 30.0] from P2
        # P1 flat: [1.0, 2.0, 3.0, 4.0]
        # P2 flat: [10.0, 20.0, 30.0, 40.0]
        # Child 1: [1.0, 20.0, 30.0, 4.0]
        crossover = TwoPointCrossover(cut_points=(1, 3))
        c1, c2 = crossover(p1, p2)

        self.assertEqual(c1[0].weights, [1.0, 20.0])
        self.assertEqual(c1[1].weights, [30.0, 4.0])
        self.assertEqual(c2[0].weights, [10.0, 2.0])
        self.assertEqual(c2[1].weights, [3.0, 40.0])

    def test_inversion_mutation_implicit_hierarchical(self) -> None:
        """Verify InversionMutation implicitly operates across multi-tier structures."""
        seg1 = Segment("s1", [1, 2, 3])
        seg2 = Segment("s2", [4, 5, 6])
        root = Genome([seg1, seg2])

        mut = InversionMutation()

        # Mock random cut points to reverse [2:5] -> [3, 4, 5] -> reversed is [5, 4, 3]
        # Original flat: [1, 2, 3, 4, 5, 6]
        # Inverted flat: [1, 2, 5, 4, 3, 6]
        with mock.patch("random.sample", return_value=[2, 5]):
            mutated = mut(root)

        self.assertEqual(list(mutated[0]), [1, 2, 5])
        self.assertEqual(list(mutated[1]), [4, 3, 6])

    def test_gaussian_mutation_implicit_hierarchical(self) -> None:
        """Verify GaussianMutation perturbs all numeric leaves in composite structures."""
        from neutral_selection.variation.mutation.real_mutation import GaussianMutation

        root = Genome([
            MockGene("l1", [1.0, 2.0], True),
            MockGene("l2", [3.0, 4.0], False),
        ])

        mut = GaussianMutation(sigma=0.5, mutation_rate=1.0)
        with mock.patch("random.gauss", return_value=0.5):
            mutated = mut(root)

        self.assertEqual(mutated[0].weights, [1.5, 2.5])
        self.assertEqual(mutated[1].weights, [3.5, 4.5])
        self.assertTrue(mutated[0].active)
        self.assertFalse(mutated[1].active)

    def test_blend_crossover_implicit_hierarchical(self) -> None:
        """Verify BlendCrossover operates elementwise across numeric leaves in composite structures."""
        from neutral_selection.variation.recombination.real_crossover import BlendCrossover

        p1 = Genome([MockGene("l1", [10.0, 20.0], True)])
        p2 = Genome([MockGene("l1", [30.0, 40.0], False)])

        bx = BlendCrossover(alpha=0.0)
        with mock.patch("random.uniform", side_effect=[15.0, 15.0, 25.0, 25.0]):
            c1, c2 = bx(p1, p2)

        self.assertEqual(c1[0].weights, [15.0, 25.0])
        self.assertEqual(c2[0].weights, [15.0, 25.0])


if __name__ == "__main__":
    unittest.main()
