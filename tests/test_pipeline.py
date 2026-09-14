from __future__ import annotations

import unittest
from neutral_selection.representation.genome import Genome
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from neutral_selection.representation.lineage import Lineage
from neutral_selection.variation.selection import (
    TournamentSelection,
    TruncationSelection,
    ElitistSelection,
)
from neutral_selection.variation.recombination import (
    OnePointCrossover,
    UniformCrossover,
    RandomNPointCrossover,
)
from neutral_selection.variation.mutation import (
    UniformMutation,
    GaussianMutation,
)
from neutral_selection.replacement import (
    GenerationalReplacement,
    PlusReplacement,
)
from neutral_selection.pipeline import EvolutionPipeline, step
from neutral_selection.builders import (
    build_selection_strategy,
    build_crossover_strategy,
    build_mutation_strategy,
    build_replacement_strategy,
    build_pipeline,
)


def _make_ind(items: list[float], fitness: float | None = None, ind_id: str | None = None) -> Individual:
    metadata = {"id": ind_id} if ind_id else {}
    return Individual(Genome(list(items)), fitness=fitness, metadata=metadata)


class TestEvolutionPipeline(unittest.TestCase):

    def test_basic_step_and_lineage_tracking(self) -> None:
        p1 = _make_ind([1.0, 1.0, 1.0], fitness=10.0, ind_id="p1")
        p2 = _make_ind([2.0, 2.0, 2.0], fitness=20.0, ind_id="p2")
        parents = Population([p1, p2])

        selection_strat = TournamentSelection(tournament_size=2)
        crossover_strat = OnePointCrossover()
        mutation_strat = GaussianMutation(sigma=0.1)

        offspring_pop = step(
            parents=parents,
            selection_strategy=selection_strat,
            crossover_strategy=crossover_strat,
            mutation_strategy=mutation_strat,
            offspring_count=5,
            crossover_prob=1.0,
            mutation_prob=1.0,
            elitism=0,
            parent_ids=["p1", "p2"],
        )

        self.assertEqual(len(offspring_pop), 5)
        for ind in offspring_pop:
            self.assertIsInstance(ind, Individual)
            self.assertIsNotNone(ind.lineage)
            lineage = ind.lineage
            self.assertIsInstance(lineage, Lineage)
            self.assertTrue(lineage.crossover_applied)
            self.assertTrue(lineage.mutated)
            self.assertGreaterEqual(len(lineage.parent_ids), 1)

    def test_elitism_preservation(self) -> None:
        p1 = _make_ind([1.0, 1.0], fitness=10.0, ind_id="p1")
        p2 = _make_ind([2.0, 2.0], fitness=50.0, ind_id="p2")  # Highest
        p3 = _make_ind([3.0, 3.0], fitness=30.0, ind_id="p3")
        parents = [p1, p2, p3]

        pipeline = EvolutionPipeline(
            selection_strategy=TournamentSelection(tournament_size=2),
            crossover_strategy=UniformCrossover(swap_prob=0.5),
            mutation_strategy=GaussianMutation(sigma=0.5),
            elitism=1,
        )

        offspring_pop = pipeline(parents, offspring_count=4, parent_ids=["p1", "p2", "p3"])
        self.assertEqual(len(offspring_pop), 4)

        # First individual must be elite (p2)
        elite_ind = offspring_pop[0]
        self.assertEqual(elite_ind.fitness, 50.0)
        self.assertEqual(list(elite_ind.genotype), [2.0, 2.0])
        self.assertIsNotNone(elite_ind.lineage)
        self.assertEqual(elite_ind.lineage.parent_ids, ["p2"])
        self.assertFalse(elite_ind.lineage.crossover_applied)
        self.assertFalse(elite_ind.lineage.mutated)

    def test_step_probabilities(self) -> None:
        p1 = _make_ind([1.0, 1.0], fitness=10.0, ind_id="p1")
        p2 = _make_ind([2.0, 2.0], fitness=20.0, ind_id="p2")
        parents = [p1, p2]

        # No crossover, no mutation
        offspring_pop = step(
            parents=parents,
            selection_strategy=TournamentSelection(tournament_size=2),
            crossover_strategy=OnePointCrossover(),
            mutation_strategy=GaussianMutation(sigma=0.1),
            crossover_prob=0.0,
            mutation_prob=0.0,
            offspring_count=2,
            parent_ids=["p1", "p2"],
        )

        self.assertEqual(len(offspring_pop), 2)
        for ind in offspring_pop:
            self.assertIsNotNone(ind.lineage)
            self.assertFalse(ind.lineage.crossover_applied)
            self.assertFalse(ind.lineage.mutated)
            self.assertEqual(len(ind.lineage.parent_ids), 1)

    def test_replacement_strategy_integration(self) -> None:
        p1 = _make_ind([1.0], fitness=10.0)
        p2 = _make_ind([2.0], fitness=50.0)
        parents = Population([p1, p2])

        # Use PlusReplacement with evaluate_fn
        pipeline = EvolutionPipeline(
            selection_strategy=TournamentSelection(tournament_size=2),
            evaluate_fn=lambda ind: sum(ind.genotype),
            replacement_strategy=PlusReplacement(),
        )

        offspring_pop = pipeline.step(parents, offspring_count=2)
        self.assertEqual(len(offspring_pop), 2)
        self.assertIsNotNone(offspring_pop[0].fitness)

    def test_strategy_config_builders(self) -> None:
        # Selection
        tourn = build_selection_strategy({"type": "tournament", "tournament_size": 4})
        self.assertIsInstance(tourn, TournamentSelection)
        self.assertEqual(tourn.tournament_size, 4)

        trunc = build_selection_strategy({"type": "truncation", "top_k": 3})
        self.assertIsInstance(trunc, TruncationSelection)
        self.assertEqual(trunc.top_k, 3)

        # Crossover
        crossover = build_crossover_strategy({"type": "random_n_point", "num_cut_points": 2})
        self.assertIsInstance(crossover, RandomNPointCrossover)
        self.assertEqual(crossover.num_cut_points, 2)

        # Mutation
        mut = build_mutation_strategy({"type": "gaussian", "sigma": 0.05, "mutation_rate": 0.5})
        self.assertIsInstance(mut, GaussianMutation)
        self.assertEqual(mut.sigma, 0.05)
        self.assertEqual(mut.mutation_rate, 0.5)

        # Replacement
        rep = build_replacement_strategy({"type": "generational", "num_elites": 2})
        self.assertEqual(rep.num_elites, 2)

        # Pipeline builder
        pipeline = build_pipeline({
            "selection": {"type": "tournament", "tournament_size": 2},
            "crossover": {"type": "uniform", "swap_prob": 0.5},
            "mutation": {"type": "gaussian", "sigma": 0.1},
            "crossover_prob": 0.8,
            "elitism": 1,
        })
        self.assertIsInstance(pipeline, EvolutionPipeline)
        self.assertEqual(pipeline.crossover_prob, 0.8)
        self.assertEqual(pipeline.elitism, 1)

    def test_fail_fast_validations(self) -> None:
        p1 = _make_ind([1.0], fitness=10.0)
        sel = TournamentSelection(tournament_size=2)

        # Empty parents
        with self.assertRaises(ValueError):
            step([], selection_strategy=sel, offspring_count=2)

        # Non-positive count
        with self.assertRaises(ValueError):
            step([p1], selection_strategy=sel, offspring_count=0)

        # Invalid crossover_prob
        with self.assertRaises(ValueError):
            EvolutionPipeline(selection_strategy=sel, crossover_prob=1.5)

        # Invalid mutation_prob
        with self.assertRaises(ValueError):
            EvolutionPipeline(selection_strategy=sel, mutation_prob=-0.1)

        # Invalid elitism > pop size
        with self.assertRaises(ValueError):
            step([p1], selection_strategy=sel, elitism=5)

        # Mismatched parent_ids length
        with self.assertRaises(ValueError):
            step([p1], selection_strategy=sel, parent_ids=["p1", "p2"])

        # Invalid strategy types
        with self.assertRaises(TypeError):
            EvolutionPipeline(selection_strategy="not-a-strategy")  # type: ignore

        # Conflicting custom replacement strategy and elitism
        with self.assertRaises(ValueError):
            EvolutionPipeline(
                selection_strategy=sel,
                replacement_strategy=PlusReplacement(),
                elitism=1,
            )


if __name__ == "__main__":
    unittest.main()
