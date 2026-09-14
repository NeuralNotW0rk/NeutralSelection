import unittest
from neutral_selection.registry import (
    register_selection,
    get_selection_strategy,
    list_selection_strategies,
    register_crossover,
    get_crossover_strategy,
    list_crossover_strategies,
    register_mutation,
    get_mutation_strategy,
    list_mutation_strategies,
    register_replacement,
    get_replacement_strategy,
    list_replacement_strategies,
)
from neutral_selection.variation.selection.base import SelectionStrategy
from neutral_selection.variation.recombination.base import RecombinationStrategy
from neutral_selection.variation.mutation.base import MutationStrategy
from neutral_selection.replacement.base import ReplacementStrategy
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.genome import Genome
from neutral_selection.builders import (
    build_selection_strategy,
    build_crossover_strategy,
    build_mutation_strategy,
    build_replacement_strategy,
    build_pipeline,
)


class TestRegistry(unittest.TestCase):
    def test_list_strategies(self):
        sel_list = list_selection_strategies()
        self.assertIn("tournament", sel_list)
        self.assertIn("truncation", sel_list)
        self.assertIn("linear_rank", sel_list)

        cross_list = list_crossover_strategies()
        self.assertIn("one_point", cross_list)
        self.assertIn("two_point", cross_list)
        self.assertIn("simulated_binary", cross_list)

        mut_list = list_mutation_strategies()
        self.assertIn("gaussian", mut_list)
        self.assertIn("uniform_real", mut_list)
        self.assertIn("bit_flip", mut_list)

        rep_list = list_replacement_strategies()
        self.assertIn("generational", rep_list)
        self.assertIn("plus", rep_list)
        self.assertIn("steady_state", rep_list)

    def test_get_strategy_success(self):
        from neutral_selection.variation.selection import TournamentSelection
        self.assertIs(get_selection_strategy("tournament"), TournamentSelection)
        self.assertIs(get_selection_strategy("TOURNAMENT"), TournamentSelection)
        self.assertIs(get_selection_strategy("tourn"), TournamentSelection)

        from neutral_selection.variation.recombination import OnePointCrossover
        self.assertIs(get_crossover_strategy("one_point"), OnePointCrossover)

        from neutral_selection.variation.mutation import GaussianMutation
        self.assertIs(get_mutation_strategy("gaussian"), GaussianMutation)
        self.assertIs(get_mutation_strategy("normal"), GaussianMutation)

        from neutral_selection.replacement import GenerationalReplacement
        self.assertIs(get_replacement_strategy("generational"), GenerationalReplacement)

    def test_get_strategy_missing(self):
        with self.assertRaises(KeyError):
            get_selection_strategy("non_existent_selection")
        with self.assertRaises(KeyError):
            get_crossover_strategy("non_existent_crossover")
        with self.assertRaises(KeyError):
            get_mutation_strategy("non_existent_mutation")
        with self.assertRaises(KeyError):
            get_replacement_strategy("non_existent_replacement")

    def test_custom_registration(self):
        @register_selection("custom_dummy_selection")
        class CustomSelection(SelectionStrategy):
            def __init__(self, multiplier: float = 2.0):
                self.multiplier = multiplier

            def select(self, population, k=1):
                return list(population)[:k]

        self.assertIs(get_selection_strategy("custom_dummy_selection"), CustomSelection)

        built = build_selection_strategy({"type": "custom_dummy_selection", "multiplier": 3.5})
        self.assertIsInstance(built, CustomSelection)
        self.assertEqual(built.multiplier, 3.5)

        # Test duplicate registration error
        with self.assertRaises(ValueError):
            @register_selection("custom_dummy_selection")
            class DuplicateSelection(SelectionStrategy):
                def select(self, population, k=1):
                    return []

        # Test invalid subclass error
        with self.assertRaises(TypeError):
            @register_selection("invalid_type_class")
            class NotAStrategy:
                pass


if __name__ == "__main__":
    unittest.main()
