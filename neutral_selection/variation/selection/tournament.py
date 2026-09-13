from __future__ import annotations

import random
from typing import Sequence, Union
from neutral_selection.representation.individual import Individual
from neutral_selection.representation.population import Population
from .base import SelectionStrategy, _extract_individuals, _validate_fitnesses, _validate_k


class TournamentSelection(SelectionStrategy):
    """
    Tournament Selection strategy.

    For each selection slot, samples `tournament_size` individuals randomly from the population.
    The fittest individual in the tournament wins with probability `winner_prob`.
    If winner_prob < 1.0 and the fittest does not win, a random contestant from the remaining
    tournament pool is chosen.

    Tournament selection is invariant to monotonic fitness scaling, naturally supports negative
    fitness values, and allows tuning selection pressure via `tournament_size`.
    """

    def __init__(
        self,
        tournament_size: int = 2,
        winner_prob: float = 1.0,
        minimize: bool = False,
    ) -> None:
        if type(tournament_size) is not int or tournament_size < 1:
            raise ValueError(f"tournament_size must be an integer >= 1, got {tournament_size}.")
        if not isinstance(winner_prob, (int, float)) or isinstance(winner_prob, bool):
            raise TypeError(f"winner_prob must be a float or int, got {type(winner_prob).__name__}.")
        if not (0.0 < winner_prob <= 1.0):
            raise ValueError(f"winner_prob must be in (0.0, 1.0], got {winner_prob}.")
        if not isinstance(minimize, bool):
            raise TypeError(f"minimize must be a boolean, got {type(minimize).__name__}.")

        self.tournament_size = tournament_size
        self.winner_prob = float(winner_prob)
        self.minimize = minimize

    def select(
        self,
        population: Union[Population, Sequence[Individual]],
        k: int = 1,
    ) -> list[Individual]:
        inds = _extract_individuals(population)
        _validate_fitnesses(inds)
        _validate_k(k)

        if len(inds) < self.tournament_size:
            raise ValueError(
                f"tournament_size ({self.tournament_size}) cannot be larger than population size ({len(inds)})."
            )

        selected: list[Individual] = []
        for _ in range(k):
            participants = random.sample(inds, self.tournament_size)
            participants.sort(
                key=lambda ind: ind.fitness,  # type: ignore[return-value]
                reverse=not self.minimize,
            )

            if self.winner_prob >= 1.0 or random.random() < self.winner_prob or len(participants) == 1:
                selected.append(participants[0])
            else:
                selected.append(random.choice(participants[1:]))

        return selected
