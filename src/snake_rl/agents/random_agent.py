from __future__ import annotations

from typing import Any

import random

from .base import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def act(self, observation: dict[str, Any], legal_actions: list[int] | None = None, explore: bool = True) -> int:
        choices = legal_actions if legal_actions else [0, 1, 2, 3]
        return int(self.rng.choice(choices))

