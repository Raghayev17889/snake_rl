from __future__ import annotations

from collections import defaultdict
from typing import Any

import random

from .base import Agent


class QLearningAgent(Agent):
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.rng = random.Random(seed)
        self.q_table: dict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.last_state: tuple[int, ...] | None = None
        self.last_action: int | None = None

    def act(self, observation: dict[str, Any], legal_actions: list[int] | None = None, explore: bool = True) -> int:
        state = self._state_key(observation)
        self.last_state = state

        if explore and float(self.rng.random()) < self.epsilon:
            action_space = legal_actions if legal_actions else [0, 1, 2, 3]
            action = int(self.rng.choice(action_space))
        else:
            q_values = list(self.q_table[state])
            if legal_actions:
                invalid_actions = [candidate for candidate in range(4) if candidate not in legal_actions]
                for invalid_action in invalid_actions:
                    q_values[invalid_action] = float("-inf")
            action = int(max(range(4), key=lambda candidate: q_values[candidate]))

        self.last_action = action
        return action

    def observe(self, state: dict[str, Any], action: int, reward: float, next_state: dict[str, Any], done: bool) -> None:
        current_key = self._state_key(state)
        next_key = self._state_key(next_state)

        current_q = self.q_table[current_key][action]
        if done:
            next_best = 0.0
        else:
            legal_next_actions = self._legal_actions_from_observation(next_state)
            next_best = max(self.q_table[next_key][candidate] for candidate in legal_next_actions)
        target = reward + self.gamma * next_best
        self.q_table[current_key][action] += self.alpha * (target - current_q)

    def end_episode(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.last_state = None
        self.last_action = None

    def _state_key(self, observation: dict[str, Any]) -> tuple[int, ...]:
        features = observation["features"]
        return tuple(int(value > 0.5) for value in features)

    def _legal_actions_from_observation(self, observation: dict[str, Any]) -> list[int]:
        from .state_utils import legal_actions_from_features

        return legal_actions_from_features(observation["features"])

