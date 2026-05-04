from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def act(self, observation: dict[str, Any], legal_actions: list[int] | None = None, explore: bool = True) -> int:
        raise NotImplementedError

    def observe(
        self,
        state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> None:
        return None

    def end_episode(self) -> None:
        return None

