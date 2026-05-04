from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SnakeConfig:
    width: int = 10
    height: int = 10
    max_steps: int = 200
    food_reward: float = 1.0
    death_reward: float = -1.0
    step_reward: float = -0.01
    shaping_closer_reward: float = 0.05
    shaping_farther_penalty: float = -0.02
    reward_shaping: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    episodes: int = 1500
    evaluation_episodes: int = 25
    seed: int | None = 42


def build_dqn_snake_config(max_steps: int = 100) -> SnakeConfig:
    return SnakeConfig(
        width=6,
        height=6,
        max_steps=max_steps,
        food_reward=10.0,
        death_reward=-10.0,
        step_reward=-0.02,
        shaping_closer_reward=0.2,
        shaping_farther_penalty=-0.2,
        reward_shaping=True,
    )
