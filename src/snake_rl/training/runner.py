from __future__ import annotations

from dataclasses import dataclass
import random

from ..agents.base import Agent
from ..environment.snake_env import SnakeEnv


@dataclass(slots=True)
class EpisodeResult:
    reward: float
    score: int
    steps: int


def run_episode(env: SnakeEnv, agent: Agent, train: bool = True, seed: int | None = None) -> EpisodeResult:
    observation = env.reset(seed=seed)
    total_reward = 0.0
    done = False

    while not done:
        legal_actions = env.legal_actions()
        action = agent.act(observation, legal_actions=legal_actions, explore=train)
        result = env.step(action)

        if train:
            agent.observe(observation, action, result.reward, result.observation, result.done)

        observation = result.observation
        total_reward += result.reward
        done = result.done

    if train:
        agent.end_episode()
    return EpisodeResult(reward=total_reward, score=env.score, steps=env.steps)


def build_episode_seeds(episodes: int, seed: int | None = None) -> list[int | None]:
    rng = random.Random(seed)
    if seed is None:
        return [None] * episodes
    return [rng.randrange(2**32 - 1) for _ in range(episodes)]


def train_agent(env: SnakeEnv, agent: Agent, episodes: int, seed: int | None = None, episode_seeds: list[int | None] | None = None) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    seeds = episode_seeds or build_episode_seeds(episodes, seed)

    for episode_seed in seeds:
        results.append(run_episode(env, agent, train=True, seed=episode_seed))

    return results


def evaluate_agent(env: SnakeEnv, agent: Agent, episodes: int, seed: int | None = None, episode_seeds: list[int | None] | None = None) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    seeds = episode_seeds or build_episode_seeds(episodes, seed)

    for episode_seed in seeds:
        results.append(run_episode(env, agent, train=False, seed=episode_seed))

    return results

