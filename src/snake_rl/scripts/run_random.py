from __future__ import annotations

from ..agents.random_agent import RandomAgent
from ..config import ExperimentConfig, SnakeConfig
from ..environment.snake_env import SnakeEnv
from ..training.runner import evaluate_agent


def main() -> None:
    snake_config = SnakeConfig()
    experiment = ExperimentConfig(episodes=10, seed=42)
    print(f"Starting Random baseline... Episodes: {experiment.episodes} | Grid: {snake_config.width}x{snake_config.height}")
    env = SnakeEnv(snake_config, seed=experiment.seed)
    agent = RandomAgent(seed=experiment.seed)
    results = evaluate_agent(env, agent, episodes=experiment.episodes, seed=experiment.seed)

    average_score = sum(result.score for result in results) / len(results)
    average_reward = sum(result.reward for result in results) / len(results)
    print("Finished Random baseline.")
    print(f"episodes={len(results)} average_score={average_score:.2f} average_reward={average_reward:.2f}")


if __name__ == "__main__":
    main()
