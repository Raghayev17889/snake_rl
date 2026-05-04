from __future__ import annotations

from ..agents.q_learning_agent import QLearningAgent
from ..config import ExperimentConfig, SnakeConfig
from ..environment.snake_env import SnakeEnv
from ..training.runner import evaluate_agent, train_agent


def main() -> None:
    snake_config = SnakeConfig()
    experiment = ExperimentConfig(episodes=200, seed=42)
    print(f"Starting Q-Learning experiment... Episodes: {experiment.episodes} | Grid: {snake_config.width}x{snake_config.height} | Reward shaping: {snake_config.reward_shaping}")
    env = SnakeEnv(snake_config, seed=experiment.seed)
    agent = QLearningAgent(seed=experiment.seed)

    train_results = train_agent(env, agent, episodes=experiment.episodes, seed=experiment.seed)
    eval_results = evaluate_agent(env, agent, episodes=20, seed=experiment.seed)

    train_average_score = sum(result.score for result in train_results) / len(train_results)
    eval_average_score = sum(result.score for result in eval_results) / len(eval_results)
    eval_average_reward = sum(result.reward for result in eval_results) / len(eval_results)

    print("Finished Q-Learning experiment.")
    print(
        f"Train avg score: {train_average_score:.2f} | Eval avg score: {eval_average_score:.2f} | Eval avg reward: {eval_average_reward:.2f} | epsilon={agent.epsilon:.3f}"
    )


if __name__ == "__main__":
    main()
