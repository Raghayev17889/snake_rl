from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from math import sqrt

from .agents.q_learning_agent import QLearningAgent
from .agents.random_agent import RandomAgent
from .agents import DQNAgent, HAS_TORCH
from .config import ExperimentConfig, SnakeConfig, build_dqn_snake_config
from .environment.snake_env import SnakeEnv
from .training.runner import EpisodeResult, evaluate_agent, train_agent, build_episode_seeds


@dataclass(slots=True)
class ExperimentSeries:
    label: str
    train_results: list[EpisodeResult]
    eval_results: list[EpisodeResult]
    reward_shaping: bool


def run_experiments(experiment: ExperimentConfig | None = None) -> tuple[list[EpisodeResult], list[ExperimentSeries]]:
    config = experiment or ExperimentConfig()  # Uses defaults from ExperimentConfig (1500 episodes)
    # generate shared episode seeds for fair comparisons
    train_seeds = build_episode_seeds(config.episodes, config.seed)
    eval_seeds = build_episode_seeds(config.evaluation_episodes, config.seed)

    # Random baseline
    print("Training/evaluating Random agent...")
    random_env = SnakeEnv(SnakeConfig(reward_shaping=False), seed=config.seed)
    random_agent = RandomAgent(seed=config.seed)
    random_results = evaluate_agent(random_env, random_agent, episodes=config.evaluation_episodes, seed=config.seed, episode_seeds=eval_seeds)

    # Q-Learning baseline
    print("Training/evaluating Q-Learning agent...")
    q_learning_env = SnakeEnv(SnakeConfig(reward_shaping=False), seed=config.seed)
    q_learning_agent = QLearningAgent(seed=config.seed)
    q_learning_train = train_agent(q_learning_env, q_learning_agent, episodes=config.episodes, seed=config.seed, episode_seeds=train_seeds)
    q_learning_eval = evaluate_agent(q_learning_env, q_learning_agent, episodes=config.evaluation_episodes, seed=config.seed, episode_seeds=eval_seeds)

    # Q-Learning with reward shaping
    print("Training/evaluating Q-Learning + reward shaping...")
    shaping_env = SnakeEnv(SnakeConfig(reward_shaping=True), seed=config.seed)
    shaping_agent = QLearningAgent(seed=config.seed)
    shaping_train = train_agent(shaping_env, shaping_agent, episodes=config.episodes, seed=config.seed, episode_seeds=train_seeds)
    shaping_eval = evaluate_agent(shaping_env, shaping_agent, episodes=config.evaluation_episodes, seed=config.seed, episode_seeds=eval_seeds)

    # Q-Learning with larger learning rate
    print("Training/evaluating Q-Learning + shaping (alpha=0.2)...")
    shaping_lr_env = SnakeEnv(SnakeConfig(reward_shaping=True), seed=config.seed)
    shaping_lr_agent = QLearningAgent(alpha=0.2, seed=config.seed)
    shaping_lr_train = train_agent(shaping_lr_env, shaping_lr_agent, episodes=config.episodes, seed=config.seed, episode_seeds=train_seeds)
    shaping_lr_eval = evaluate_agent(shaping_lr_env, shaping_lr_agent, episodes=config.evaluation_episodes, seed=config.seed, episode_seeds=eval_seeds)

    # DQN experiments (if available) — run representation sweep
    dqn_series: list[ExperimentSeries] = []
    if HAS_TORCH and DQNAgent.available():
        print("Training/evaluating DQN representation sweep on 6x6 grid...")
        for mode, label in [("features", "DQN (features)"), ("features_grid", "DQN (features+grid)")]:
            dqn_env = SnakeEnv(build_dqn_snake_config(), seed=config.seed)
            dqn_agent = DQNAgent(seed=config.seed, state_mode=mode)
            dqn_train = train_agent(dqn_env, dqn_agent, episodes=config.episodes, seed=config.seed, episode_seeds=train_seeds)
            dqn_eval = evaluate_agent(dqn_env, dqn_agent, episodes=config.evaluation_episodes, seed=config.seed, episode_seeds=eval_seeds)
            dqn_series.append(ExperimentSeries(label=label, train_results=dqn_train, eval_results=dqn_eval, reward_shaping=True))

    series = [
        ExperimentSeries(
            label="Q-Learning",
            train_results=q_learning_train,
            eval_results=q_learning_eval,
            reward_shaping=False,
        ),
        ExperimentSeries(
            label="Q-Learning + Reward Shaping",
            train_results=shaping_train,
            eval_results=shaping_eval,
            reward_shaping=True,
        ),
        ExperimentSeries(
            label="Q-Learning + Shaping (α=0.2)",
            train_results=shaping_lr_train,
            eval_results=shaping_lr_eval,
            reward_shaping=True,
        ),
    ]

    series.extend(dqn_series)

    return random_results, series


def summarize_results(results: list[EpisodeResult]) -> dict[str, float]:
    n = len(results)
    rewards = [r.reward for r in results]
    scores = [r.score for r in results]
    steps = [r.steps for r in results]
    return {
        "n": n,
        "average_reward": mean(rewards),
        "reward_std": stdev(rewards) if n > 1 else 0.0,
        "average_score": mean(scores),
        "score_std": stdev(scores) if n > 1 else 0.0,
        "average_steps": mean(steps),
        "steps_std": stdev(steps) if n > 1 else 0.0,
        "best_score": max(scores),
    }
