from __future__ import annotations

from ..agents.dqn_agent import DQNAgent
from ..agents import HAS_TORCH
from ..config import ExperimentConfig, build_dqn_snake_config
from pathlib import Path
from ..environment.snake_env import SnakeEnv
from ..training.runner import build_episode_seeds, evaluate_agent, run_episode


def main() -> int:
    if not HAS_TORCH or not DQNAgent.available():
        print("PyTorch is not installed in this environment. Install the 'torch' extra to run DQN training.")
        return 1

    snake_config = build_dqn_snake_config()
    experiment = ExperimentConfig(episodes=3000, evaluation_episodes=30, seed=42)
    
    print(f"Starting DQN experiment...")
    print(f"  Episodes: {experiment.episodes}")
    print(f"  Grid: {snake_config.width}x{snake_config.height}")
    print(f"  Reward shaping: {snake_config.reward_shaping}")
    print(f"  Seed: {experiment.seed}\n")
    
    env = SnakeEnv(snake_config, seed=experiment.seed)
    agent = DQNAgent(
        hidden_sizes=(128, 128),
        gamma=0.95,
        lr=1e-4,
        batch_size=64,
        memory_size=50000,
        epsilon=1.0,
        epsilon_decay=0.9995,
        min_epsilon=0.1,
        target_update_frequency=200,
        learning_starts=1000,
        train_frequency=4,
        max_grad_norm=10.0,
        seed=experiment.seed,
        state_mode="features",
    )
    train_seeds = build_episode_seeds(experiment.episodes, experiment.seed)
    eval_seeds = build_episode_seeds(experiment.evaluation_episodes, (experiment.seed or 0) + 1)
    assets_dir = Path(__file__).resolve().parents[3] / "report_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = assets_dir / "dqn_best_checkpoint.pth"
    checkpoint_metadata = {
        "grid_width": snake_config.width,
        "grid_height": snake_config.height,
        "max_steps": snake_config.max_steps,
        "state_mode": agent.state_mode,
        "reward_shaping": snake_config.reward_shaping,
        "food_reward": snake_config.food_reward,
        "death_reward": snake_config.death_reward,
        "step_reward": snake_config.step_reward,
        "training_episodes": experiment.episodes,
        "seed": experiment.seed,
    }

    # Train with logging every 100 episodes
    print("Training DQN agent...")
    train_results, best_eval_score, best_eval_episode = _train_with_logging(
        env,
        agent,
        train_seeds,
        eval_seeds,
        experiment.evaluation_episodes,
        best_ckpt,
        checkpoint_metadata,
        experiment.seed,
    )
    
    print("\nEvaluating DQN agent...")
    eval_results = evaluate_agent(env, agent, episodes=experiment.evaluation_episodes, seed=experiment.seed, episode_seeds=eval_seeds)

    train_average_score = sum(result.score for result in train_results) / len(train_results)
    eval_average_score = sum(result.score for result in eval_results) / len(eval_results) if eval_results else 0.0
    eval_average_reward = sum(result.reward for result in eval_results) / len(eval_results) if eval_results else 0.0

    print(f"\nTraining complete.")
    print(
        f"  Train avg score: {train_average_score:.2f} | Eval avg score: {eval_average_score:.2f} | Eval avg reward: {eval_average_reward:.2f}"
    )
    print(f"  Best eval score during training: {best_eval_score:.2f} at episode {best_eval_episode}")
    print(f"  Final epsilon: {agent.epsilon:.3f} | Training steps: {agent.training_steps}")

    print(f"Best DQN checkpoint saved to {best_ckpt}")

    return 0


def _train_with_logging(
    env: SnakeEnv,
    agent: DQNAgent,
    train_seeds: list[int | None],
    eval_seeds: list[int | None],
    evaluation_episodes: int,
    best_ckpt: Path,
    checkpoint_metadata: dict[str, object],
    seed: int | None = None,
) -> tuple[list, float, int]:
    from ..training.runner import EpisodeResult
    
    results: list[EpisodeResult] = []
    best_eval_score = float("-inf")
    best_eval_episode = 0
    
    for episode_num, episode_seed in enumerate(train_seeds, 1):
        result = run_episode(env, agent, train=True, seed=episode_seed)
        results.append(result)
        
        # Log every 100 episodes
        if episode_num % 100 == 0:
            window_results = results[-100:]
            avg_score = sum(r.score for r in window_results) / len(window_results)
            avg_reward = sum(r.reward for r in window_results) / len(window_results)
            eval_env = SnakeEnv(env.config, seed=seed)
            eval_results = evaluate_agent(eval_env, agent, episodes=evaluation_episodes, seed=seed, episode_seeds=eval_seeds)
            eval_avg_score = sum(r.score for r in eval_results) / len(eval_results) if eval_results else 0.0
            eval_avg_reward = sum(r.reward for r in eval_results) / len(eval_results) if eval_results else 0.0
            print(
                f"  Episode {episode_num:4d}/{len(train_seeds)} | Train avg score: {avg_score:6.2f} | Train avg reward: {avg_reward:7.3f} | Eval avg score: {eval_avg_score:6.2f} | Eval avg reward: {eval_avg_reward:7.3f} | Epsilon: {agent.epsilon:.3f}"
            )

            if eval_avg_score > best_eval_score:
                best_eval_score = eval_avg_score
                best_eval_episode = episode_num
                best_metadata = dict(checkpoint_metadata)
                best_metadata.update(
                    {
                        "best_eval_avg_score": float(best_eval_score),
                        "best_eval_episode": int(best_eval_episode),
                        "seed": seed,
                    }
                )
                agent.save(str(best_ckpt), metadata=best_metadata)
                print(f"    New best checkpoint saved to {best_ckpt}")

    return results, best_eval_score, best_eval_episode


if __name__ == "__main__":
    raise SystemExit(main())
