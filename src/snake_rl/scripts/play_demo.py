"""
Simple demo/visualization script for the Snake environment.

Supports:
  - Random agent (baseline)
  - Trained Q-Learning agent (if checkpoint exists)
  - Trained DQN agent (if checkpoint exists and PyTorch is available)

Usage:
  python -m snake_rl.scripts.play_demo --agent random
  python -m snake_rl.scripts.play_demo --agent q_learning
  python -m snake_rl.scripts.play_demo --agent dqn
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from typing import Any

from ..config import SnakeConfig, build_dqn_snake_config
from ..environment.snake_env import ACTION_NAMES
from ..agents.random_agent import RandomAgent
from ..agents.q_learning_agent import QLearningAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Snake with different agents")
    parser.add_argument(
        "--agent",
        choices=["random", "q_learning", "dqn"],
        default="random",
        help="Agent to use for playing (default: random)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Delay between steps in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    args = parser.parse_args()

    if args.agent == "dqn":
        checkpoint_path = Path(__file__).resolve().parents[3] / "report_assets" / "dqn_best_checkpoint.pth"
        agent, env_config, _ = _load_dqn_checkpoint(checkpoint_path)
        if agent is None or env_config is None:
            env_config = build_dqn_snake_config()
            from ..agents.dqn_agent import DQNAgent

            agent = DQNAgent(state_mode="features")
        print(f"Using DQN checkpoint: {checkpoint_path}")
    else:
        env_config = SnakeConfig(max_steps=args.max_steps, reward_shaping=True)
        agent = _build_agent(args.agent)

    if agent is None:
        print(f"Error: Could not build '{args.agent}' agent.")
        sys.exit(1)

    print(f"\n=== Snake Demo: {args.agent.upper()} Agent ===\n")
    play_episode(env_config, agent, args.delay)


def play_episode(config: SnakeConfig, agent, delay: float) -> None:
    """Run a single episode with the given agent and render ASCII output."""
    from ..environment.snake_env import SnakeEnv

    env = SnakeEnv(config, seed=42)
    observation = env.reset()
    action_history: list[int] = []
    warned_cycle = False

    step = 0
    print(env.render_ascii())
    print(f"Step: {step}, Score: {env.score}, Food: {env.food}\n")

    while True:
        legal_actions = env.legal_actions()
        action = agent.act(observation, legal_actions=legal_actions, explore=False)
        action_history.append(action)
        if not warned_cycle and _has_repeated_cycle(action_history):
            print("Warning: agent appears stuck in a repeated action cycle.")
            warned_cycle = True

        result = env.step(action)
        observation = result.observation
        reward = result.reward
        done = result.done

        step += 1
        action_name = ACTION_NAMES.get(action, "unknown")

        print(f"\033[2J\033[H")  # Clear screen
        print(env.render_ascii())
        print(f"\nStep: {step}, Score: {env.score}, Reward: {reward:.3f}")
        print(f"Action: {action_name}, Snake Length: {len(env.snake)}")
        print(f"Food: {env.food}\n")

        if done:
            if env.steps >= config.max_steps:
                print(f"Episode ended: MAX STEPS REACHED ({config.max_steps} steps)")
            else:
                print(f"Episode ended: COLLISION (snake hit wall or itself)")
            print(f"Final Score: {env.score}, Total Steps: {env.steps}\n")
            break

        time.sleep(delay)


def _build_agent(agent_name: str):
    """Build and load the specified agent."""
    if agent_name == "random":
        return RandomAgent()

    elif agent_name == "q_learning":
        agent = QLearningAgent()
        checkpoint_path = Path(__file__).resolve().parents[3] / "report_assets" / "q_learning_checkpoint.pkl"
        if checkpoint_path.exists():
            agent.load(checkpoint_path)
            print(f"Loaded Q-Learning agent from {checkpoint_path}")
        else:
            print(f"Warning: Q-Learning checkpoint not found at {checkpoint_path}")
            print("Using untrained agent.")
        return agent

    elif agent_name == "dqn":
        try:
            from ..agents.dqn_agent import DQNAgent

            checkpoint_path = Path(__file__).resolve().parents[3] / "report_assets" / "dqn_best_checkpoint.pth"
            agent, _, _ = _load_dqn_checkpoint(checkpoint_path)
            if agent is not None:
                return agent

            print(f"Warning: DQN checkpoint not found at {checkpoint_path}")
            print("Using untrained DQN agent with state_mode='features'.")
            return DQNAgent(state_mode="features")
        except ImportError:
            print("Error: PyTorch not available. Install with: pip install -e .[torch]")
            return None

    return None


def _load_dqn_checkpoint(checkpoint_path: Path) -> tuple[Any, SnakeConfig | None, dict[str, Any] | None]:
    if not checkpoint_path.exists():
        return None, None, None

    try:
        torch = importlib.import_module("torch")
    except Exception:
        return None, None, None

    from ..agents.dqn_agent import DQNAgent

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_mode = payload.get("state_mode", "features")
    agent = DQNAgent(state_mode=state_mode)
    agent.load(str(checkpoint_path))

    env_config = SnakeConfig(
        width=int(payload.get("grid_width", 6)),
        height=int(payload.get("grid_height", 6)),
        max_steps=int(payload.get("max_steps", 100)),
        food_reward=float(payload.get("food_reward", 10.0)),
        death_reward=float(payload.get("death_reward", -10.0)),
        step_reward=float(payload.get("step_reward", -0.02)),
        reward_shaping=bool(payload.get("reward_shaping", True)),
    )

    print(f"Loaded DQN agent from {checkpoint_path}")
    print(f"  State mode: {state_mode}")
    print(f"  Grid: {env_config.width}x{env_config.height}")
    print(f"  Max steps: {env_config.max_steps}")
    print(f"  Reward shaping: {env_config.reward_shaping}")
    print(f"  Rewards: food={env_config.food_reward}, death={env_config.death_reward}, step={env_config.step_reward}")

    warnings: list[str] = []
    if payload.get("reward_shaping", True) != env_config.reward_shaping:
        warnings.append("Reward shaping mismatch in checkpoint metadata")
    if warnings:
        print("  Warnings:")
        for warning in warnings:
            print(f"      - {warning}")

    return agent, env_config, payload


def _has_repeated_cycle(action_history: list[int]) -> bool:
    if len(action_history) < 16:
        return False

    cycle = action_history[-4:]
    return (
        action_history[-8:-4] == cycle
        and action_history[-12:-8] == cycle
        and action_history[-16:-12] == cycle
    )


if __name__ == "__main__":
    main()
