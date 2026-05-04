from __future__ import annotations

import pytest

from snake_rl.agents.q_learning_agent import QLearningAgent
from snake_rl.config import SnakeConfig
from snake_rl.environment.snake_env import RIGHT, UP, SnakeEnv
from snake_rl.training.runner import evaluate_agent


def _observation_with_direction(direction: int) -> dict[str, object]:
    features = [0.0] * 11
    features[7 + direction] = 1.0
    return {
        "features": features,
        "grid": [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(3)],
    }


def test_evaluate_agent_does_not_decay_epsilon() -> None:
    env = SnakeEnv(SnakeConfig(width=6, height=6, max_steps=30), seed=7)
    agent = QLearningAgent(epsilon=0.9, epsilon_decay=0.5, min_epsilon=0.1, seed=7)

    before = agent.epsilon
    evaluate_agent(env, agent, episodes=3, seed=7)

    assert agent.epsilon == before


def test_tail_move_is_allowed_when_not_eating() -> None:
    env = SnakeEnv(SnakeConfig(width=5, height=5, max_steps=50), seed=1)
    env.reset(seed=1)

    env.snake = [(2, 2), (2, 1), (1, 1), (1, 2)]
    env.direction = UP
    env.food = (4, 4)

    result = env.step(UP)

    assert result.done is False
    assert env.snake[0] == (1, 2)


def test_q_learning_bootstrap_ignores_illegal_opposite_action() -> None:
    agent = QLearningAgent(alpha=1.0, gamma=1.0, epsilon=0.0, seed=0)
    state = _observation_with_direction(RIGHT)
    next_state = _observation_with_direction(RIGHT)

    current_key = agent._state_key(state)
    next_key = agent._state_key(next_state)

    agent.q_table[current_key] = [0.0, 0.0, 0.0, 0.0]
    agent.q_table[next_key] = [1.0, 5.0, 2.0, 100.0]

    agent.observe(state, action=0, reward=0.0, next_state=next_state, done=False)

    assert agent.q_table[current_key][0] == pytest.approx(5.0)


def test_dqn_observe_stores_legal_actions_for_next_state() -> None:
    torch = pytest.importorskip("torch")
    assert torch is not None

    from snake_rl.agents.dqn_agent import DQNAgent

    agent = DQNAgent(batch_size=64, seed=123)
    state = _observation_with_direction(RIGHT)
    next_state = _observation_with_direction(RIGHT)

    agent.observe(state, action=1, reward=0.1, next_state=next_state, done=False)

    stored = agent.memory.buffer[-1]
    assert stored.next_legal_actions == [0, 1, 2]
