from __future__ import annotations

import pytest

from snake_rl.agents.dqn_agent import DQNAgent
from snake_rl.scripts.play_demo import _load_dqn_checkpoint


def _sample_observation() -> dict[str, object]:
    return {
        "features": [0.0] * 11,
        "grid": [[[0.0 for _ in range(2)] for _ in range(2)] for _ in range(3)],
    }


def test_dqn_state_mode_features_vector_size_consistency() -> None:
    pytest.importorskip("torch")

    agent = DQNAgent(state_mode="features")
    vector = agent._vectorize_observation(_sample_observation())

    assert len(vector) == 11


def test_dqn_checkpoint_metadata_round_trip(tmp_path) -> None:
    pytest.importorskip("torch")

    agent = DQNAgent(state_mode="features")
    agent._ensure_networks(11)

    checkpoint_path = tmp_path / "dqn_best_checkpoint.pth"
    metadata = {
        "grid_width": 6,
        "grid_height": 6,
        "max_steps": 100,
        "reward_shaping": True,
        "food_reward": 10.0,
        "death_reward": -10.0,
        "step_reward": -0.02,
        "state_mode": "features",
        "training_episodes": 3000,
        "best_eval_avg_score": 12.5,
        "best_eval_episode": 1800,
        "seed": 42,
    }

    agent.save(str(checkpoint_path), metadata=metadata)

    loaded_agent = DQNAgent(state_mode="features")
    loaded_agent.load(str(checkpoint_path))

    assert loaded_agent.state_mode == "features"
    assert loaded_agent.epsilon == pytest.approx(agent.epsilon)

    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    for key, value in metadata.items():
        assert payload[key] == value


def test_play_demo_builds_env_from_checkpoint_metadata(tmp_path) -> None:
    pytest.importorskip("torch")

    agent = DQNAgent(state_mode="features")
    agent._ensure_networks(11)

    checkpoint_path = tmp_path / "dqn_best_checkpoint.pth"
    metadata = {
        "grid_width": 6,
        "grid_height": 6,
        "max_steps": 100,
        "reward_shaping": True,
        "food_reward": 10.0,
        "death_reward": -10.0,
        "step_reward": -0.02,
        "state_mode": "features",
        "training_episodes": 3000,
        "best_eval_avg_score": 12.5,
        "best_eval_episode": 1800,
        "seed": 42,
    }

    agent.save(str(checkpoint_path), metadata=metadata)

    loaded_agent, env_config, payload = _load_dqn_checkpoint(checkpoint_path)

    assert loaded_agent is not None
    assert env_config is not None
    assert payload is not None
    assert loaded_agent.state_mode == "features"
    assert env_config.width == 6
    assert env_config.height == 6
    assert env_config.max_steps == 100
    assert env_config.reward_shaping is True
    assert env_config.food_reward == 10.0
    assert env_config.death_reward == -10.0
    assert env_config.step_reward == -0.02
    assert payload["best_eval_avg_score"] == 12.5
    assert payload["best_eval_episode"] == 1800
