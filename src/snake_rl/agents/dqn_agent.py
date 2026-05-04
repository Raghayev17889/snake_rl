from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
import random

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover - import guard for optional dependency
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from .base import Agent


@dataclass(slots=True)
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    next_legal_actions: list[int]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int, rng: random.Random) -> list[Transition]:
        batch_size = min(batch_size, len(self.buffer))
        indices = rng.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(Agent):
    def __init__(
        self,
        hidden_sizes: tuple[int, int] = (128, 128),
        gamma: float = 0.95,
        lr: float = 1e-4,
        batch_size: int = 64,
        memory_size: int = 50000,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.9995,
        min_epsilon: float = 0.1,
        target_update_frequency: int = 200,
        learning_starts: int = 1000,
        train_frequency: int = 4,
        max_grad_norm: float = 10.0,
        seed: int | None = None,
        state_mode: str = "features",
    ) -> None:
        if torch is None or nn is None:
            raise RuntimeError(
                "PyTorch is required for DQNAgent. Install the 'torch' extra to enable this agent."
            ) from TORCH_IMPORT_ERROR

        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.target_update_frequency = target_update_frequency
        self.hidden_sizes = hidden_sizes
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.max_grad_norm = max_grad_norm
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.memory = ReplayBuffer(memory_size)
        self.device = torch.device("cpu")
        self.policy_net: nn.Module | None = None
        self.target_net: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.loss_fn = nn.SmoothL1Loss()
        self.training_steps = 0
        self.observe_steps = 0
        self.action_size = 4
        self.state_mode = state_mode

    @classmethod
    def available(cls) -> bool:
        return torch is not None and nn is not None

    def act(self, observation: dict[str, Any], legal_actions: list[int] | None = None, explore: bool = True) -> int:
        state_vector = self._vectorize_observation(observation)
        self._ensure_networks(len(state_vector))

        if explore and float(self.rng.random()) < self.epsilon:
            choices = legal_actions if legal_actions else list(range(self.action_size))
            return int(self.rng.choice(choices))

        # normalize input to small range to stabilize training
        state_vector = [float(x) for x in state_vector]
        mean = sum(state_vector) / max(1, len(state_vector))
        var = sum((x - mean) ** 2 for x in state_vector) / max(1, len(state_vector))
        std = var ** 0.5 if var > 1e-6 else 1.0
        state_vector = [(x - mean) / std for x in state_vector]
        state_tensor = self._state_tensor(state_vector)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0].tolist()

        if legal_actions:
            for action_index in range(self.action_size):
                if action_index not in legal_actions:
                    q_values[action_index] = float("-inf")

        return int(max(range(self.action_size), key=lambda index: q_values[index]))

    def observe(self, state: dict[str, Any], action: int, reward: float, next_state: dict[str, Any], done: bool) -> None:
        self.observe_steps += 1
        state_vector = self._vectorize_observation(state)
        next_state_vector = self._vectorize_observation(next_state)
        next_legal_actions = self._legal_actions_from_observation(next_state)
        self._ensure_networks(len(state_vector))

        self.memory.append(
            Transition(
                state=state_vector,
                action=action,
                reward=reward,
                next_state=next_state_vector,
                next_legal_actions=next_legal_actions,
                done=done,
            )
        )
        self._train_step()

    def end_episode(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _ensure_networks(self, input_dim: int) -> None:
        if self.policy_net is not None and self.target_net is not None and self.optimizer is not None:
            return

        self.policy_net = self._build_network(input_dim)
        self.target_net = self._build_network(input_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def _build_network(self, input_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, self.action_size))
        return nn.Sequential(*layers)

    def _state_tensor(self, state_vector: list[float]):
        return torch.tensor([state_vector], dtype=torch.float32, device=self.device)

    def _train_step(self) -> None:
        if len(self.memory) < max(self.batch_size, self.learning_starts):
            return
        if self.train_frequency > 1 and self.observe_steps % self.train_frequency != 0:
            return

        assert self.policy_net is not None
        assert self.target_net is not None
        assert self.optimizer is not None

        batch = self.memory.sample(self.batch_size, self.rng)

        states = torch.tensor([transition.state for transition in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([[transition.action] for transition in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([transition.next_state for transition in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)

        current_q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            all_next_q_values = self.target_net(next_states)
            next_q_values = torch.full((len(batch),), float("-inf"), dtype=torch.float32, device=self.device)
            for index, transition in enumerate(batch):
                legal_indices = torch.tensor(transition.next_legal_actions, dtype=torch.int64, device=self.device)
                next_q_values[index] = all_next_q_values[index, legal_indices].max()
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping to stabilize updates
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _vectorize_observation(self, observation: dict[str, Any]) -> list[float]:
        features = [float(value) for value in observation["features"]]
        if self.state_mode == "features":
            return features
        # features_grid
        grid = observation["grid"]
        flattened_grid: list[float] = []
        for channel in grid:
            for row in channel:
                for value in row:
                    flattened_grid.append(float(value))
        return features + flattened_grid

    def _legal_actions_from_observation(self, observation: dict[str, Any]) -> list[int]:
        from .state_utils import legal_actions_from_features

        return legal_actions_from_features(observation["features"])

    def save(self, path: str, metadata: dict | None = None) -> None:
        assert self.policy_net is not None and self.target_net is not None and self.optimizer is not None
        payload = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "hidden_sizes": self.hidden_sizes,
            "action_size": self.action_size,
            "state_mode": self.state_mode,
        }
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.state_mode = payload.get("state_mode", self.state_mode)
        # ensure networks exist by building with current observation size if necessary
        if self.policy_net is None:
            sample_input_dim = None
            if "policy_state_dict" in payload:
                for k, v in payload["policy_state_dict"].items():
                    if k.endswith(".weight") and v.ndim == 2:
                        sample_input_dim = v.shape[1]
                        break
            if sample_input_dim is not None:
                self._ensure_networks(sample_input_dim)

        self.policy_net.load_state_dict(payload["policy_state_dict"])
        self.target_net.load_state_dict(payload["target_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epsilon = float(payload["epsilon"])
        self.training_steps = int(payload["training_steps"])
