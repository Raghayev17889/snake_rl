from .base import Agent
from .q_learning_agent import QLearningAgent
from .random_agent import RandomAgent

try:
    from .dqn_agent import DQNAgent
    HAS_TORCH = True
except Exception:
    DQNAgent = None
    HAS_TORCH = False

__all__ = ["Agent", "QLearningAgent", "RandomAgent", "DQNAgent", "HAS_TORCH"]
