from __future__ import annotations

from typing import Sequence


def direction_from_one_hot(features: Sequence[float]) -> int:
    # features expected to have direction one-hot at indices 7..10
    direction_one_hot = features[7:11]
    return int(max(range(4), key=lambda i: direction_one_hot[i]))


def legal_actions_from_features(features: Sequence[float]) -> list[int]:
    direction = direction_from_one_hot(features)
    opposite = (direction + 2) % 4
    return [a for a in range(4) if a != opposite]
