from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import random

from ..config import SnakeConfig


Direction = int
Action = int
Point = tuple[int, int]

UP: Direction = 0
RIGHT: Direction = 1
DOWN: Direction = 2
LEFT: Direction = 3

DIR_VECTORS: dict[Direction, tuple[int, int]] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

ACTION_NAMES = {UP: "up", RIGHT: "right", DOWN: "down", LEFT: "left"}


@dataclass(slots=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class SnakeEnv:
    def __init__(self, config: SnakeConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SnakeConfig()
        self.rng = random.Random(seed)
        self.snake: list[Point] = []
        self.food: Point = (0, 0)
        self.direction: Direction = RIGHT
        self.steps = 0
        self.score = 0

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = random.Random(seed)

        center_row = self.config.height // 2
        center_col = self.config.width // 2
        self.snake = [(center_row, center_col), (center_row, center_col - 1), (center_row, center_col - 2)]
        self.direction = RIGHT
        self.steps = 0
        self.score = 0
        self.food = self._spawn_food()
        return self._get_observation()

    def step(self, action: Action) -> StepResult:
        if action not in ACTION_NAMES:
            raise ValueError(f"Invalid action: {action}")

        if not self._is_opposite(action, self.direction):
            self.direction = action

        next_head = self._move(self.snake[0], self.direction)
        self.steps += 1

        reward = self.config.step_reward
        done = False
        will_grow = next_head == self.food
        occupied_body = self.snake if will_grow else self.snake[:-1]

        if self._hits_wall(next_head) or next_head in occupied_body:
            reward = self.config.death_reward
            done = True
        else:
            self.snake.insert(0, next_head)
            if will_grow:
                self.score += 1
                reward = self.config.food_reward
                if self.config.reward_shaping:
                    reward += 0.1
                self.food = self._spawn_food()
            else:
                self.snake.pop()

            if self.config.reward_shaping:
                reward += self._shape_reward(next_head)

        if self.steps >= self.config.max_steps:
            done = True

        observation = self._get_observation()
        info = {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
            "action_names": ACTION_NAMES,
        }
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    def legal_actions(self) -> list[Action]:
        return [action for action in ACTION_NAMES if not self._is_opposite(action, self.direction)]

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.config.width)] for _ in range(self.config.height)]
        for row, col in self.snake:
            grid[row][col] = "o"
        head_row, head_col = self.snake[0]
        grid[head_row][head_col] = "H"
        food_row, food_col = self.food
        grid[food_row][food_col] = "F"
        return "\n".join(" ".join(row) for row in grid)

    def _get_observation(self) -> dict[str, Any]:
        features = self._feature_vector()
        grid = self._empty_grid()

        for row, col in self.snake:
            grid[0][row][col] = 1.0
        grid[1][self.food[0]][self.food[1]] = 1.0
        for row in range(self.config.height):
            for col in range(self.config.width):
                grid[2][row][col] = self.direction / 3.0

        return {"features": features, "grid": grid}

    def _feature_vector(self) -> list[float]:
        head_row, head_col = self.snake[0]
        food_row, food_col = self.food

        danger_straight = 1.0 if self._danger_ahead(self.direction) else 0.0
        danger_right = 1.0 if self._danger_ahead(self._turn_right(self.direction)) else 0.0
        danger_left = 1.0 if self._danger_ahead(self._turn_left(self.direction)) else 0.0

        food_up = 1.0 if food_row < head_row else 0.0
        food_down = 1.0 if food_row > head_row else 0.0
        food_left = 1.0 if food_col < head_col else 0.0
        food_right = 1.0 if food_col > head_col else 0.0

        direction_one_hot = [1.0 if self.direction == i else 0.0 for i in range(4)]

        return [
            danger_straight,
            danger_right,
            danger_left,
            food_up,
            food_down,
            food_left,
            food_right,
            *direction_one_hot,
        ]

    def _shape_reward(self, next_head: Point) -> float:
        current_distance = self._manhattan_distance(self.snake[0], self.food)
        next_distance = self._manhattan_distance(next_head, self.food)
        if next_distance < current_distance:
            return 0.2
        if next_distance > current_distance:
            return -0.2
        return -0.05

    def _spawn_food(self) -> Point:
        available = [(row, col) for row in range(self.config.height) for col in range(self.config.width) if (row, col) not in self.snake]
        if not available:
            return self.snake[0]
        return self.rng.choice(available)

    def _move(self, point: Point, direction: Direction) -> Point:
        delta_row, delta_col = DIR_VECTORS[direction]
        return point[0] + delta_row, point[1] + delta_col

    def _hits_wall(self, point: Point) -> bool:
        row, col = point
        return row < 0 or row >= self.config.height or col < 0 or col >= self.config.width

    def _danger_ahead(self, direction: Direction) -> bool:
        next_point = self._move(self.snake[0], direction)
        if self._hits_wall(next_point):
            return True
        will_grow = next_point == self.food
        occupied_body = self.snake if will_grow else self.snake[:-1]
        return next_point in occupied_body

    def _manhattan_distance(self, first: Point, second: Point) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _is_opposite(self, first: Direction, second: Direction) -> bool:
        return (first + 2) % 4 == second

    def _turn_left(self, direction: Direction) -> Direction:
        return (direction - 1) % 4

    def _turn_right(self, direction: Direction) -> Direction:
        return (direction + 1) % 4

    def _empty_grid(self) -> list[list[list[float]]]:
        return [
            [[0.0 for _ in range(self.config.width)] for _ in range(self.config.height)]
            for _ in range(3)
        ]

