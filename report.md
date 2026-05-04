# Snake RL Experimental Report

This report studies Snake as a sequential decision problem and compares a random baseline, tabular Q-learning, and DQN under reward shaping and state-representation variations.

## Project Objective

The goal is to compare simple reinforcement-learning agents on Snake, analyze the effect of reward shaping, and study whether a richer state representation helps DQN learn a stronger policy than a compact feature-only input. The DQN checkpoint used for the demo is trained on a 6x6 grid for sample efficiency, while the random and Q-learning baselines keep the standard environment setup unless explicitly noted.

## Snake as an MDP

Snake can be modeled as a finite Markov decision process with a discrete grid state, four actions, a step penalty, terminal failure on collision, and sparse positive reward when food is eaten. Reward shaping adds a dense auxiliary signal that preserves the main task while reducing the credit-assignment burden.

## Implemented Agents

- **Random Agent**: samples uniformly from legal actions and serves as a non-learning baseline.
- **Q-Learning**: tabular agent that learns action values over hand-crafted features.
- **DQN**: neural agent with replay memory and a target network; it is more sample-hungry than tabular methods and is evaluated under two state representations.

## State Representations

Two DQN inputs are compared. The feature-only representation is compact and encodes danger, food direction, and heading. The feature+grid representation concatenates the feature vector with a flattened spatial grid, which increases expressiveness but also the input size and optimization difficulty. In this project, the feature-only representation is the default DQN setting because the grid-augmented version is noisier and usually needs more samples to stabilize.

## Reward Design

The DQN training environment uses food reward = 10, death reward = -10, step reward = -0.02, and shaping rewards of +0.2 when moving closer to food, -0.2 when moving away, and -0.05 when movement does not change the distance. These values keep the terminal reward dominant so the agent is still incentivized to actually eat food rather than simply cycle near it.

## Experimental Setup

- Random seed: 42
- Training episodes: 250
- Evaluation episodes: 30
- Random and Q-learning agents use the standard environment setup
- DQN uses a 6 x 6 grid with a shorter horizon to improve sample efficiency
- DQN uses the same evaluation seeds as the other agents for fair comparison

## Results Table

| Agent | Evaluation Episodes (n) | Average Score ± std | Average Reward ± std | Average Steps ± std | Best Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 30 | 0.37 (±0.76) | -0.85 (±0.77) | 23.30 (±16.04) | 3 |
| Q-Learning | 30 | 7.60 (±3.85) | 6.13 (±3.61) | 55.90 (±28.68) | 16 |
| Q-Learning + Reward Shaping | 30 | 8.43 (±4.11) | 4.60 (±2.89) | 63.63 (±30.62) | 16 |
| Q-Learning + Shaping (α=0.2) | 30 | 5.67 (±3.71) | 2.36 (±2.37) | 49.83 (±32.37) | 16 |
| DQN (features) | 30 | 6.27 (±2.90) | 51.48 (±28.55) | 28.67 (±12.58) | 13 |
| DQN (features+grid) | 30 | 1.60 (±1.45) | 7.15 (±13.09) | 62.90 (±40.85) | 6 |

## Figure Explanations

![Training learning curves](report_assets/learning_curves.svg)

**Training learning curves.** This figure shows smoothed training return over episodes for each learning agent. Curves that rise sooner indicate faster learning, while instability or flattening suggests that exploration, reward design, or representation capacity is limiting progress. In this project the shaped Q-learning variants should usually rise earlier than the unshaped baseline, and the feature-only DQN is expected to be steadier than the feature+grid version on a small sample budget because it has less input noise to fit.

![Evaluation comparison](report_assets/evaluation_comparison.svg)

**Evaluation comparison bar chart.** This plot compares evaluation score and reward across agents. The random baseline should remain lowest because it does not learn. Q-learning should outperform random because it can assign values to useful food-seeking actions. DQN may improve with the smaller 6x6 grid, but it is still a more complex approximator and can underperform a simpler tabular method if optimization is unstable or the representation is too noisy. The comparison therefore shows the effect of reward shaping and representation choice rather than assuming DQN is automatically best.

![Reward shaping analysis](report_assets/reward_shaping_analysis.svg)

**Reward shaping comparison.** This figure isolates the effect of shaped reward on the Q-learning family. If the shaped curve rises earlier or stays smoother, the shaping signal is helping the agent move toward food more consistently. If the gap is small, that suggests the sparse food reward is already sufficient under the current grid size and hyperparameters. A strong early improvement with similar or better final scores is the desirable pattern because it means shaping improved learning without overwhelming the actual task objective.

![Score distribution](report_assets/score_distribution.svg)

**Score distribution.** This plot shows evaluation score spread across episodes. A narrow distribution suggests stable behavior, while a wide spread indicates stochasticity or policy brittleness. It is useful for distinguishing a strong average result from one that only succeeds in a few lucky episodes; for DQN, a tighter spread is often a sign that the feature-only input is easier to optimize than the full grid encoding.

## Discussion / Analysis

The random agent performs poorly because it has no memory or value function and therefore cannot reliably move toward food. Q-learning improves over random by learning which feature combinations tend to lead to food while avoiding collisions. Reward shaping changes learning behavior by adding dense progress feedback, which usually makes the learning curve rise earlier and more smoothly, although too much shaping can distort the main objective if it becomes larger than the food reward. DQN often needs more training than tabular Q-learning because it must optimize a neural network and discover useful feature interactions from experience rather than direct table updates. State representation matters because the feature-only input is compact but can miss spatial detail, whereas the feature+grid input is richer but harder to optimize. Results may vary between runs because reinforcement learning is stochastic: initialization, environment randomness, replay sampling, and exploration all influence the final policy. The report therefore treats DQN as a more sample-hungry model rather than assuming it is automatically superior.

## Limitations

- The project keeps to a small grid-world setting, so results may not transfer to larger or more complex Snake variants.
- DQN training remains sensitive to hyperparameters and may require additional tuning for consistent gains.
- The evaluation set is finite, so average metrics still carry sampling noise and should be read together with standard deviation.

## Conclusion

The project shows that simple baselines can be meaningfully compared in a compact RL setting and that reward shaping and state representation materially affect performance. The most reliable gains come from preserving a clear terminal reward while giving the agent enough dense signal to discover food-seeking behavior. The results also show that DQN is not automatically better than tabular methods unless training stability and state encoding are chosen carefully, especially when the action cycle can become trapped in repetitive behavior.

## Reproducibility Notes

- Command used to generate this report: `python -m snake_rl.scripts.generate_report`
- Seed used for the experiment: `42`
- Training episodes: `250`
- Evaluation episodes: `30`
- DQN checkpoint path: `C:/Users/DE/Desktop/AI/report_assets/dqn_best_checkpoint.pth`
- Best eval checkpoint score: `6.366666666666666` at episode `400`
- DQN checkpoint training episodes: `3000`
- Plot output folder: `C:/Users/DE/Desktop/AI/report_assets`
- Reported values are written as `average ± std`, where `std` is the sample standard deviation across evaluation episodes.

## How To Demo Each Agent

A unified demo script is available for all implemented agents. Use one of the following commands:

- Random agent demo: `python -m snake_rl.scripts.play_demo --agent random`
- Q-learning agent demo: `python -m snake_rl.scripts.play_demo --agent q_learning`
- DQN agent demo: `python -m snake_rl.scripts.play_demo --agent dqn`

Optional demo controls:

- Speed: add `--delay 0.1` (smaller is faster, larger is slower)
- Episode length cap: add `--max-steps 200`

Recommended preparation before demos:

- Train Q-learning checkpoint: `python -m snake_rl.scripts.run_q_learning`
- Train DQN best checkpoint: `python -m snake_rl.scripts.run_dqn`
- Random demo requires no training checkpoint