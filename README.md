# Snake RL Project

This project compares reinforcement learning approaches for the game Snake.

## Current scope

- DQN agent with replay memory and a target network (optional PyTorch dependency)
- Reward shaping experiments
- Auto-generated report with SVG plots for training and evaluation comparisons

## Planned next steps

1. Expand automated tests for environment dynamics and agent updates.
2. Add model checkpointing and reproducible experiment logging.
3. Compare additional DQN variants (target update cadence, replay size, epsilon schedule).

## Quick start

```bash
python -m pip install -e .
python -m snake_rl.scripts.generate_report
python -m snake_rl.scripts.run_random
```

To try the DQN path later, install PyTorch and run `python -m snake_rl.scripts.run_dqn`.

The core environment, agents, and report renderer stay lightweight and self-contained.

## Demo / Visualization

Watch the agents play Snake in real-time with ASCII rendering:

```bash
python -m snake_rl.scripts.play_demo --agent random
python -m snake_rl.scripts.play_demo --agent q_learning
python -m snake_rl.scripts.play_demo --agent dqn
```

The script prints the board, score, reward, and chosen action after each step. Use `--delay` to adjust the speed (default 0.2 seconds):

```bash
python -m snake_rl.scripts.play_demo --agent dqn --delay 0.5
```

If trained agent checkpoints exist (e.g., from `generate_report`), they are loaded automatically.

## Reproducibility

This project uses deterministic episode seeds to enable fair comparisons across agents. Seeds are generated from a single experiment-level seed and used for each episode consistently across all agents. To reproduce results exactly:

1. Activate the project Python environment you used to run experiments (e.g., the `.venv_torch` created for DQN).

```powershell
.\.venv_torch\Scripts\Activate.ps1
python -m snake_rl.scripts.generate_report
```

2. Use the same experiment seed (default `42`) to reproduce identical episode initializations.

Interpretation of `average ± std` values in the report:

- `average` is the sample mean across evaluation episodes.
- `std` is the sample standard deviation, giving a sense of variability across episodes.
- Use the `std` to assess uncertainty; smaller `std` indicates more consistent agent behavior.

For exact bitwise reproducibility of DQN you should also fix the Python interpreter (CPython) and package versions — see `pyproject.toml` and install the `torch` extra into the same interpreter used for running experiments.
