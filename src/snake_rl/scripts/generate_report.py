from __future__ import annotations

from html import escape
from pathlib import Path

from ..config import ExperimentConfig
from ..reporting import run_experiments, summarize_results


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "report_assets"
REPORT_PATH = ROOT / "report.md"


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    config = ExperimentConfig(episodes=250, evaluation_episodes=30, seed=42)
    print("Starting experiments...")
    random_results, series = run_experiments(config)
    print("Training and evaluation complete.")
    print("Generating plots...")

    random_summary = summarize_results(random_results)
    evaluation_summaries = {item.label: summarize_results(item.eval_results) for item in series}

    score_plot = ASSETS / "learning_curves.svg"
    comparison_plot = ASSETS / "evaluation_comparison.svg"
    shaping_plot = ASSETS / "reward_shaping_analysis.svg"
    distribution_plot = ASSETS / "score_distribution.svg"

    _plot_learning_curves(series, score_plot)
    _plot_evaluation_comparison(random_summary, evaluation_summaries, comparison_plot)
    _plot_reward_shaping_analysis(series, shaping_plot)
    _plot_score_distribution(random_results, series, distribution_plot)
    print("Writing report...")
    _write_report(
        config,
        random_summary,
        evaluation_summaries,
        series,
        score_plot.relative_to(ROOT),
        comparison_plot.relative_to(ROOT),
        shaping_plot.relative_to(ROOT),
        distribution_plot.relative_to(ROOT),
    )
    print(f"Report written to: {REPORT_PATH}")
    print(f"Assets written to: {ASSETS}")


def _plot_learning_curves(series: list, output_path: Path) -> None:
    curves = []
    for index, item in enumerate(series):
        scores = [result.score for result in item.train_results]
        window = 15
        smoothed = _rolling_mean(scores, window)
        curves.append((item.label, smoothed, _palette(index)))

    output_path.write_text(_render_line_chart(curves, title="Training Learning Curves", x_label="Episode", y_label="Smoothed Score"), encoding="utf-8")


def _plot_evaluation_comparison(random_summary: dict[str, float], evaluation_summaries: dict[str, dict[str, float]], output_path: Path) -> None:
    labels = ["Random"] + list(evaluation_summaries.keys())
    scores = [random_summary["average_score"]] + [summary["average_score"] for summary in evaluation_summaries.values()]
    rewards = [random_summary["average_reward"]] + [summary["average_reward"] for summary in evaluation_summaries.values()]

    bars = [
        ("Average Score", scores, _palette(0)),
        ("Average Reward", rewards, _palette(1)),
    ]
    output_path.write_text(_render_grouped_bar_chart(labels, bars, title="Evaluation Comparison", y_label="Value"), encoding="utf-8")


def _plot_reward_shaping_analysis(series: list, output_path: Path) -> None:
    shaping_series = [item for item in series if item.reward_shaping]
    curves = []
    for index, item in enumerate(shaping_series):
        scores = [result.score for result in item.train_results]
        window = 15
        smoothed = _rolling_mean(scores, window)
        curves.append((item.label, smoothed, _palette(index)))

    output_path.write_text(_render_line_chart(curves, title="Reward Shaping Effect on Learning", x_label="Episode", y_label="Smoothed Score"), encoding="utf-8")


def _plot_score_distribution(random_results: list, series: list, output_path: Path) -> None:
    distributions = [("Random", [result.score for result in random_results], _palette(0))]
    for index, item in enumerate(series, start=1):
        distributions.append((item.label, [result.score for result in item.eval_results], _palette(index)))
    output_path.write_text(_render_distribution_plot(distributions, title="Evaluation Score Distribution", x_label="Agent", y_label="Evaluation Score"), encoding="utf-8")


def _write_report(
    config: ExperimentConfig,
    random_summary: dict[str, float],
    evaluation_summaries: dict[str, dict[str, float]],
    series: list,
    score_plot: Path,
    comparison_plot: Path,
    shaping_plot: Path,
    distribution_plot: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Snake RL Experimental Report")
    lines.append("")
    lines.append("This report studies Snake as a sequential decision problem and compares a random baseline, tabular Q-learning, and DQN under reward shaping and state-representation variations.")
    lines.append("")
    lines.append("## Project Objective")
    lines.append("")
    lines.append("The goal is to compare simple reinforcement-learning agents on Snake, analyze the effect of reward shaping, and study whether a richer state representation helps DQN learn a stronger policy than a compact feature-only input. The DQN checkpoint used for the demo is trained on a 6x6 grid for sample efficiency, while the random and Q-learning baselines keep the standard environment setup unless explicitly noted.")
    lines.append("")
    lines.append("## Snake as an MDP")
    lines.append("")
    lines.append("Snake can be modeled as a finite Markov decision process with a discrete grid state, four actions, a step penalty, terminal failure on collision, and sparse positive reward when food is eaten. Reward shaping adds a dense auxiliary signal that preserves the main task while reducing the credit-assignment burden.")
    lines.append("")
    lines.append("## Implemented Agents")
    lines.append("")
    lines.append("- **Random Agent**: samples uniformly from legal actions and serves as a non-learning baseline.")
    lines.append("- **Q-Learning**: tabular agent that learns action values over hand-crafted features.")
    lines.append("- **DQN**: neural agent with replay memory and a target network; it is more sample-hungry than tabular methods and is evaluated under two state representations.")
    lines.append("")
    lines.append("## State Representations")
    lines.append("")
    lines.append("Two DQN inputs are compared. The feature-only representation is compact and encodes danger, food direction, and heading. The feature+grid representation concatenates the feature vector with a flattened spatial grid, which increases expressiveness but also the input size and optimization difficulty. In this project, the feature-only representation is the default DQN setting because the grid-augmented version is noisier and usually needs more samples to stabilize.")
    lines.append("")
    lines.append("## Reward Design")
    lines.append("")
    lines.append("The DQN training environment uses food reward = 10, death reward = -10, step reward = -0.02, and shaping rewards of +0.2 when moving closer to food, -0.2 when moving away, and -0.05 when movement does not change the distance. These values keep the terminal reward dominant so the agent is still incentivized to actually eat food rather than simply cycle near it.")
    lines.append("")
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append(f"- Random seed: {config.seed}")
    lines.append(f"- Training episodes: {config.episodes}")
    lines.append(f"- Evaluation episodes: {config.evaluation_episodes}")
    lines.append("- Random and Q-learning agents use the standard environment setup")
    lines.append("- DQN uses a 6 x 6 grid with a shorter horizon to improve sample efficiency")
    lines.append("- DQN uses the same evaluation seeds as the other agents for fair comparison")
    lines.append("")
    lines.append("## Results Table")
    lines.append("")
    lines.append("| Agent | Evaluation Episodes (n) | Average Score ± std | Average Reward ± std | Average Steps ± std | Best Score |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(_summary_row("Random", random_summary, int(random_summary.get("n", 0))))
    for item in series:
        summary = evaluation_summaries[item.label]
        lines.append(_summary_row(item.label, summary, int(summary.get("n", 0))))
    lines.append("")
    lines.append("## Figure Explanations")
    lines.append("")
    lines.append(f"![Training learning curves]({score_plot.as_posix()})")
    lines.append("")
    lines.append("**Training learning curves.** This figure shows smoothed training return over episodes for each learning agent. Curves that rise sooner indicate faster learning, while instability or flattening suggests that exploration, reward design, or representation capacity is limiting progress. In this project the shaped Q-learning variants should usually rise earlier than the unshaped baseline, and the feature-only DQN is expected to be steadier than the feature+grid version on a small sample budget because it has less input noise to fit.")
    lines.append("")
    lines.append(f"![Evaluation comparison]({comparison_plot.as_posix()})")
    lines.append("")
    lines.append("**Evaluation comparison bar chart.** This plot compares evaluation score and reward across agents. The random baseline should remain lowest because it does not learn. Q-learning should outperform random because it can assign values to useful food-seeking actions. DQN may improve with the smaller 6x6 grid, but it is still a more complex approximator and can underperform a simpler tabular method if optimization is unstable or the representation is too noisy. The comparison therefore shows the effect of reward shaping and representation choice rather than assuming DQN is automatically best.")
    lines.append("")
    lines.append(f"![Reward shaping analysis]({shaping_plot.as_posix()})")
    lines.append("")
    lines.append("**Reward shaping comparison.** This figure isolates the effect of shaped reward on the Q-learning family. If the shaped curve rises earlier or stays smoother, the shaping signal is helping the agent move toward food more consistently. If the gap is small, that suggests the sparse food reward is already sufficient under the current grid size and hyperparameters. A strong early improvement with similar or better final scores is the desirable pattern because it means shaping improved learning without overwhelming the actual task objective.")
    lines.append("")
    lines.append(f"![Score distribution]({distribution_plot.as_posix()})")
    lines.append("")
    lines.append("**Score distribution.** This plot shows evaluation score spread across episodes. A narrow distribution suggests stable behavior, while a wide spread indicates stochasticity or policy brittleness. It is useful for distinguishing a strong average result from one that only succeeds in a few lucky episodes; for DQN, a tighter spread is often a sign that the feature-only input is easier to optimize than the full grid encoding.")
    lines.append("")
    lines.append("## Discussion / Analysis")
    lines.append("")
    lines.append("The random agent performs poorly because it has no memory or value function and therefore cannot reliably move toward food. Q-learning improves over random by learning which feature combinations tend to lead to food while avoiding collisions. Reward shaping changes learning behavior by adding dense progress feedback, which usually makes the learning curve rise earlier and more smoothly, although too much shaping can distort the main objective if it becomes larger than the food reward. DQN often needs more training than tabular Q-learning because it must optimize a neural network and discover useful feature interactions from experience rather than direct table updates. State representation matters because the feature-only input is compact but can miss spatial detail, whereas the feature+grid input is richer but harder to optimize. Results may vary between runs because reinforcement learning is stochastic: initialization, environment randomness, replay sampling, and exploration all influence the final policy. The report therefore treats DQN as a more sample-hungry model rather than assuming it is automatically superior.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- The project keeps to a small grid-world setting, so results may not transfer to larger or more complex Snake variants.")
    lines.append("- DQN training remains sensitive to hyperparameters and may require additional tuning for consistent gains.")
    lines.append("- The evaluation set is finite, so average metrics still carry sampling noise and should be read together with standard deviation.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append("The project shows that simple baselines can be meaningfully compared in a compact RL setting and that reward shaping and state representation materially affect performance. The most reliable gains come from preserving a clear terminal reward while giving the agent enough dense signal to discover food-seeking behavior. The results also show that DQN is not automatically better than tabular methods unless training stability and state encoding are chosen carefully, especially when the action cycle can become trapped in repetitive behavior.")
    lines.append("")
    lines.append("## Reproducibility Notes")
    lines.append("")
    lines.append(f"- Command used to generate this report: `python -m snake_rl.scripts.generate_report`")
    lines.append(f"- Seed used for the experiment: `{config.seed}`")
    lines.append(f"- Training episodes: `{config.episodes}`")
    lines.append(f"- Evaluation episodes: `{config.evaluation_episodes}`")
    checkpoint_summary = _load_dqn_checkpoint_summary(ASSETS / "dqn_best_checkpoint.pth")
    lines.append(f"- DQN checkpoint path: `{(ASSETS / 'dqn_best_checkpoint.pth').as_posix()}`")
    if checkpoint_summary is not None:
        lines.append(f"- Best eval checkpoint score: `{checkpoint_summary['best_eval_avg_score']}` at episode `{checkpoint_summary['best_eval_episode']}`")
        lines.append(f"- DQN checkpoint training episodes: `{checkpoint_summary['training_episodes']}`")
    else:
        lines.append("- Best eval checkpoint score: unavailable until a DQN checkpoint is generated")
    lines.append(f"- Plot output folder: `{ASSETS.as_posix()}`")
    lines.append("- Reported values are written as `average ± std`, where `std` is the sample standard deviation across evaluation episodes.")
    lines.append("")
    lines.append("## How To Demo Each Agent")
    lines.append("")
    lines.append("A unified demo script is available for all implemented agents. Use one of the following commands:")
    lines.append("")
    lines.append("- Random agent demo: `python -m snake_rl.scripts.play_demo --agent random`")
    lines.append("- Q-learning agent demo: `python -m snake_rl.scripts.play_demo --agent q_learning`")
    lines.append("- DQN agent demo: `python -m snake_rl.scripts.play_demo --agent dqn`")
    lines.append("")
    lines.append("Optional demo controls:")
    lines.append("")
    lines.append("- Speed: add `--delay 0.1` (smaller is faster, larger is slower)")
    lines.append("- Episode length cap: add `--max-steps 200`")
    lines.append("")
    lines.append("Recommended preparation before demos:")
    lines.append("")
    lines.append("- Train Q-learning checkpoint: `python -m snake_rl.scripts.run_q_learning`")
    lines.append("- Train DQN best checkpoint: `python -m snake_rl.scripts.run_dqn`")
    lines.append("- Random demo requires no training checkpoint")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _summary_row(label: str, summary: dict[str, float], n: int) -> str:
    score_std = summary.get("score_std", 0.0)
    reward_std = summary.get("reward_std", 0.0)
    steps_std = summary.get("steps_std", 0.0)
    return (
        f"| {label} | {n} | {summary['average_score']:.2f} (±{score_std:.2f}) | {summary['average_reward']:.2f} (±{reward_std:.2f}) | "
        f"{summary['average_steps']:.2f} (±{steps_std:.2f}) | {summary['best_score']:.0f} |"
    )


def _load_dqn_checkpoint_summary(checkpoint_path: Path) -> dict[str, str] | None:
    if not checkpoint_path.exists():
        return None

    try:
        import torch
    except Exception:
        return None

    payload = torch.load(checkpoint_path, map_location="cpu")
    return {
        "best_eval_avg_score": str(payload.get("best_eval_avg_score", payload.get("best_eval_score", "unknown"))),
        "best_eval_episode": str(payload.get("best_eval_episode", "unknown")),
        "training_episodes": str(payload.get("training_episodes", "unknown")),
    }


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []

    smoothed: list[float] = []
    rolling_sum = 0.0
    for index, value in enumerate(values):
        rolling_sum += value
        if index >= window:
            rolling_sum -= values[index - window]
        current_window_size = min(window, index + 1)
        smoothed.append(rolling_sum / current_window_size)
    return smoothed


def _palette(index: int) -> str:
    colors = ["#1f4e79", "#8a5cf6", "#e07a5f", "#3d9970"]
    return colors[index % len(colors)]


def _render_line_chart(series: list[tuple[str, list[float], str]], title: str, x_label: str, y_label: str) -> str:
    width = 960
    height = 540
    margin_left = 80
    margin_right = 30
    margin_top = 70
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [value for _, values, _ in series for value in values]
    max_value = max(all_values) if all_values else 1.0
    min_value = min(all_values) if all_values else 0.0
    if max_value == min_value:
        max_value += 1.0

    max_points = max((len(values) for _, values, _ in series), default=1)
    lines: list[str] = [_svg_header(width, height)]
    lines.append(_svg_background(width, height))
    lines.append(_svg_text(width / 2, 34, title, size=24, anchor="middle", weight="700"))
    lines.append(_draw_axes(margin_left, margin_top, plot_width, plot_height))

    for tick_index in range(6):
        y_value = min_value + (max_value - min_value) * tick_index / 5
        y_position = margin_top + plot_height - plot_height * tick_index / 5
        lines.append(_svg_line(margin_left - 6, y_position, margin_left, y_position, stroke="#d0d7de"))
        lines.append(_svg_text(margin_left - 12, y_position + 4, f"{y_value:.1f}", size=12, anchor="end", fill="#57606a"))

    for tick_index in range(6):
        x_value = max_points * tick_index / 5 if max_points else 0
        x_position = margin_left + plot_width * tick_index / 5
        lines.append(_svg_line(x_position, margin_top + plot_height, x_position, margin_top + plot_height + 6, stroke="#d0d7de"))
        lines.append(_svg_text(x_position, margin_top + plot_height + 22, f"{int(x_value)}", size=12, anchor="middle", fill="#57606a"))

    lines.append(_svg_text(margin_left + plot_width / 2, height - 16, x_label, size=14, anchor="middle", fill="#24292f"))
    lines.append(_svg_rotated_text(22, margin_top + plot_height / 2, y_label, size=14, anchor="middle", fill="#24292f"))

    for label, values, color in series:
        if not values:
            continue
        points = []
        for index, value in enumerate(values):
            x = margin_left + (plot_width * index / max(1, max_points - 1))
            y = margin_top + plot_height - ((value - min_value) / (max_value - min_value)) * plot_height
            points.append(f"{x:.2f},{y:.2f}")
            lines.append(_svg_circle(x, y, 2.8, fill=color))
        lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{' '.join(points)}' />")

    legend_x = margin_left + plot_width - 5
    legend_y = margin_top + 10
    for index, (label, _, color) in enumerate(series):
        entry_y = legend_y + index * 24
        lines.append(_svg_rect(legend_x - 140, entry_y - 12, 14, 14, fill=color))
        lines.append(_svg_text(legend_x - 120, entry_y, label, size=13, anchor="start", fill="#24292f"))

    lines.append(_svg_footer())
    return "\n".join(lines)


def _render_grouped_bar_chart(labels: list[str], bars: list[tuple[str, list[float], str]], title: str, y_label: str) -> str:
    width = 960
    height = 540
    margin_left = 80
    margin_right = 30
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(value for _, values, _ in bars for value in values)
    min_value = min(0.0, min(value for _, values, _ in bars for value in values))
    if max_value == min_value:
        max_value += 1.0

    group_count = len(labels)
    bar_count = len(bars)
    group_width = plot_width / max(1, group_count)
    bar_width = min(42.0, group_width / (bar_count + 1))

    lines: list[str] = [_svg_header(width, height)]
    lines.append(_svg_background(width, height))
    lines.append(_svg_text(width / 2, 34, title, size=24, anchor="middle", weight="700"))
    lines.append(_draw_axes(margin_left, margin_top, plot_width, plot_height))

    for tick_index in range(6):
        y_value = min_value + (max_value - min_value) * tick_index / 5
        y_position = margin_top + plot_height - plot_height * tick_index / 5
        lines.append(_svg_line(margin_left - 6, y_position, margin_left, y_position, stroke="#d0d7de"))
        lines.append(_svg_text(margin_left - 12, y_position + 4, f"{y_value:.1f}", size=12, anchor="end", fill="#57606a"))

    for group_index, label in enumerate(labels):
        center_x = margin_left + group_width * group_index + group_width / 2
        lines.append(_svg_text(center_x, margin_top + plot_height + 24, label, size=12, anchor="middle", fill="#24292f"))
        for bar_index, (bar_label, values, color) in enumerate(bars):
            value = values[group_index]
            ratio = (value - min_value) / (max_value - min_value)
            bar_height = ratio * plot_height
            x = margin_left + group_width * group_index + (group_width - (bar_count * bar_width)) / 2 + bar_index * bar_width
            y = margin_top + plot_height - bar_height
            lines.append(_svg_rect(x, y, bar_width - 4, bar_height, fill=color))

    legend_x = margin_left + plot_width - 5
    legend_y = margin_top + 10
    for index, (bar_label, _, color) in enumerate(bars):
        entry_y = legend_y + index * 24
        lines.append(_svg_rect(legend_x - 160, entry_y - 12, 14, 14, fill=color))
        lines.append(_svg_text(legend_x - 140, entry_y, bar_label, size=13, anchor="start", fill="#24292f"))

    lines.append(_svg_text(margin_left + plot_width / 2, height - 16, "Agent", size=14, anchor="middle", fill="#24292f"))
    lines.append(_svg_rotated_text(22, margin_top + plot_height / 2, y_label, size=14, anchor="middle", fill="#24292f"))
    lines.append(_svg_footer())
    return "\n".join(lines)


def _render_distribution_plot(series: list[tuple[str, list[float], str]], title: str, x_label: str, y_label: str) -> str:
    width = 960
    height = 540
    margin_left = 90
    margin_right = 30
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = [value for _, values, _ in series for value in values]
    max_value = max(all_values) if all_values else 1.0
    min_value = min(all_values) if all_values else 0.0
    if max_value == min_value:
        max_value += 1.0

    lines: list[str] = [_svg_header(width, height)]
    lines.append(_svg_background(width, height))
    lines.append(_svg_text(width / 2, 34, title, size=24, anchor="middle", weight="700"))
    lines.append(_draw_axes(margin_left, margin_top, plot_width, plot_height))

    for tick_index in range(6):
        y_value = min_value + (max_value - min_value) * tick_index / 5
        y_position = margin_top + plot_height - plot_height * tick_index / 5
        lines.append(_svg_line(margin_left - 6, y_position, margin_left, y_position, stroke="#d0d7de"))
        lines.append(_svg_text(margin_left - 12, y_position + 4, f"{y_value:.1f}", size=12, anchor="end", fill="#57606a"))

    if series:
        slot_width = plot_width / len(series)
        for index, (label, values, color) in enumerate(series):
            center_x = margin_left + slot_width * index + slot_width / 2
            lines.append(_svg_text(center_x, margin_top + plot_height + 24, label, size=12, anchor="middle", fill="#24292f"))
            for point_index, value in enumerate(values):
                offset = ((point_index % 7) - 3) * 5
                x = center_x + offset
                y = margin_top + plot_height - ((value - min_value) / (max_value - min_value)) * plot_height
                lines.append(_svg_circle(x, y, 3.2, fill=color))
            if values:
                mean_value = sum(values) / len(values)
                mean_y = margin_top + plot_height - ((mean_value - min_value) / (max_value - min_value)) * plot_height
                lines.append(_svg_line(center_x - 18, mean_y, center_x + 18, mean_y, stroke=color, width=3.0))

    lines.append(_svg_text(margin_left + plot_width / 2, height - 16, x_label, size=14, anchor="middle", fill="#24292f"))
    lines.append(_svg_rotated_text(22, margin_top + plot_height / 2, y_label, size=14, anchor="middle", fill="#24292f"))
    lines.append(_svg_footer())
    return "\n".join(lines)


def _draw_axes(x: float, y: float, width: float, height: float) -> str:
    return "\n".join([
        _svg_line(x, y, x, y + height, stroke="#24292f", width=1.8),
        _svg_line(x, y + height, x + width, y + height, stroke="#24292f", width=1.8),
    ])


def _svg_header(width: int, height: int) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"


def _svg_footer() -> str:
    return "</svg>"


def _svg_background(width: int, height: int) -> str:
    return f"<rect x='0' y='0' width='{width}' height='{height}' rx='20' fill='#f8fafc' />"


def _svg_line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.0) -> str:
    return f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' stroke='{stroke}' stroke-width='{width}' />"


def _svg_rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return f"<rect x='{x:.2f}' y='{y:.2f}' width='{width:.2f}' height='{height:.2f}' rx='4' fill='{fill}' />"


def _svg_circle(cx: float, cy: float, radius: float, fill: str) -> str:
    return f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='{radius:.2f}' fill='{fill}' />"


def _svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", fill: str = "#24292f", weight: str = "400") -> str:
    return (
        f"<text x='{x:.2f}' y='{y:.2f}' fill='{fill}' font-size='{size}' font-family='Segoe UI, Arial, sans-serif' "
        f"font-weight='{weight}' text-anchor='{anchor}'>{escape(text)}</text>"
    )


def _svg_rotated_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", fill: str = "#24292f") -> str:
    return (
        f"<text x='{x:.2f}' y='{y:.2f}' fill='{fill}' font-size='{size}' font-family='Segoe UI, Arial, sans-serif' "
        f"text-anchor='{anchor}' transform='rotate(-90 {x:.2f} {y:.2f})'>{escape(text)}</text>"
    )


if __name__ == "__main__":
    main()
