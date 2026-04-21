#!/usr/bin/env python3
"""
Create a one-page summary figure for heuristic-primary validity results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_summary(summary_path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        summary[key.strip()] = value.strip()
    return summary


def _prepare_selected_metrics(threshold_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = [
        "primary_failure_probability",
        "learned_failure_probability",
        "learned_predicted_pose_error",
        "heuristic_epipolar_error_risk",
        "fusion_pred_error_epipolar_rankavg",
    ]
    frame = threshold_metrics[threshold_metrics["score_name"].isin(selected)].copy()
    frame["failure_threshold"] = frame["failure_threshold"].astype(float)
    return frame.sort_values(["score_name", "failure_threshold"])


def _plot_auc_panel(ax: plt.Axes, metrics: pd.DataFrame, value_column: str, title: str):
    labels = {
        "primary_failure_probability": "Primary",
        "learned_failure_probability": "Learned failure",
        "learned_predicted_pose_error": "Learned error",
        "heuristic_epipolar_error_risk": "Best single heuristic",
        "fusion_pred_error_epipolar_rankavg": "Best fixed fusion",
    }
    colors = {
        "primary_failure_probability": "#7a3db8",
        "learned_failure_probability": "#c44536",
        "learned_predicted_pose_error": "#d17b0f",
        "heuristic_epipolar_error_risk": "#2a6f97",
        "fusion_pred_error_epipolar_rankavg": "#2b9348",
    }
    for score_name, group in metrics.groupby("score_name", sort=False):
        ax.plot(
            group["failure_threshold"],
            group[value_column],
            marker="o",
            linewidth=2.2,
            label=labels.get(score_name, score_name),
            color=colors.get(score_name),
        )
    ax.set_xlabel("Failure Threshold (m)")
    ax.set_ylabel(title)
    ax.set_xticks(sorted(metrics["failure_threshold"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="best")


def _plot_correlation_panel(ax: plt.Axes, summary: dict[str, str]):
    labels = ["Learned failure", "Learned error", "Primary"]
    pearson = [
        float(summary.get("overall_learned_failure_vs_actual_pearson", "nan")),
        float(summary.get("overall_learned_pred_error_vs_actual_pearson", "nan")),
        float(summary.get("overall_primary_failure_vs_actual_pearson", "nan")),
    ]
    colors = ["#c44536", "#d17b0f", "#7a3db8"]
    bars = ax.bar(labels, pearson, color=colors, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_ylabel("Overall Pearson vs actual error")
    ax.set_title("Global Correlation")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, pearson):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.01 if np.isfinite(value) and value >= 0 else -0.02),
            f"{value:.3f}",
            ha="center",
            va="bottom" if np.isfinite(value) and value >= 0 else "top",
            fontsize=9,
        )
    primary_positive = summary.get("positive_primary_failure_correlation_runs", "")
    learned_positive = summary.get("positive_learned_failure_correlation_runs", "")
    ax.text(
        0.02,
        0.96,
        f"Positive-run counts\nLearned: {learned_positive}\nPrimary: {primary_positive}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#dddddd"},
    )


def _plot_sequence_panel(ax: plt.Axes, sequence_summary: pd.DataFrame):
    x = np.arange(len(sequence_summary))
    width = 0.24
    columns = [
        ("learned_failure_vs_actual_pearson", "Learned failure", "#c44536"),
        ("learned_pred_error_vs_actual_pearson", "Learned error", "#d17b0f"),
        ("primary_failure_vs_actual_pearson", "Primary", "#7a3db8"),
    ]
    for idx, (column, label, color) in enumerate(columns):
        ax.bar(x + (idx - 1) * width, sequence_summary[column], width=width, label=label, color=color, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(sequence_summary["sequence_short"])
    ax.set_ylabel("Pearson vs actual error")
    ax.set_title("Per-Sequence Correlation")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="best")


def create_summary(validity_dir: Path, output_dir: Path):
    summary = _load_summary(validity_dir / "validity_summary.txt")
    threshold_metrics = pd.read_csv(validity_dir / "threshold_metrics.csv")
    sequence_summary = pd.read_csv(validity_dir / "sequence_validity_summary.csv")

    selected_metrics = _prepare_selected_metrics(threshold_metrics)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    _plot_auc_panel(axes[0, 0], selected_metrics, "roc_auc", "ROC-AUC")
    axes[0, 0].set_title("ROC-AUC Across Failure Thresholds")

    _plot_auc_panel(axes[0, 1], selected_metrics, "average_precision", "PR-AUC / AP")
    axes[0, 1].set_title("Average Precision Across Failure Thresholds")

    _plot_correlation_panel(axes[1, 0], summary)
    _plot_sequence_panel(axes[1, 1], sequence_summary)

    title = (
        "Heuristic-Primary Multi-Sequence Replay Benchmark\n"
        f"{summary.get('num_sequences', '?')} sequences | "
        f"{summary.get('num_degraded_runs', '?')} degraded runs | "
        f"{summary.get('num_frames', '?')} frames | "
        f"summary threshold = {summary.get('summary_failure_threshold', '?')}m"
    )
    fig.suptitle(title, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "primary_benchmark_summary.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    summary_text = output_dir / "primary_benchmark_summary.txt"
    summary_text.write_text(
        "\n".join(
            [
                "Heuristic-primary benchmark summary",
                f"validity_dir: {validity_dir}",
                f"num_sequences: {summary.get('num_sequences', '?')}",
                f"num_degraded_runs: {summary.get('num_degraded_runs', '?')}",
                f"num_frames: {summary.get('num_frames', '?')}",
                f"summary_failure_threshold: {summary.get('summary_failure_threshold', '?')}",
                f"primary_failure_probability_roc_auc: {summary.get('primary_failure_probability_roc_auc', 'nan')}",
                f"learned_failure_probability_roc_auc: {summary.get('learned_failure_probability_roc_auc', 'nan')}",
                f"learned_predicted_pose_error_roc_auc: {summary.get('learned_predicted_pose_error_roc_auc', 'nan')}",
                f"best_heuristic_score: {summary.get('best_heuristic_score', 'n/a')}",
                f"best_heuristic_roc_auc: {summary.get('best_heuristic_roc_auc', 'nan')}",
                f"best_fusion_score: {summary.get('best_fusion_score', 'n/a')}",
                f"best_fusion_roc_auc: {summary.get('best_fusion_roc_auc', 'nan')}",
                f"overall_primary_failure_vs_actual_pearson: {summary.get('overall_primary_failure_vs_actual_pearson', 'nan')}",
                f"overall_learned_failure_vs_actual_pearson: {summary.get('overall_learned_failure_vs_actual_pearson', 'nan')}",
                f"overall_learned_pred_error_vs_actual_pearson: {summary.get('overall_learned_pred_error_vs_actual_pearson', 'nan')}",
                "",
                f"figure_path: {figure_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"summary_figure: {figure_path}")
    print(f"summary_text: {summary_text}")


def main():
    parser = argparse.ArgumentParser(description="Create one-page heuristic-primary benchmark summary figure")
    parser.add_argument("--validity-dir", required=True, help="Directory containing validity_summary.txt and threshold_metrics.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for summary figure")
    args = parser.parse_args()

    create_summary(
        validity_dir=Path(args.validity_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
