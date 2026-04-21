#!/usr/bin/env python3
"""
Audit label difficulty and split skew for a built training dataset.

This is intended for the aligned-feature dataset path, where the main questions
are:

1. How hard is the resolved label threshold?
2. How different are the train / val / test label distributions?
3. Are the aligned splits now operating in a very different difficulty regime
   from the earlier mixed diagnostic datasets?
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

def _parse_percentiles(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values.astype(float))
    probs = np.arange(1, len(values) + 1, dtype=float) / max(len(values), 1)
    return values, probs


def _summarize_split(name: str, split: Dict[str, np.ndarray]) -> Dict[str, float]:
    y_error = split["y_error"].astype(float)
    y_failure = split["y_failure"].astype(float)
    return {
        "split": name,
        "n": int(len(y_error)),
        "failure_rate": float(np.mean(y_failure)) if len(y_failure) else float("nan"),
        "y_error_min": float(np.min(y_error)) if len(y_error) else float("nan"),
        "y_error_p25": float(np.percentile(y_error, 25)) if len(y_error) else float("nan"),
        "y_error_p50": float(np.percentile(y_error, 50)) if len(y_error) else float("nan"),
        "y_error_p75": float(np.percentile(y_error, 75)) if len(y_error) else float("nan"),
        "y_error_p90": float(np.percentile(y_error, 90)) if len(y_error) else float("nan"),
        "y_error_p95": float(np.percentile(y_error, 95)) if len(y_error) else float("nan"),
        "y_error_max": float(np.max(y_error)) if len(y_error) else float("nan"),
    }


def _threshold_table(dataset: Dict, percentiles: Iterable[float]) -> pd.DataFrame:
    train_errors = dataset["train"]["y_error"].astype(float)
    rows: List[Dict[str, float]] = []
    for percentile in percentiles:
        threshold = float(np.percentile(train_errors, percentile))
        row = {
            "percentile": float(percentile),
            "threshold": threshold,
        }
        for split_name in ("train", "val", "test"):
            y_error = dataset[split_name]["y_error"].astype(float)
            row[f"{split_name}_positive_rate"] = float(np.mean(y_error > threshold))
            row[f"{split_name}_median_error"] = float(np.median(y_error))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_histograms(dataset: Dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for idx, split_name in enumerate(("train", "val", "test")):
        y_error = dataset[split_name]["y_error"].astype(float)
        axes[idx].hist(y_error, bins=20, color="#4C78A8", alpha=0.85, edgecolor="white")
        axes[idx].set_title(f"{split_name} y_error")
        axes[idx].set_xlabel("future target")
        axes[idx].grid(True, alpha=0.2)
        if idx == 0:
            axes[idx].set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output_dir / "aligned_label_histograms.png", dpi=180)
    plt.close(fig)


def _plot_cdf(dataset: Dict, output_dir: Path, resolved_threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {
        "train": "#4C78A8",
        "val": "#F58518",
        "test": "#54A24B",
    }
    for split_name in ("train", "val", "test"):
        x, y = _ecdf(dataset[split_name]["y_error"].astype(float))
        ax.plot(x, y, label=split_name, color=colors[split_name], linewidth=2)
    ax.axvline(resolved_threshold, linestyle="--", color="#C44E52", label=f"resolved threshold={resolved_threshold:.3f}")
    ax.set_title("Aligned Dataset Target CDF")
    ax.set_xlabel("future target")
    ax.set_ylabel("ECDF")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "aligned_label_cdf.png", dpi=180)
    plt.close(fig)


def _plot_threshold_sensitivity(threshold_table: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for split_name, color in (("train", "#4C78A8"), ("val", "#F58518"), ("test", "#54A24B")):
        ax.plot(
            threshold_table["percentile"],
            threshold_table[f"{split_name}_positive_rate"],
            marker="o",
            label=split_name,
            color=color,
        )
    ax.set_title("Positive Rate vs Train Percentile Threshold")
    ax.set_xlabel("train percentile")
    ax.set_ylabel("positive rate")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "aligned_threshold_sensitivity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit label distribution for an aligned training dataset.")
    parser.add_argument("--dataset-path", required=True, help="Path to dataset pickle.")
    parser.add_argument("--output-dir", required=True, help="Directory to write audit outputs.")
    parser.add_argument(
        "--percentiles",
        default="70,80,85,90,95",
        help="Comma-separated train percentiles to evaluate.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write summaries/csv only and skip matplotlib outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset_path, "rb") as handle:
        dataset = pickle.load(handle)
    target_definition = dataset.get("target_definition", {})
    resolved_threshold = float(target_definition.get("resolved_classification_threshold", float("nan")))

    split_summary = pd.DataFrame(
        [_summarize_split(split_name, dataset[split_name]) for split_name in ("train", "val", "test")]
    )
    split_summary.to_csv(output_dir / "split_label_summary.csv", index=False)

    threshold_table = _threshold_table(dataset, _parse_percentiles(args.percentiles))
    threshold_table.to_csv(output_dir / "threshold_sensitivity.csv", index=False)

    if not args.skip_plots:
        _plot_histograms(dataset, output_dir)
        _plot_cdf(dataset, output_dir, resolved_threshold)
        _plot_threshold_sensitivity(threshold_table, output_dir)

    lines = [
        "Aligned dataset label audit",
        f"dataset_path: {args.dataset_path}",
        f"source_mode: {dataset.get('source_mode')}",
        f"split_protocol: {dataset.get('split_protocol')}",
        f"target_definition: {target_definition}",
        "",
        "Split summary:",
    ]
    for row in split_summary.to_dict(orient="records"):
        lines.append(
            f"- {row['split']}: n={row['n']}, failure_rate={row['failure_rate']:.4f}, "
            f"median={row['y_error_p50']:.3f}, p90={row['y_error_p90']:.3f}, max={row['y_error_max']:.3f}"
        )

    lines.append("")
    lines.append("Threshold sensitivity from train percentiles:")
    for row in threshold_table.to_dict(orient="records"):
        lines.append(
            f"- P{int(row['percentile'])}: threshold={row['threshold']:.3f}, "
            f"train={row['train_positive_rate']:.3f}, val={row['val_positive_rate']:.3f}, "
            f"test={row['test_positive_rate']:.3f}"
        )

    (output_dir / "label_audit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"split_summary_csv: {output_dir / 'split_label_summary.csv'}")
    print(f"threshold_sensitivity_csv: {output_dir / 'threshold_sensitivity.csv'}")
    print(f"summary_txt: {output_dir / 'label_audit_summary.txt'}")
    if not args.skip_plots:
        print(f"hist_png: {output_dir / 'aligned_label_histograms.png'}")
        print(f"cdf_png: {output_dir / 'aligned_label_cdf.png'}")
        print(f"threshold_png: {output_dir / 'aligned_threshold_sensitivity.png'}")


if __name__ == "__main__":
    main()
