#!/usr/bin/env python3
"""
Check train/val/test vs runtime feature-distribution alignment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SELF_AWARE_ROOT = ROOT_DIR / "self_aware_slam"
sys.path.insert(0, str(SELF_AWARE_ROOT))

from src.data.dataset_builder import load_dataset  # noqa: E402
from src.data.feature_engineering import extract_learning_features, normalize_features  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402


def _load_slam_metrics(run_dir: Path) -> pd.DataFrame:
    metrics_path = run_dir / "slam_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing slam_metrics.csv in {run_dir}")
    return pd.read_csv(metrics_path)


def _extract_raw_features(metrics_df: pd.DataFrame, feature_names: List[str], rolling_window: int) -> np.ndarray:
    return extract_learning_features(
        metrics_df,
        feature_columns=feature_names,
        rolling_window=rolling_window,
    )


def _collect_dataset_split_features(
    dataset: Dict,
    split_name: str,
    feature_names: List[str],
    rolling_window: int,
    run_kind: str | None = None,
) -> np.ndarray:
    features = []
    for record in dataset.get("split_runs", {}).get(split_name, []):
        if run_kind and record.get("run_kind") != run_kind:
            continue
        run_dir = Path(record["path"])
        metrics_df = _load_slam_metrics(run_dir)
        features.append(_extract_raw_features(metrics_df, feature_names, rolling_window))
    if not features:
        raise ValueError(f"No features collected for split '{split_name}' (run_kind={run_kind})")
    return np.concatenate(features, axis=0)


def _iter_runtime_metric_paths(runtime_root: Path) -> Iterable[Path]:
    preferred = sorted(runtime_root.rglob("*_self_aware/slam_metrics.csv"))
    if preferred:
        return preferred
    return sorted(runtime_root.rglob("slam_metrics.csv"))


def _collect_runtime_features(runtime_root: Path, feature_names: List[str], rolling_window: int) -> np.ndarray:
    metric_paths = list(_iter_runtime_metric_paths(runtime_root))
    if not metric_paths:
        raise FileNotFoundError(f"No runtime slam_metrics.csv files found under {runtime_root}")

    features = []
    for metrics_path in metric_paths:
        metrics_df = pd.read_csv(metrics_path)
        features.append(_extract_raw_features(metrics_df, feature_names, rolling_window))
    return np.concatenate(features, axis=0)


def _summarize_matrix(name: str, raw: np.ndarray, norm_stats: Dict[str, np.ndarray]) -> pd.DataFrame:
    normalized = normalize_features(raw, norm_stats)
    rows = []
    for idx in range(raw.shape[1]):
        raw_col = raw[:, idx]
        norm_col = normalized[:, idx]
        rows.append(
            {
                "group": name,
                "feature_index": idx,
                "raw_mean": float(np.mean(raw_col)),
                "raw_std": float(np.std(raw_col)),
                "raw_p01": float(np.percentile(raw_col, 1)),
                "raw_p50": float(np.percentile(raw_col, 50)),
                "raw_p99": float(np.percentile(raw_col, 99)),
                "norm_mean": float(np.mean(norm_col)),
                "norm_std": float(np.std(norm_col)),
                "pct_abs_z_gt_2": float(np.mean(np.abs(norm_col) > 2.0) * 100.0),
                "pct_abs_z_gt_3": float(np.mean(np.abs(norm_col) > 3.0) * 100.0),
                "pct_abs_z_gt_5": float(np.mean(np.abs(norm_col) > 5.0) * 100.0),
                "n_rows": int(len(raw_col)),
            }
        )
    return pd.DataFrame(rows)


def _make_alignment_table(summary_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    indexed = {
        group: df.set_index("feature_index")
        for group, df in summary_df.groupby("group")
    }
    train = indexed["train"]
    runtime = indexed["runtime"]

    rows = []
    for idx, feature_name in enumerate(feature_names):
        train_raw_std = float(train.loc[idx, "raw_std"])
        if train_raw_std < 1e-8:
            train_raw_std = 1.0
        rows.append(
            {
                "feature_name": feature_name,
                "train_raw_mean": float(train.loc[idx, "raw_mean"]),
                "train_raw_std": float(train.loc[idx, "raw_std"]),
                "runtime_raw_mean": float(runtime.loc[idx, "raw_mean"]),
                "runtime_raw_std": float(runtime.loc[idx, "raw_std"]),
                "runtime_norm_mean": float(runtime.loc[idx, "norm_mean"]),
                "runtime_norm_std": float(runtime.loc[idx, "norm_std"]),
                "runtime_pct_abs_z_gt_2": float(runtime.loc[idx, "pct_abs_z_gt_2"]),
                "runtime_pct_abs_z_gt_3": float(runtime.loc[idx, "pct_abs_z_gt_3"]),
                "runtime_pct_abs_z_gt_5": float(runtime.loc[idx, "pct_abs_z_gt_5"]),
                "mean_shift_in_train_std": float(
                    abs(runtime.loc[idx, "raw_mean"] - train.loc[idx, "raw_mean"]) / train_raw_std
                ),
                "std_ratio_runtime_over_train": float(
                    (runtime.loc[idx, "raw_std"] + 1e-8) / (float(train.loc[idx, "raw_std"]) + 1e-8)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["mean_shift_in_train_std", "runtime_pct_abs_z_gt_3"],
        ascending=[False, False],
    )


def _write_summary_text(
    output_path: Path,
    dataset_path: Path,
    runtime_root: Path,
    summary_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
) -> None:
    runtime_rows = summary_df[summary_df["group"] == "runtime"]
    mean_abs_norm = float(np.mean(np.abs(runtime_rows["norm_mean"])))
    mean_pct_z3 = float(np.mean(runtime_rows["pct_abs_z_gt_3"]))
    max_shift_feature = alignment_df.iloc[0]

    train_std = alignment_df.set_index("feature_name")["train_raw_std"].to_dict()
    runtime_means = summary_df[summary_df["group"] == "runtime"].set_index("feature_name")["raw_mean"].to_dict()
    candidate_groups = [g for g in summary_df["group"].unique().tolist() if g != "runtime"]
    group_scores = []
    for group in candidate_groups:
        group_means = summary_df[summary_df["group"] == group].set_index("feature_name")["raw_mean"].to_dict()
        score_terms = []
        for feature_name, runtime_mean in runtime_means.items():
            std = train_std.get(feature_name, 1.0) or 1.0
            group_mean = group_means[feature_name]
            score_terms.append(abs(runtime_mean - group_mean) / std)
        group_scores.append((group, float(np.mean(score_terms))))
    group_scores.sort(key=lambda item: item[1])

    lines = [
        "Feature distribution alignment summary",
        f"dataset_path: {dataset_path}",
        f"runtime_root: {runtime_root}",
        "",
        "Global runtime-vs-train summary:",
        f"- mean(|runtime normalized feature mean|): {mean_abs_norm:.4f}",
        f"- mean(runtime pct |z| > 3): {mean_pct_z3:.2f}%",
        "",
        "Largest feature shift:",
        f"- feature: {max_shift_feature['feature_name']}",
        f"- mean_shift_in_train_std: {max_shift_feature['mean_shift_in_train_std']:.4f}",
        f"- runtime_norm_mean: {max_shift_feature['runtime_norm_mean']:.4f}",
        f"- runtime_norm_std: {max_shift_feature['runtime_norm_std']:.4f}",
        f"- runtime_pct_abs_z_gt_3: {max_shift_feature['runtime_pct_abs_z_gt_3']:.2f}%",
        "",
        "Top shifted features:",
    ]

    for _, row in alignment_df.head(8).iterrows():
        lines.append(
            f"- {row['feature_name']}: shift={row['mean_shift_in_train_std']:.3f} train-std, "
            f"runtime_norm_mean={row['runtime_norm_mean']:.3f}, "
            f"runtime_pct_abs_z_gt_3={row['runtime_pct_abs_z_gt_3']:.2f}%"
        )

    lines.extend(
        [
            "",
            "Closest dataset groups to runtime (lower is better):",
        ]
    )
    for group, score in group_scores[:6]:
        lines.append(f"- {group}: mean feature shift = {score:.3f} train-std")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_alignment(alignment_df: pd.DataFrame, output_path: Path) -> None:
    top = alignment_df.head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

    axes[0].barh(top["feature_name"], top["mean_shift_in_train_std"], color="#4C78A8")
    axes[0].set_title("Runtime Mean Shift (train std units)")
    axes[0].set_xlabel("abs(runtime_mean - train_mean) / train_std")

    axes[1].barh(top["feature_name"], top["runtime_pct_abs_z_gt_3"], color="#E45756")
    axes[1].set_title("Runtime Out-of-Range Rate")
    axes[1].set_xlabel("% runtime rows with |z| > 3")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check train vs runtime feature-distribution alignment.")
    parser.add_argument("--dataset-path", required=True, help="Path to the training dataset pickle.")
    parser.add_argument("--runtime-root", required=True, help="Root directory containing runtime self-aware outputs.")
    parser.add_argument("--output-dir", required=True, help="Directory to write alignment outputs.")
    parser.add_argument("--config", default=None, help="Optional config.yaml path.")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    rolling_window = config["features"].get("rolling_window", 5)

    dataset_path = Path(args.dataset_path).resolve()
    runtime_root = Path(args.runtime_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(str(dataset_path))
    feature_names = dataset["feature_names"]
    norm_stats = dataset["norm_stats"]

    matrices = {
        split_name: _collect_dataset_split_features(dataset, split_name, feature_names, rolling_window)
        for split_name in ["train", "val", "test"]
    }
    for split_name in ["train", "val", "test"]:
        run_kinds = sorted({record.get("run_kind") for record in dataset.get("split_runs", {}).get(split_name, [])})
        for run_kind in run_kinds:
            if run_kind is None:
                continue
            try:
                matrices[f"{split_name}_{run_kind}"] = _collect_dataset_split_features(
                    dataset,
                    split_name,
                    feature_names,
                    rolling_window,
                    run_kind=run_kind,
                )
            except ValueError:
                continue
    matrices["runtime"] = _collect_runtime_features(runtime_root, feature_names, rolling_window)

    summary_parts = [
        _summarize_matrix(group_name, matrix, norm_stats)
        for group_name, matrix in matrices.items()
    ]
    summary_df = pd.concat(summary_parts, ignore_index=True)
    summary_df["feature_name"] = summary_df["feature_index"].map(dict(enumerate(feature_names)))
    alignment_df = _make_alignment_table(summary_df, feature_names)

    summary_df.to_csv(output_dir / "feature_distribution_summary.csv", index=False)
    alignment_df.to_csv(output_dir / "feature_distribution_alignment.csv", index=False)
    _write_summary_text(
        output_dir / "feature_distribution_summary.txt",
        dataset_path=dataset_path,
        runtime_root=runtime_root,
        summary_df=summary_df,
        alignment_df=alignment_df,
    )
    _plot_alignment(alignment_df, output_dir / "feature_distribution_alignment.png")

    print(f"summary_csv: {output_dir / 'feature_distribution_summary.csv'}")
    print(f"alignment_csv: {output_dir / 'feature_distribution_alignment.csv'}")
    print(f"summary_txt: {output_dir / 'feature_distribution_summary.txt'}")
    print(f"alignment_png: {output_dir / 'feature_distribution_alignment.png'}")


if __name__ == "__main__":
    main()
