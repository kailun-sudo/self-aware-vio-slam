#!/usr/bin/env python3
"""
Run a model validity benchmark over multisequence Self-Aware VIO-SLAM outputs.

The benchmark focuses on whether the predictor is merely reactive or actually
tracks real localization error:

- correlation with actual pose error
- binary failure-detection quality across configurable error thresholds
- calibration quality for failure probability
- comparison against simple heuristic baselines from SLAM internals
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent.parent / ".mplcache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent


def _slug(value: float) -> str:
    return str(value).replace(".", "p")


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    frame = pd.concat([a, b], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1], method=method))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _infer_baseline_dir(comparison_gui_path: str) -> Path:
    comparison_path = Path(comparison_gui_path)
    scenario_root = comparison_path.parents[2]
    return scenario_root.parent / "baseline_self_aware"


def _infer_degraded_dir(comparison_gui_path: str) -> Path:
    comparison_path = Path(comparison_gui_path)
    scenario_root = comparison_path.parents[2]
    return scenario_root / "degraded_self_aware"


def _merge_predictions_and_metrics(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [column for column in ("timestamp", "frame_id") if column in predictions.columns and column in metrics.columns]
    if keys:
        merged = predictions.merge(metrics, on=keys, how="left", suffixes=("", "_metric"))
    else:
        metrics_tail = metrics.tail(len(predictions)).reset_index(drop=True)
        merged = pd.concat([predictions.reset_index(drop=True), metrics_tail], axis=1)
    return merged


def _load_run_frame(run_dir: Path, metadata: Dict[str, object]) -> pd.DataFrame:
    predictions = _read_csv(run_dir / "reliability_predictions.csv")
    metrics = _read_csv(run_dir / "slam_metrics.csv")
    merged = _merge_predictions_and_metrics(predictions, metrics)
    for key, value in metadata.items():
        merged[key] = value
    merged["run_dir"] = str(run_dir)
    merged["run_id"] = f"{metadata['sequence_short']}::{metadata['scenario']}"
    return merged


def _build_frame_dataset(sweep_results_path: Path, include_baseline: bool = True) -> pd.DataFrame:
    results = _read_csv(sweep_results_path)
    frames: List[pd.DataFrame] = []
    seen_baselines: set[str] = set()

    for row in results.to_dict(orient="records"):
        sequence = row["sequence"]
        sequence_short = row["sequence_short"]

        baseline_dir = _infer_baseline_dir(row["comparison_gui_path"])
        degraded_dir = _infer_degraded_dir(row["comparison_gui_path"])

        if include_baseline and sequence not in seen_baselines:
            frames.append(
                _load_run_frame(
                    baseline_dir,
                    {
                        "sequence": sequence,
                        "sequence_short": sequence_short,
                        "run_kind": "baseline",
                        "scenario": "baseline",
                        "base_scenario": "baseline",
                        "camera_degradation": "none",
                        "imu_degradation": "none",
                        "severity": 0.0,
                        "description": "No injected degradation",
                    },
                )
            )
            seen_baselines.add(sequence)

        frames.append(
            _load_run_frame(
                degraded_dir,
                {
                    "sequence": sequence,
                    "sequence_short": sequence_short,
                    "run_kind": "degraded",
                    "scenario": row["scenario"],
                    "base_scenario": row.get("base_scenario", row["scenario"]),
                    "camera_degradation": row.get("camera_degradation", "unknown"),
                    "imu_degradation": row.get("imu_degradation", "none") or "none",
                    "severity": float(row.get("severity", 0.0)),
                    "description": row.get("description", ""),
                },
            )
        )

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    return dataset


def _best_f1(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    curve = _binary_curve(y_true, scores)
    thresholds = curve["thresholds"]
    if len(thresholds) == 0:
        return {
            "best_f1": float("nan"),
            "best_threshold": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
        }
    precision = curve["precision"]
    recall = curve["recall"]
    f1_values = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_idx = int(np.nanargmax(f1_values))
    return {
        "best_f1": float(f1_values[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
    }


def _score_definitions(frame_data: pd.DataFrame) -> Dict[str, pd.Series]:
    scores: Dict[str, pd.Series] = {
        "model_failure_probability": frame_data["failure_probability"],
        "model_predicted_pose_error": frame_data["predicted_pose_error"],
    }
    if "inlier_ratio" in frame_data.columns:
        scores["heuristic_inlier_ratio_risk"] = 1.0 - frame_data["inlier_ratio"]
    if "pose_optimization_residual" in frame_data.columns:
        scores["heuristic_pose_residual_risk"] = frame_data["pose_optimization_residual"]
        scores["heuristic_imu_residual_risk"] = frame_data["pose_optimization_residual"]
    if "num_inliers" in frame_data.columns:
        scores["heuristic_num_inliers_risk"] = -frame_data["num_inliers"]
    if "mean_epipolar_error" in frame_data.columns:
        scores["heuristic_epipolar_error_risk"] = frame_data["mean_epipolar_error"]
    return scores


def _binary_curve(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    mask = np.isfinite(scores)
    y_true = y_true[mask].astype(int)
    scores = scores[mask].astype(float)
    if len(scores) == 0:
        empty = np.array([], dtype=float)
        return {
            "thresholds": empty,
            "precision": empty,
            "recall": empty,
            "tpr": empty,
            "fpr": empty,
        }

    order = np.argsort(-scores, kind="mergesort")
    scores = scores[order]
    y_true = y_true[order]

    distinct = np.where(np.diff(scores))[0]
    threshold_idxs = np.r_[distinct, len(scores) - 1]
    thresholds = scores[threshold_idxs]

    tps = np.cumsum(y_true)[threshold_idxs].astype(float)
    fps = (1 + threshold_idxs - tps).astype(float)

    positives = float(np.sum(y_true))
    negatives = float(len(y_true) - positives)

    precision = tps / np.clip(tps + fps, 1e-12, None)
    recall = tps / positives if positives > 0 else np.full_like(tps, np.nan)
    tpr = recall.copy()
    fpr = fps / negatives if negatives > 0 else np.full_like(fps, np.nan)

    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
        "fpr": fpr,
    }


def _roc_auc_from_curve(fpr: np.ndarray, tpr: np.ndarray) -> float:
    mask = np.isfinite(fpr) & np.isfinite(tpr)
    if np.sum(mask) < 2:
        return float("nan")
    fpr = np.r_[0.0, fpr[mask], 1.0]
    tpr = np.r_[0.0, tpr[mask], 1.0]
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def _average_precision_from_curve(recall: np.ndarray, precision: np.ndarray) -> float:
    mask = np.isfinite(recall) & np.isfinite(precision)
    if np.sum(mask) < 2:
        return float("nan")
    recall = recall[mask]
    precision = precision[mask]
    order = np.argsort(recall)
    recall = recall[order]
    precision = precision[order]
    recall = np.r_[0.0, recall]
    precision = np.r_[precision[0], precision]
    delta = np.diff(recall)
    return float(np.sum(delta * precision[1:]))


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    mask = np.isfinite(prob)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask].astype(float)
    prob = prob[mask].astype(float)
    return float(np.mean((prob - y_true) ** 2))


def _evaluate_score(y_true: np.ndarray, scores: np.ndarray, score_name: str) -> Dict[str, float]:
    mask = np.isfinite(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    metrics: Dict[str, float] = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "best_f1": float("nan"),
        "best_threshold": float("nan"),
        "best_precision": float("nan"),
        "best_recall": float("nan"),
        "fixed_f1": float("nan"),
        "fixed_precision": float("nan"),
        "fixed_recall": float("nan"),
        "brier_score": float("nan"),
    }
    if len(np.unique(y_true)) < 2:
        return metrics

    curve = _binary_curve(y_true, scores)
    metrics["roc_auc"] = _roc_auc_from_curve(curve["fpr"], curve["tpr"])
    metrics["average_precision"] = _average_precision_from_curve(curve["recall"], curve["precision"])
    metrics.update(_best_f1(y_true, scores))

    if score_name == "model_failure_probability":
        fixed_pred = (scores >= 0.5).astype(int)
        fixed = _binary_metrics(y_true, fixed_pred)
        metrics["fixed_f1"] = fixed["f1"]
        metrics["fixed_precision"] = fixed["precision"]
        metrics["fixed_recall"] = fixed["recall"]
        metrics["brier_score"] = _brier_score(y_true, scores)

    return metrics


def _run_level_correlations(frame_data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_columns = [
        "sequence",
        "sequence_short",
        "run_kind",
        "scenario",
        "base_scenario",
        "camera_degradation",
        "imu_degradation",
        "severity",
    ]
    for keys, group in frame_data.groupby(group_columns, dropna=False):
        record = dict(zip(group_columns, keys))
        record["num_frames"] = int(len(group))
        record["failure_vs_actual_pearson"] = _safe_corr(group["failure_probability"], group["actual_pose_error"])
        record["failure_vs_actual_spearman"] = _safe_corr(group["failure_probability"], group["actual_pose_error"], method="spearman")
        record["pred_error_vs_actual_pearson"] = _safe_corr(group["predicted_pose_error"], group["actual_pose_error"])
        record["pred_error_vs_actual_spearman"] = _safe_corr(group["predicted_pose_error"], group["actual_pose_error"], method="spearman")
        if "inlier_ratio" in group.columns:
            record["heuristic_inlier_vs_actual_pearson"] = _safe_corr(1.0 - group["inlier_ratio"], group["actual_pose_error"])
        if "pose_optimization_residual" in group.columns:
            record["heuristic_residual_vs_actual_pearson"] = _safe_corr(group["pose_optimization_residual"], group["actual_pose_error"])
        rows.append(record)
    return pd.DataFrame(rows)


def _aggregate_correlations(run_correlations: pd.DataFrame, output_dir: Path) -> None:
    sequence_summary = (
        run_correlations.groupby("sequence_short", dropna=False)[
            [
                "failure_vs_actual_pearson",
                "failure_vs_actual_spearman",
                "pred_error_vs_actual_pearson",
                "pred_error_vs_actual_spearman",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("sequence_short")
    )
    sequence_summary.to_csv(output_dir / "sequence_validity_summary.csv", index=False)

    scenario_summary = (
        run_correlations.groupby(["run_kind", "base_scenario"], dropna=False)[
            [
                "failure_vs_actual_pearson",
                "failure_vs_actual_spearman",
                "pred_error_vs_actual_pearson",
                "pred_error_vs_actual_spearman",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["run_kind", "base_scenario"])
    )
    scenario_summary.to_csv(output_dir / "scenario_validity_summary.csv", index=False)


def _threshold_metrics(frame_data: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    score_map = _score_definitions(frame_data)
    rows: List[Dict[str, object]] = []
    actual_error = frame_data["actual_pose_error"].to_numpy(dtype=float)

    for threshold in thresholds:
        y_true = (actual_error >= threshold).astype(int)
        for score_name, score_series in score_map.items():
            metrics = _evaluate_score(y_true, score_series.to_numpy(dtype=float), score_name)
            rows.append(
                {
                    "failure_threshold": float(threshold),
                    "score_name": score_name,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _calibration_bins(probabilities: np.ndarray, y_true: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: List[Dict[str, float]] = []
    for idx in range(bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == bins - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "bin_index": idx,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "mean_confidence": float("nan"),
                    "observed_failure_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin_index": idx,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": float(np.mean(probabilities[mask])),
                "observed_failure_rate": float(np.mean(y_true[mask])),
            }
        )
    return pd.DataFrame(rows)


def _plot_scatter(frame_data: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        frame_data["actual_pose_error"],
        frame_data["failure_probability"],
        s=8,
        alpha=0.35,
        c=np.where(frame_data["run_kind"] == "degraded", "#c44536", "#2a6f97"),
    )
    axes[0].set_xlabel("actual_pose_error (m)")
    axes[0].set_ylabel("failure_probability")
    axes[0].set_title("Failure Probability vs Actual Pose Error")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(
        frame_data["actual_pose_error"],
        frame_data["predicted_pose_error"],
        s=8,
        alpha=0.35,
        c=np.where(frame_data["run_kind"] == "degraded", "#d17b0f", "#2a6f97"),
    )
    axes[1].set_xlabel("actual_pose_error (m)")
    axes[1].set_ylabel("predicted_pose_error")
    axes[1].set_title("Predicted Pose Error vs Actual Pose Error")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "model_vs_actual_scatter.png")
    plt.close(fig)


def _plot_sequence_correlations(sequence_summary: pd.DataFrame, output_dir: Path):
    if sequence_summary.empty:
        return
    x = np.arange(len(sequence_summary))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    axes[0].bar(x - width / 2, sequence_summary["failure_vs_actual_pearson"], width=width, label="failure_probability")
    axes[0].bar(x + width / 2, sequence_summary["pred_error_vs_actual_pearson"], width=width, label="predicted_pose_error")
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[0].set_title("Pearson Correlation by Sequence")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sequence_summary["sequence_short"], rotation=0)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - width / 2, sequence_summary["failure_vs_actual_spearman"], width=width, label="failure_probability")
    axes[1].bar(x + width / 2, sequence_summary["pred_error_vs_actual_spearman"], width=width, label="predicted_pose_error")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    axes[1].set_title("Spearman Correlation by Sequence")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sequence_summary["sequence_short"], rotation=0)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "sequence_correlation_overview.png")
    plt.close(fig)


def _plot_roc(frame_data: pd.DataFrame, threshold: float, output_dir: Path):
    y_true = (frame_data["actual_pose_error"].to_numpy(dtype=float) >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return
    score_map = _score_definitions(frame_data)
    selected = [
        "model_failure_probability",
        "model_predicted_pose_error",
        "heuristic_inlier_ratio_risk",
        "heuristic_pose_residual_risk",
        "heuristic_num_inliers_risk",
    ]
    fig, ax = plt.subplots(figsize=(7, 6))
    for score_name in selected:
        if score_name not in score_map:
            continue
        score = score_map[score_name].to_numpy(dtype=float)
        mask = np.isfinite(score)
        if len(np.unique(y_true[mask])) < 2:
            continue
        curve = _binary_curve(y_true[mask], score[mask])
        fpr = np.r_[0.0, curve["fpr"], 1.0]
        tpr = np.r_[0.0, curve["tpr"], 1.0]
        auc = _roc_auc_from_curve(curve["fpr"], curve["tpr"])
        ax.plot(fpr, tpr, label=f"{score_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Comparison @ failure threshold {threshold:.2f} m")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_comparison_t{_slug(threshold)}.png")
    plt.close(fig)


def _plot_calibration(frame_data: pd.DataFrame, threshold: float, bins_frame: pd.DataFrame, output_dir: Path):
    y_true = (frame_data["actual_pose_error"].to_numpy(dtype=float) >= threshold).astype(int)
    probabilities = frame_data["failure_probability"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    valid = bins_frame["count"] > 0
    ax.plot(
        bins_frame.loc[valid, "mean_confidence"],
        bins_frame.loc[valid, "observed_failure_rate"],
        marker="o",
        color="#c44536",
        linewidth=2,
    )
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Observed failure rate")
    ax.set_title(f"Calibration @ failure threshold {threshold:.2f} m")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"calibration_t{_slug(threshold)}.png")
    plt.close(fig)


def _write_summary(
    frame_data: pd.DataFrame,
    run_correlations: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    output_dir: Path,
    summary_threshold: float,
):
    summary_path = output_dir / "validity_summary.txt"
    model_failure = threshold_metrics[
        (threshold_metrics["failure_threshold"] == summary_threshold)
        & (threshold_metrics["score_name"] == "model_failure_probability")
    ]
    model_error = threshold_metrics[
        (threshold_metrics["failure_threshold"] == summary_threshold)
        & (threshold_metrics["score_name"] == "model_predicted_pose_error")
    ]
    heuristic_rows = threshold_metrics[
        (threshold_metrics["failure_threshold"] == summary_threshold)
        & threshold_metrics["score_name"].str.startswith("heuristic_")
    ].sort_values("roc_auc", ascending=False)
    best_heuristic = heuristic_rows.iloc[0] if not heuristic_rows.empty else None

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Model validity benchmark\n")
        handle.write(f"num_runs: {len(run_correlations)}\n")
        handle.write(f"num_frames: {len(frame_data)}\n")
        handle.write(f"num_sequences: {frame_data['sequence'].nunique()}\n")
        handle.write(f"num_baseline_runs: {int((run_correlations['run_kind'] == 'baseline').sum())}\n")
        handle.write(f"num_degraded_runs: {int((run_correlations['run_kind'] == 'degraded').sum())}\n")
        handle.write(
            f"overall_failure_vs_actual_pearson: {_safe_corr(frame_data['failure_probability'], frame_data['actual_pose_error']):.6f}\n"
        )
        handle.write(
            f"overall_failure_vs_actual_spearman: {_safe_corr(frame_data['failure_probability'], frame_data['actual_pose_error'], method='spearman'):.6f}\n"
        )
        handle.write(
            f"overall_pred_error_vs_actual_pearson: {_safe_corr(frame_data['predicted_pose_error'], frame_data['actual_pose_error']):.6f}\n"
        )
        handle.write(
            f"overall_pred_error_vs_actual_spearman: {_safe_corr(frame_data['predicted_pose_error'], frame_data['actual_pose_error'], method='spearman'):.6f}\n"
        )
        handle.write(
            f"mean_run_failure_vs_actual_pearson: {run_correlations['failure_vs_actual_pearson'].mean():.6f}\n"
        )
        handle.write(
            f"positive_failure_correlation_runs: {int((run_correlations['failure_vs_actual_pearson'] > 0).sum())}/{len(run_correlations)}\n"
        )
        handle.write(f"summary_failure_threshold: {summary_threshold:.2f}\n")

        if not model_failure.empty:
            row = model_failure.iloc[0]
            handle.write(
                f"model_failure_probability_roc_auc: {row['roc_auc']:.6f}\n"
                f"model_failure_probability_ap: {row['average_precision']:.6f}\n"
                f"model_failure_probability_fixed_f1: {row['fixed_f1']:.6f}\n"
                f"model_failure_probability_brier: {row['brier_score']:.6f}\n"
            )
        if not model_error.empty:
            row = model_error.iloc[0]
            handle.write(
                f"model_predicted_pose_error_roc_auc: {row['roc_auc']:.6f}\n"
                f"model_predicted_pose_error_ap: {row['average_precision']:.6f}\n"
                f"model_predicted_pose_error_best_f1: {row['best_f1']:.6f}\n"
            )
        if best_heuristic is not None:
            handle.write(
                f"best_heuristic_score: {best_heuristic['score_name']}\n"
                f"best_heuristic_roc_auc: {best_heuristic['roc_auc']:.6f}\n"
                f"best_heuristic_best_f1: {best_heuristic['best_f1']:.6f}\n"
            )


def run_benchmark(
    sweep_results_path: str,
    output_dir: str,
    failure_thresholds: List[float],
    summary_threshold: float,
    include_baseline: bool,
):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    frame_data = _build_frame_dataset(Path(sweep_results_path), include_baseline=include_baseline)
    frame_data.to_csv(output_root / "frame_level_validity_data.csv", index=False)

    run_correlations = _run_level_correlations(frame_data)
    run_correlations.to_csv(output_root / "run_level_correlations.csv", index=False)
    _aggregate_correlations(run_correlations, output_root)

    threshold_metrics = _threshold_metrics(frame_data, failure_thresholds)
    threshold_metrics.to_csv(output_root / "threshold_metrics.csv", index=False)

    for threshold in failure_thresholds:
        y_true = (frame_data["actual_pose_error"].to_numpy(dtype=float) >= threshold).astype(int)
        bins_frame = _calibration_bins(frame_data["failure_probability"].to_numpy(dtype=float), y_true, bins=10)
        bins_frame.to_csv(output_root / f"calibration_bins_t{_slug(threshold)}.csv", index=False)
        _plot_roc(frame_data, threshold, output_root)
        _plot_calibration(frame_data, threshold, bins_frame, output_root)

    _plot_scatter(frame_data, output_root)
    sequence_summary = pd.read_csv(output_root / "sequence_validity_summary.csv")
    _plot_sequence_correlations(sequence_summary, output_root)
    _write_summary(frame_data, run_correlations, threshold_metrics, output_root, summary_threshold)


def main():
    parser = argparse.ArgumentParser(description="Run model validity benchmark over multisequence outputs")
    parser.add_argument("--sweep-results", required=True, help="Path to sweep_results.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for validity benchmark outputs")
    parser.add_argument(
        "--failure-thresholds",
        default="0.3,1.0,3.0",
        help="Comma-separated actual pose error thresholds (meters) used to define failure labels",
    )
    parser.add_argument(
        "--summary-threshold",
        type=float,
        default=3.0,
        help="Which failure threshold to emphasize in validity_summary.txt and calibration plots",
    )
    parser.add_argument(
        "--exclude-baseline",
        action="store_true",
        help="Only benchmark degraded runs",
    )
    args = parser.parse_args()

    thresholds = [float(item) for item in args.failure_thresholds.split(",") if item.strip()]
    if args.summary_threshold not in thresholds:
        thresholds = sorted(set(thresholds + [args.summary_threshold]))

    run_benchmark(
        sweep_results_path=args.sweep_results,
        output_dir=args.output_dir,
        failure_thresholds=thresholds,
        summary_threshold=args.summary_threshold,
        include_baseline=not args.exclude_baseline,
    )


if __name__ == "__main__":
    main()
