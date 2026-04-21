#!/usr/bin/env python3
"""
Plot percentile-threshold experiment figures for Self-Aware VIO-SLAM.

Outputs:
  1. ROC overlay for multiple percentile experiments
  2. AUC vs percentile curve
  3. Failure timeline from an offline unified demo prediction CSV
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent.parent / ".mplcache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from self_aware_slam.src.data.dataset_builder import load_dataset
from self_aware_slam.src.models.failure_predictor import build_model
from self_aware_slam.src.utils.config_loader import load_config


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _binary_curve(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    mask = np.isfinite(scores)
    y_true = y_true[mask].astype(int)
    scores = scores[mask].astype(float)
    if len(scores) == 0:
        empty = np.array([], dtype=float)
        return {"thresholds": empty, "tpr": empty, "fpr": empty, "precision": empty, "recall": empty}

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

    tpr = tps / positives if positives > 0 else np.full_like(tps, np.nan)
    fpr = fps / negatives if negatives > 0 else np.full_like(fps, np.nan)
    precision = tps / np.clip(tps + fps, 1e-12, None)
    recall = tpr.copy()

    return {
        "thresholds": thresholds,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
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


def _load_model(checkpoint_path: Path, dataset: Dict) -> tuple[torch.nn.Module, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", load_config())
    model = build_model(
        checkpoint_config,
        n_features=dataset["test"]["X"].shape[-1],
        window_size=dataset.get("window_size", checkpoint_config["temporal"]["window_size"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint_config


def _predict_test_split(checkpoint_path: Path, dataset_path: Path) -> Dict[str, object]:
    dataset = load_dataset(str(dataset_path))
    model, checkpoint_config = _load_model(checkpoint_path, dataset)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    model = model.to(device)

    X = torch.FloatTensor(dataset["test"]["X"]).to(device)
    with torch.no_grad():
        failure_prob, pred_error = model(X)

    y_true = dataset["test"]["y_failure"].astype(np.float32)
    y_error = dataset["test"]["y_error"].astype(np.float32)
    scores = failure_prob.squeeze(-1).cpu().numpy().astype(np.float32)
    pred_error = pred_error.squeeze(-1).cpu().numpy().astype(np.float32)
    curve = _binary_curve(y_true, scores)
    auc = _roc_auc_from_curve(curve["fpr"], curve["tpr"])
    pr_auc = _average_precision_from_curve(curve["recall"], curve["precision"])

    return {
        "dataset": dataset,
        "checkpoint_config": checkpoint_config,
        "y_true": y_true,
        "scores": scores,
        "y_error": y_error,
        "pred_error": pred_error,
        "curve": curve,
        "auc": auc,
        "pr_auc": pr_auc,
    }


def _write_prediction_dump(output_dir: Path, name: str, payload: Dict[str, object]) -> Path:
    frame = pd.DataFrame(
        {
            "y_failure": payload["y_true"],
            "failure_probability": payload["scores"],
            "y_error": payload["y_error"],
            "predicted_pose_error": payload["pred_error"],
        }
    )
    out_path = output_dir / f"{_slug(name)}_test_predictions.csv"
    frame.to_csv(out_path, index=False)
    return out_path


def _plot_roc_overlay(summaries: List[Dict[str, object]], output_path: Path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for item in summaries:
        curve = item["curve"]
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=2,
            label=f"{item['label']} (AUC={item['auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Percentile Threshold Experiments")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_auc_vs_percentile(summary_df: pd.DataFrame, output_path: Path):
    ordered = summary_df.sort_values("percentile")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(ordered["percentile"], ordered["auc"], marker="o", linewidth=2, color="tab:blue")
    for row in ordered.itertuples(index=False):
        ax.annotate(
            f"{row.auc:.3f}",
            (row.percentile, row.auc),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    ax.set_xlabel("Percentile Threshold")
    ax.set_ylabel("ROC AUC")
    ax.set_title("AUC vs Percentile")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pr_auc_vs_percentile(summary_df: pd.DataFrame, output_path: Path):
    ordered = summary_df.sort_values("percentile")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(ordered["percentile"], ordered["pr_auc"], marker="o", linewidth=2, color="tab:green")
    for row in ordered.itertuples(index=False):
        ax.annotate(
            f"{row.pr_auc:.3f}",
            (row.percentile, row.pr_auc),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    ax.set_xlabel("Percentile Threshold")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC vs Percentile")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_failure_timeline(
    predictions_path: Path,
    output_path: Path,
    title: str,
    error_threshold: float,
    probability_threshold: float,
):
    frame = pd.read_csv(predictions_path)
    if "actual_pose_error" not in frame.columns:
        raise ValueError(f"{predictions_path} does not contain actual_pose_error")

    if "timestamp" in frame.columns:
        x = frame["timestamp"].astype(float).to_numpy()
        x = x - x[0]
        x_label = "Time Since Start (s)"
    else:
        x = np.arange(len(frame), dtype=float)
        x_label = "Frame Index"

    first_error_idx = None
    if np.isfinite(error_threshold):
        error_hits = np.where(frame["actual_pose_error"].to_numpy() >= error_threshold)[0]
        if len(error_hits) > 0:
            first_error_idx = int(error_hits[0])

    prob_hits = np.where(frame["failure_probability"].to_numpy() >= probability_threshold)[0]
    first_prob_idx = int(prob_hits[0]) if len(prob_hits) > 0 else None

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, frame["actual_pose_error"], color="tab:green", linewidth=2, label="actual_pose_error")
    if "predicted_pose_error" in frame.columns:
        axes[0].plot(x, frame["predicted_pose_error"], color="tab:orange", linewidth=1.8, label="predicted_pose_error")
    axes[0].axhline(error_threshold, linestyle="--", color="tab:red", alpha=0.7, label=f"error threshold={error_threshold:g}")
    axes[0].set_ylabel("Pose Error (m)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, frame["failure_probability"], color="tab:red", linewidth=2, label="failure_probability")
    if "confidence_score" in frame.columns:
        axes[1].plot(x, frame["confidence_score"], color="tab:blue", linewidth=1.8, alpha=0.8, label="confidence_score")
    axes[1].axhline(probability_threshold, linestyle="--", color="black", alpha=0.5, label=f"probability threshold={probability_threshold:g}")
    axes[1].set_ylabel("Probability / Score")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    prob_min = float(frame["failure_probability"].min())
    prob_max = float(frame["failure_probability"].max())
    padding = max(0.01, 0.2 * (prob_max - prob_min))
    y_low = max(0.0, prob_min - padding)
    y_high = min(1.0, prob_max + padding)
    if y_high - y_low < 0.05:
        center = 0.5 * (prob_min + prob_max)
        y_low = max(0.0, center - 0.03)
        y_high = min(1.0, center + 0.03)

    axes[2].plot(x, frame["failure_probability"], color="tab:red", linewidth=2, label="failure_probability (zoom)")
    axes[2].axhline(probability_threshold, linestyle="--", color="black", alpha=0.5, label=f"probability threshold={probability_threshold:g}")
    axes[2].set_ylim(y_low, y_high)
    axes[2].set_ylabel("Failure Prob. (zoom)")
    axes[2].set_xlabel(x_label)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    if first_prob_idx is not None:
        for axis in axes:
            axis.axvline(x[first_prob_idx], color="tab:red", linestyle=":", alpha=0.7)
        axes[1].annotate(
            "probability rises",
            (x[first_prob_idx], frame["failure_probability"].iloc[first_prob_idx]),
            textcoords="offset points",
            xytext=(8, 10),
            ha="left",
            color="tab:red",
        )

    if first_error_idx is not None:
        for axis in axes:
            axis.axvline(x[first_error_idx], color="tab:green", linestyle=":", alpha=0.7)
        axes[0].annotate(
            "error crosses threshold",
            (x[first_error_idx], frame["actual_pose_error"].iloc[first_error_idx]),
            textcoords="offset points",
            xytext=(8, 10),
            ha="left",
            color="tab:green",
        )

    if first_prob_idx is not None and first_error_idx is not None:
        lead_frames = first_error_idx - first_prob_idx
        lead_seconds = float(x[first_error_idx] - x[first_prob_idx])
        axes[2].text(
            0.01,
            0.03,
            f"lead = {lead_frames} frames ({lead_seconds:.2f}s)",
            transform=axes[2].transAxes,
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _collect_prediction_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/reliability_predictions.csv"))


def _select_timeline_candidate(
    prediction_files: List[Path],
    error_threshold: float,
    probability_threshold: float,
) -> Optional[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for path in prediction_files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"actual_pose_error", "failure_probability"}
        if not required.issubset(frame.columns) or len(frame) < 5:
            continue

        err = frame["actual_pose_error"].to_numpy(dtype=float)
        prob = frame["failure_probability"].to_numpy(dtype=float)
        first_error_idx = next((i for i, value in enumerate(err) if value >= error_threshold), None)
        first_prob_idx = next((i for i, value in enumerate(prob) if value >= probability_threshold), None)
        if first_error_idx is None or first_error_idx <= 0:
            continue

        if first_prob_idx is None:
            lead_frames = -9999
        else:
            lead_frames = first_error_idx - first_prob_idx

        candidates.append(
            {
                "path": path,
                "lead_frames": lead_frames,
                "prob_std": float(np.std(prob)),
                "prob_range": float(np.max(prob) - np.min(prob)),
                "first_error_idx": first_error_idx,
                "first_prob_idx": first_prob_idx,
                "error_start": float(err[0]),
                "error_max": float(np.max(err)),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda row: (row["lead_frames"], row["prob_std"], row["prob_range"], row["error_max"]),
        reverse=True,
    )
    return candidates[0]


def _build_lead_summary(
    prediction_files: List[Path],
    error_threshold: float,
    probability_threshold: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in prediction_files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        required = {"actual_pose_error", "failure_probability"}
        if not required.issubset(frame.columns) or len(frame) < 2:
            continue

        err = frame["actual_pose_error"].to_numpy(dtype=float)
        prob = frame["failure_probability"].to_numpy(dtype=float)
        first_error_idx = next((i for i, value in enumerate(err) if value >= error_threshold), None)
        first_prob_idx = next((i for i, value in enumerate(prob) if value >= probability_threshold), None)
        if first_error_idx is None:
            continue

        if "timestamp" in frame.columns:
            ts = frame["timestamp"].astype(float).to_numpy()
            ts = ts - ts[0]
        else:
            ts = np.arange(len(frame), dtype=float)

        if first_prob_idx is None:
            lead_frames = np.nan
            lead_seconds = np.nan
        else:
            lead_frames = float(first_error_idx - first_prob_idx)
            lead_seconds = float(ts[first_error_idx] - ts[first_prob_idx])

        rows.append(
            {
                "path": str(path),
                "case": path.parent.parent.name,
                "n_frames": int(len(frame)),
                "error_start": float(err[0]),
                "error_max": float(np.max(err)),
                "prob_min": float(np.min(prob)),
                "prob_max": float(np.max(prob)),
                "prob_std": float(np.std(prob)),
                "first_prob_cross_idx": first_prob_idx,
                "first_error_cross_idx": first_error_idx,
                "lead_frames": lead_frames,
                "lead_seconds": lead_seconds,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "path",
                "case",
                "n_frames",
                "error_start",
                "error_max",
                "prob_min",
                "prob_max",
                "prob_std",
                "first_prob_cross_idx",
                "first_error_cross_idx",
                "lead_frames",
                "lead_seconds",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["lead_frames", "prob_std", "error_max"],
        ascending=[False, False, False],
        na_position="last",
    )


def _write_lead_statistics(summary: pd.DataFrame, output_dir: Path):
    summary_path = output_dir / "lead_time_summary.csv"
    summary.to_csv(summary_path, index=False)

    finite = summary[np.isfinite(summary["lead_frames"])].copy()
    positive = finite[finite["lead_frames"] > 0]
    nonnegative = finite[finite["lead_frames"] >= 0]
    ge3 = finite[finite["lead_frames"] >= 3]
    ge5 = finite[finite["lead_frames"] >= 5]
    stats_lines = [
        f"n_cases_with_error_cross: {len(summary)}",
        f"n_cases_with_probability_cross: {len(finite)}",
        f"mean_lead_frames: {float(finite['lead_frames'].mean()) if len(finite) else float('nan'):.4f}",
        f"median_lead_frames: {float(finite['lead_frames'].median()) if len(finite) else float('nan'):.4f}",
        f"mean_lead_seconds: {float(finite['lead_seconds'].mean()) if len(finite) else float('nan'):.4f}",
        f"median_lead_seconds: {float(finite['lead_seconds'].median()) if len(finite) else float('nan'):.4f}",
        f"pct_cases_lead_gt_0: {100.0 * len(positive) / len(finite) if len(finite) else float('nan'):.2f}",
        f"pct_cases_lead_ge_0: {100.0 * len(nonnegative) / len(finite) if len(finite) else float('nan'):.2f}",
        f"pct_cases_lead_ge_3: {100.0 * len(ge3) / len(finite) if len(finite) else float('nan'):.2f}",
        f"pct_cases_lead_ge_5: {100.0 * len(ge5) / len(finite) if len(finite) else float('nan'):.2f}",
    ]
    (output_dir / "lead_time_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")

    if len(finite):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        bins = np.arange(np.floor(finite["lead_frames"].min()) - 0.5, np.ceil(finite["lead_frames"].max()) + 1.5, 1.0)
        if len(bins) < 2:
            bins = np.array([finite["lead_frames"].iloc[0] - 0.5, finite["lead_frames"].iloc[0] + 0.5])
        ax.hist(finite["lead_frames"], bins=bins, color="tab:purple", alpha=0.85, edgecolor="white")
        ax.axvline(float(finite["lead_frames"].mean()), color="tab:red", linestyle="--", label=f"mean={finite['lead_frames'].mean():.2f}")
        ax.axvline(float(finite["lead_frames"].median()), color="tab:blue", linestyle=":", label=f"median={finite['lead_frames'].median():.2f}")
        ax.set_xlabel("Lead Time (frames)")
        ax.set_ylabel("Count")
        ax.set_title("Lead-Time Distribution")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "lead_time_histogram.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot ROC/AUC/timeline figures for percentile experiments.")
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=4,
        metavar=("LABEL", "PERCENTILE", "CHECKPOINT", "DATASET"),
        help="Add one experiment as: LABEL PERCENTILE CHECKPOINT DATASET",
        required=True,
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated figures and CSV summaries")
    parser.add_argument("--timeline-predictions", default=None, help="Path to reliability_predictions.csv for timeline plot")
    parser.add_argument("--timeline-search-root", default=None, help="Search this directory for the best timeline candidate")
    parser.add_argument("--timeline-title", default="Failure Timeline", help="Title for the timeline figure")
    parser.add_argument("--timeline-error-threshold", type=float, default=1.0, help="Actual pose error threshold to mark")
    parser.add_argument("--timeline-probability-threshold", type=float, default=0.5, help="Failure probability threshold to mark")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []

    for label, percentile_text, checkpoint_text, dataset_text in args.experiment:
        percentile = float(percentile_text)
        checkpoint_path = Path(checkpoint_text)
        dataset_path = Path(dataset_text)
        payload = _predict_test_split(checkpoint_path, dataset_path)
        pred_dump = _write_prediction_dump(output_dir, label, payload)
        summaries.append(
            {
                "label": label,
                "percentile": percentile,
                "auc": payload["auc"],
                "pr_auc": payload["pr_auc"],
                "curve": payload["curve"],
            }
        )
        table_rows.append(
            {
                "label": label,
                "percentile": percentile,
                "auc": payload["auc"],
                "pr_auc": payload["pr_auc"],
                "n_test": int(len(payload["y_true"])),
                "failure_rate": float(np.mean(payload["y_true"])),
                "checkpoint_path": str(checkpoint_path),
                "dataset_path": str(dataset_path),
                "prediction_dump_path": str(pred_dump),
            }
        )

    summary_df = pd.DataFrame(table_rows).sort_values("percentile")
    summary_df.to_csv(output_dir / "percentile_experiment_summary.csv", index=False)

    _plot_roc_overlay(summaries, output_dir / "roc_curve_percentiles.png")
    _plot_auc_vs_percentile(summary_df, output_dir / "auc_vs_percentile.png")
    _plot_pr_auc_vs_percentile(summary_df, output_dir / "pr_auc_vs_percentile.png")

    timeline_predictions = Path(args.timeline_predictions) if args.timeline_predictions else None
    timeline_selected = None
    searched_files: List[Path] = []
    if args.timeline_search_root:
        searched_files = _collect_prediction_files(Path(args.timeline_search_root))
        candidate = _select_timeline_candidate(
            searched_files,
            error_threshold=args.timeline_error_threshold,
            probability_threshold=args.timeline_probability_threshold,
        )
        if candidate is not None:
            timeline_predictions = candidate["path"]
            timeline_selected = candidate

    if timeline_predictions:
        _plot_failure_timeline(
            predictions_path=timeline_predictions,
            output_path=output_dir / "failure_timeline.png",
            title=args.timeline_title if timeline_selected is None else f"{args.timeline_title} | {timeline_selected['path'].parent.parent.name}",
            error_threshold=args.timeline_error_threshold,
            probability_threshold=args.timeline_probability_threshold,
        )

    if searched_files:
        lead_summary = _build_lead_summary(
            searched_files,
            error_threshold=args.timeline_error_threshold,
            probability_threshold=args.timeline_probability_threshold,
        )
        _write_lead_statistics(lead_summary, output_dir)

    print(f"summary_csv: {output_dir / 'percentile_experiment_summary.csv'}")
    print(f"roc_plot: {output_dir / 'roc_curve_percentiles.png'}")
    print(f"auc_plot: {output_dir / 'auc_vs_percentile.png'}")
    print(f"pr_auc_plot: {output_dir / 'pr_auc_vs_percentile.png'}")
    if timeline_predictions:
        print(f"timeline_plot: {output_dir / 'failure_timeline.png'}")
        print(f"timeline_source: {timeline_predictions}")
        if timeline_selected is not None:
            print(
                "timeline_candidate:"
                f" lead_frames={timeline_selected['lead_frames']},"
                f" prob_std={timeline_selected['prob_std']:.6f},"
                f" prob_range={timeline_selected['prob_range']:.6f}"
            )
    if searched_files:
        print(f"lead_summary_csv: {output_dir / 'lead_time_summary.csv'}")
        print(f"lead_stats_txt: {output_dir / 'lead_time_stats.txt'}")
        if (output_dir / 'lead_time_histogram.png').exists():
            print(f"lead_histogram: {output_dir / 'lead_time_histogram.png'}")


if __name__ == "__main__":
    main()
