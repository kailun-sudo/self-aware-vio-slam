#!/usr/bin/env python3
"""
Small head-to-head analysis between learned scores and heuristic baselines.

This is narrower than the full validity benchmark. The focus is:

1. Does the learned model win anywhere at all?
2. Which score wins overall / by scenario / by threshold?
3. Is there any local regime where the learned score has an advantage?
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

def _parse_thresholds(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _rank_normalize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = pd.Series(np.nan, index=series.index, dtype=float)
    mask = np.isfinite(values.to_numpy(dtype=float))
    if not np.any(mask):
        return ranks
    ranked = values[mask].rank(method="average", pct=True).astype(float)
    ranks.loc[mask] = ranked.to_numpy(dtype=float)
    return ranks


def _rank_average(a: pd.Series, b: pd.Series) -> pd.Series:
    a_rank = _rank_normalize(a)
    b_rank = _rank_normalize(b)
    return 0.5 * (a_rank + b_rank)


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
        return predictions.merge(metrics, on=keys, how="left", suffixes=("", "_metric"))
    metrics_tail = metrics.tail(len(predictions)).reset_index(drop=True)
    return pd.concat([predictions.reset_index(drop=True), metrics_tail], axis=1)


def _load_run_frame(run_dir: Path, metadata: Dict[str, object]) -> pd.DataFrame:
    predictions = _read_csv(run_dir / "reliability_predictions.csv")
    metrics = _read_csv(run_dir / "slam_metrics.csv")
    merged = _merge_predictions_and_metrics(predictions, metrics)
    for key, value in metadata.items():
        merged[key] = value
    return merged


def _build_frame_dataset(sweep_results_path: Path, include_baseline: bool = True) -> pd.DataFrame:
    results = _read_csv(sweep_results_path)
    frames: List[pd.DataFrame] = []
    seen_baselines: set[str] = set()

    for row in results.to_dict(orient="records"):
        sequence = row["sequence"]
        baseline_dir = _infer_baseline_dir(row["comparison_gui_path"])
        degraded_dir = _infer_degraded_dir(row["comparison_gui_path"])

        if include_baseline and sequence not in seen_baselines:
            frames.append(
                _load_run_frame(
                    baseline_dir,
                    {
                        "sequence": sequence,
                        "run_kind": "baseline",
                        "scenario": "baseline",
                        "base_scenario": "baseline",
                        "severity": 0.0,
                    },
                )
            )
            seen_baselines.add(sequence)

        frames.append(
            _load_run_frame(
                degraded_dir,
                {
                    "sequence": sequence,
                    "run_kind": "degraded",
                    "scenario": row["scenario"],
                    "base_scenario": row.get("base_scenario", row["scenario"]),
                    "severity": float(row.get("severity", 0.0)),
                },
            )
        )

    return pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def _binary_curve(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    mask = np.isfinite(scores)
    y_true = y_true[mask].astype(int)
    scores = scores[mask].astype(float)
    if len(scores) == 0:
        empty = np.array([], dtype=float)
        return {"thresholds": empty, "precision": empty, "recall": empty, "tpr": empty, "fpr": empty}

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
    return {"thresholds": thresholds, "precision": precision, "recall": recall, "tpr": tpr, "fpr": fpr}


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
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    mask = np.isfinite(prob)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask].astype(float)
    prob = prob[mask].astype(float)
    return float(np.mean((prob - y_true) ** 2))


def _best_f1(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    curve = _binary_curve(y_true, scores)
    thresholds = curve["thresholds"]
    if len(thresholds) == 0:
        return {"best_f1": float("nan"), "best_threshold": float("nan"), "best_precision": float("nan"), "best_recall": float("nan")}
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


def _score_definitions(frame_data: pd.DataFrame) -> Dict[str, pd.Series]:
    scores: Dict[str, pd.Series] = {
        "model_failure_probability": frame_data["failure_probability"],
        "model_predicted_pose_error": frame_data["predicted_pose_error"],
    }
    if "inlier_ratio" in frame_data.columns:
        scores["heuristic_inlier_ratio_risk"] = 1.0 - frame_data["inlier_ratio"]
    if "pose_optimization_residual" in frame_data.columns:
        scores["heuristic_pose_residual_risk"] = frame_data["pose_optimization_residual"]
    if "num_inliers" in frame_data.columns:
        scores["heuristic_num_inliers_risk"] = -frame_data["num_inliers"]
    if "mean_epipolar_error" in frame_data.columns:
        scores["heuristic_epipolar_error_risk"] = frame_data["mean_epipolar_error"]
    if "model_failure_probability" in scores and "heuristic_epipolar_error_risk" in scores:
        scores["fusion_failure_epipolar_rankavg"] = _rank_average(
            scores["model_failure_probability"],
            scores["heuristic_epipolar_error_risk"],
        )
    if "model_failure_probability" in scores and "heuristic_num_inliers_risk" in scores:
        scores["fusion_failure_num_inliers_rankavg"] = _rank_average(
            scores["model_failure_probability"],
            scores["heuristic_num_inliers_risk"],
        )
    if "model_predicted_pose_error" in scores and "heuristic_epipolar_error_risk" in scores:
        scores["fusion_pred_error_epipolar_rankavg"] = _rank_average(
            scores["model_predicted_pose_error"],
            scores["heuristic_epipolar_error_risk"],
        )
    if "model_predicted_pose_error" in scores and "heuristic_num_inliers_risk" in scores:
        scores["fusion_pred_error_num_inliers_rankavg"] = _rank_average(
            scores["model_predicted_pose_error"],
            scores["heuristic_num_inliers_risk"],
        )
    return scores


def _selected_scores(score_map: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    preferred = [
        "model_failure_probability",
        "model_predicted_pose_error",
        "fusion_failure_epipolar_rankavg",
        "fusion_failure_num_inliers_rankavg",
        "fusion_pred_error_epipolar_rankavg",
        "fusion_pred_error_num_inliers_rankavg",
        "heuristic_num_inliers_risk",
        "heuristic_inlier_ratio_risk",
        "heuristic_pose_residual_risk",
        "heuristic_epipolar_error_risk",
    ]
    return {name: score_map[name] for name in preferred if name in score_map}


def _evaluate_group(frame_data: pd.DataFrame, thresholds: Iterable[float]) -> pd.DataFrame:
    actual_error = frame_data["actual_pose_error"].to_numpy(dtype=float)
    score_map = _selected_scores(_score_definitions(frame_data))
    rows: List[Dict[str, float]] = []
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


def _winner_table(metrics: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in metrics.groupby(group_columns, dropna=False):
        if isinstance(keys, tuple):
            row = dict(zip(group_columns, keys))
        else:
            row = {group_columns[0]: keys}
        ranked = group.sort_values(["roc_auc", "average_precision"], ascending=False, na_position="last")
        winner = ranked.iloc[0]
        row.update(
            {
                "winner_score": winner["score_name"],
                "winner_roc_auc": float(winner["roc_auc"]),
                "winner_ap": float(winner["average_precision"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_overall_auc(metrics: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = metrics.pivot(index="score_name", columns="failure_threshold", values="roc_auc")
    fig, ax = plt.subplots(figsize=(9, 5))
    for score_name, row in pivot.iterrows():
        ax.plot(pivot.columns.astype(float), row.values.astype(float), marker="o", label=score_name)
    ax.set_title("ROC-AUC: model vs heuristics")
    ax.set_xlabel("failure threshold (m)")
    ax.set_ylabel("ROC-AUC")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "head_to_head_auc.png", dpi=180)
    plt.close(fig)


def _plot_scenario_auc(metrics: pd.DataFrame, threshold: float, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = metrics[np.isclose(metrics["failure_threshold"], threshold)]
    if subset.empty:
        return
    pivot = subset.pivot(index="base_scenario", columns="score_name", values="roc_auc")
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(pivot.index))
    scores = list(pivot.columns)
    width = 0.8 / max(len(scores), 1)
    for idx, score_name in enumerate(scores):
        ax.bar(x + idx * width - 0.4 + width / 2, pivot[score_name].values, width=width, label=score_name)
    ax.set_title(f"ROC-AUC by scenario @ {threshold:.1f}m")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=15)
    ax.set_ylabel("ROC-AUC")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"head_to_head_auc_by_scenario_t{str(threshold).replace('.', 'p')}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare learned scores against heuristics.")
    parser.add_argument("--sweep-results", required=True, help="Path to sweep_results.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    parser.add_argument(
        "--failure-thresholds",
        default="3.0,6.0,9.0",
        help="Comma-separated actual-error thresholds in meters.",
    )
    parser.add_argument(
        "--scenario-threshold",
        type=float,
        default=6.0,
        help="Threshold to use for the by-scenario comparison plot.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write summaries/csv only and skip matplotlib outputs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.failure_thresholds)
    scenario_slug = str(args.scenario_threshold).replace(".", "p")
    frame_data = _build_frame_dataset(Path(args.sweep_results), include_baseline=True)

    overall_metrics = _evaluate_group(frame_data, thresholds)
    overall_metrics.to_csv(output_dir / "overall_head_to_head.csv", index=False)

    scenario_rows: List[pd.DataFrame] = []
    for scenario, group in frame_data.groupby("base_scenario", dropna=False):
        table = _evaluate_group(group, thresholds)
        table["base_scenario"] = scenario
        scenario_rows.append(table)
    scenario_metrics = pd.concat(scenario_rows, ignore_index=True)
    scenario_metrics.to_csv(output_dir / "scenario_head_to_head.csv", index=False)

    overall_winners = _winner_table(overall_metrics, ["failure_threshold"])
    overall_winners.to_csv(output_dir / "overall_winners.csv", index=False)

    scenario_winners = _winner_table(scenario_metrics, ["failure_threshold", "base_scenario"])
    scenario_winners.to_csv(output_dir / "scenario_winners.csv", index=False)

    if not args.skip_plots:
        _plot_overall_auc(overall_metrics, output_dir)
        _plot_scenario_auc(scenario_metrics, args.scenario_threshold, output_dir)

    lines = [
        "Model vs heuristic head-to-head",
        f"sweep_results: {args.sweep_results}",
        "",
        "Overall winners by threshold:",
    ]
    for row in overall_winners.sort_values("failure_threshold").to_dict(orient="records"):
        lines.append(
            f"- threshold={row['failure_threshold']:.2f}m: "
            f"{row['winner_score']} (ROC-AUC={row['winner_roc_auc']:.3f}, AP={row['winner_ap']:.3f})"
        )

    winner_scores = overall_winners["winner_score"].astype(str)
    model_wins = int(np.sum(winner_scores.str.startswith("model_")))
    fusion_wins = int(np.sum(winner_scores.str.startswith("fusion_")))
    lines.append("")
    lines.append(f"Model overall wins: {model_wins}/{len(overall_winners)} thresholds")
    lines.append(f"Fusion overall wins: {fusion_wins}/{len(overall_winners)} thresholds")

    if not scenario_winners.empty:
        scenario_scores = scenario_winners["winner_score"].astype(str)
        scenario_model_wins = int(np.sum(scenario_scores.str.startswith("model_")))
        scenario_fusion_wins = int(np.sum(scenario_scores.str.startswith("fusion_")))
        lines.append(
            f"Model scenario wins: {scenario_model_wins}/{len(scenario_winners)} threshold-scenario cells"
        )
        lines.append(
            f"Fusion scenario wins: {scenario_fusion_wins}/{len(scenario_winners)} threshold-scenario cells"
        )

    (output_dir / "head_to_head_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"overall_csv: {output_dir / 'overall_head_to_head.csv'}")
    print(f"scenario_csv: {output_dir / 'scenario_head_to_head.csv'}")
    print(f"overall_winners_csv: {output_dir / 'overall_winners.csv'}")
    print(f"scenario_winners_csv: {output_dir / 'scenario_winners.csv'}")
    print(f"summary_txt: {output_dir / 'head_to_head_summary.txt'}")
    if not args.skip_plots:
        print(f"overall_auc_png: {output_dir / 'head_to_head_auc.png'}")
        print(f"scenario_auc_png: {output_dir / f'head_to_head_auc_by_scenario_t{scenario_slug}.png'}")


if __name__ == "__main__":
    main()
