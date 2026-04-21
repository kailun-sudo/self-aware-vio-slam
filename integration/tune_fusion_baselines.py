#!/usr/bin/env python3
"""
Tune fixed-weight heuristic+learned fusion baselines on a validation benchmark
and evaluate them on a separate test benchmark.

This keeps the fusion fully non-parametric:

1. rank-normalize each component score
2. grid-search a convex fusion weight on validation only
3. evaluate the selected weight on held-out test data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from integration.analyze_model_vs_heuristics import (  # noqa: E402
    _build_frame_dataset,
    _evaluate_score,
    _parse_thresholds,
    _rank_normalize,
    _score_definitions,
)


FUSION_CANDIDATES: List[Tuple[str, str, str]] = [
    ("fusion_failure_epipolar", "model_failure_probability", "heuristic_epipolar_error_risk"),
    ("fusion_failure_num_inliers", "model_failure_probability", "heuristic_num_inliers_risk"),
    ("fusion_pred_error_epipolar", "model_predicted_pose_error", "heuristic_epipolar_error_risk"),
    ("fusion_pred_error_num_inliers", "model_predicted_pose_error", "heuristic_num_inliers_risk"),
]


def _parse_weights(value: str) -> List[float]:
    if ":" in value:
        start_raw, stop_raw, step_raw = value.split(":")
        start = float(start_raw)
        stop = float(stop_raw)
        step = float(step_raw)
        weights = []
        current = start
        while current <= stop + 1e-9:
            weights.append(round(current, 6))
            current += step
        return weights
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _fusion_score(model_score: pd.Series, heuristic_score: pd.Series, model_weight: float) -> pd.Series:
    model_rank = _rank_normalize(model_score)
    heuristic_rank = _rank_normalize(heuristic_score)
    return model_weight * model_rank + (1.0 - model_weight) * heuristic_rank


def _evaluate_candidates(
    frame_data: pd.DataFrame,
    thresholds: Iterable[float],
    weights: Iterable[float],
) -> pd.DataFrame:
    actual_error = frame_data["actual_pose_error"].to_numpy(dtype=float)
    score_map = _score_definitions(frame_data)
    rows: List[Dict[str, object]] = []

    for threshold in thresholds:
        y_true = (actual_error >= threshold).astype(int)
        for fusion_name, model_name, heuristic_name in FUSION_CANDIDATES:
            if model_name not in score_map or heuristic_name not in score_map:
                continue
            for model_weight in weights:
                fused = _fusion_score(score_map[model_name], score_map[heuristic_name], model_weight)
                metrics = _evaluate_score(y_true, fused.to_numpy(dtype=float), fusion_name)
                rows.append(
                    {
                        "failure_threshold": float(threshold),
                        "fusion_name": fusion_name,
                        "model_score": model_name,
                        "heuristic_score": heuristic_name,
                        "model_weight": float(model_weight),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _select_best(validation_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for threshold, group in validation_table.groupby("failure_threshold", dropna=False):
        ranked = group.sort_values(
            ["roc_auc", "average_precision", "best_f1", "model_weight"],
            ascending=[False, False, False, False],
            na_position="last",
        )
        winner = ranked.iloc[0]
        rows.append(winner.to_dict())
    return pd.DataFrame(rows)


def _evaluate_selected_on_test(
    frame_data: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    actual_error = frame_data["actual_pose_error"].to_numpy(dtype=float)
    score_map = _score_definitions(frame_data)
    rows: List[Dict[str, object]] = []
    for row in selected.to_dict(orient="records"):
        threshold = float(row["failure_threshold"])
        model_name = str(row["model_score"])
        heuristic_name = str(row["heuristic_score"])
        model_weight = float(row["model_weight"])
        if model_name not in score_map or heuristic_name not in score_map:
            continue
        y_true = (actual_error >= threshold).astype(int)
        fused = _fusion_score(score_map[model_name], score_map[heuristic_name], model_weight)
        metrics = _evaluate_score(y_true, fused.to_numpy(dtype=float), str(row["fusion_name"]))
        rows.append(
            {
                "failure_threshold": threshold,
                "fusion_name": row["fusion_name"],
                "model_score": model_name,
                "heuristic_score": heuristic_name,
                "model_weight": model_weight,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune rank-fusion baselines on validation and evaluate on test.")
    parser.add_argument("--val-sweep-results", required=True, help="Validation sweep_results.csv path.")
    parser.add_argument("--test-sweep-results", required=True, help="Test sweep_results.csv path.")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    parser.add_argument(
        "--failure-thresholds",
        default="6.0,8.0,10.0",
        help="Comma-separated actual-error thresholds in meters.",
    )
    parser.add_argument(
        "--weight-grid",
        default="0.0:1.0:0.05",
        help="Either start:stop:step or a comma-separated list of model weights.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = _parse_thresholds(args.failure_thresholds)
    weights = _parse_weights(args.weight_grid)

    val_frame = _build_frame_dataset(Path(args.val_sweep_results), include_baseline=True)
    test_frame = _build_frame_dataset(Path(args.test_sweep_results), include_baseline=True)

    val_candidates = _evaluate_candidates(val_frame, thresholds, weights)
    val_candidates.to_csv(output_dir / "validation_fusion_candidates.csv", index=False)

    selected = _select_best(val_candidates)
    selected.to_csv(output_dir / "selected_fusion_by_threshold.csv", index=False)

    test_results = _evaluate_selected_on_test(test_frame, selected)
    test_results.to_csv(output_dir / "test_fusion_results.csv", index=False)

    lines = [
        "Validation-tuned fusion benchmark",
        f"val_sweep_results: {args.val_sweep_results}",
        f"test_sweep_results: {args.test_sweep_results}",
        "",
        "Selected fusion per threshold (chosen on validation):",
    ]
    for row in selected.sort_values("failure_threshold").to_dict(orient="records"):
        lines.append(
            f"- threshold={row['failure_threshold']:.2f}m: {row['fusion_name']} "
            f"(model={row['model_score']}, heuristic={row['heuristic_score']}, "
            f"model_weight={row['model_weight']:.2f}, val_roc_auc={row['roc_auc']:.3f}, "
            f"val_ap={row['average_precision']:.3f})"
        )

    lines.append("")
    lines.append("Held-out test performance of selected fusion:")
    for row in test_results.sort_values("failure_threshold").to_dict(orient="records"):
        lines.append(
            f"- threshold={row['failure_threshold']:.2f}m: {row['fusion_name']} "
            f"(model_weight={row['model_weight']:.2f}, test_roc_auc={row['roc_auc']:.3f}, "
            f"test_ap={row['average_precision']:.3f}, test_best_f1={row['best_f1']:.3f})"
        )

    (output_dir / "fusion_tuning_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"validation_candidates_csv: {output_dir / 'validation_fusion_candidates.csv'}")
    print(f"selected_csv: {output_dir / 'selected_fusion_by_threshold.csv'}")
    print(f"test_results_csv: {output_dir / 'test_fusion_results.csv'}")
    print(f"summary_txt: {output_dir / 'fusion_tuning_summary.txt'}")


if __name__ == "__main__":
    main()
