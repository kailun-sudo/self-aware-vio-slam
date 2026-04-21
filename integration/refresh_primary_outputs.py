#!/usr/bin/env python3
"""
Refresh self-aware outputs and reports for an existing multi-sequence replay directory.

This script intentionally reuses existing `baseline_vio/` and `degraded_vio/`
artifacts. It does not rerun the VIO pipeline. Instead it:

1. regenerates `baseline_self_aware/`
2. regenerates `degraded_self_aware/`
3. regenerates baseline visual GUIs and baseline-vs-degraded comparison GUIs
4. rewrites `sweep_results.csv` from the refreshed summaries
5. regenerates the aggregate HTML report
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
INTEGRATION_ROOT = ROOT_DIR / "integration"

# Reuse the same summary parsing and naming logic as the main sweep script.
from run_multisequence_degradation_sweep import _normalize_name, _read_summary, _run  # noqa: E402


def _summary_metric(summary: Dict[str, object], key: str):
    aliases = {
        "failure_probability_mean": [
            "failure_probability_mean",
            "primary_failure_probability_mean",
            "learned_failure_probability_mean",
        ],
        "confidence_mean": [
            "confidence_mean",
            "primary_confidence_mean",
            "learned_confidence_mean",
        ],
    }
    for candidate in aliases.get(key, [key]):
        if candidate in summary and summary[candidate] is not None:
            return summary[candidate]
    return None


def _refresh_offline_demo(
    *,
    self_aware_python: str,
    config_path: str,
    checkpoint: str | None,
    dataset_stats: str | None,
    metrics_path: Path,
    estimated_path: Path,
    groundtruth_path: Path,
    output_dir: Path,
    dry_run: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        self_aware_python,
        str(INTEGRATION_ROOT / "run_offline_unified_demo.py"),
        "--metrics",
        str(metrics_path),
        "--estimated",
        str(estimated_path),
        "--groundtruth",
        str(groundtruth_path),
        "--output-dir",
        str(output_dir),
        "--config",
        config_path,
    ]
    if checkpoint:
        command.extend(["--checkpoint", checkpoint])
    if dataset_stats:
        command.extend(["--dataset-stats", dataset_stats])
    _run(command, dry_run)


def _refresh_visual_demo(
    *,
    self_aware_python: str,
    self_aware_dir: Path,
    output_dir: Path,
    dry_run: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        self_aware_python,
        str(INTEGRATION_ROOT / "create_visual_demo.py"),
        "--metrics",
        str(self_aware_dir / "slam_metrics.csv"),
        "--predictions",
        str(self_aware_dir / "reliability_predictions.csv"),
        "--estimated",
        str(self_aware_dir / "estimated.txt"),
        "--output-dir",
        str(output_dir),
    ]
    _run(command, dry_run)


def _refresh_comparison_gui(
    *,
    self_aware_python: str,
    baseline_dir: Path,
    degraded_dir: Path,
    output_dir: Path,
    dry_run: bool,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        self_aware_python,
        str(INTEGRATION_ROOT / "create_degradation_comparison_demo.py"),
        "--baseline-dir",
        str(baseline_dir),
        "--degraded-dir",
        str(degraded_dir),
        "--output-dir",
        str(output_dir),
    ]
    _run(command, dry_run)


def _refresh_report(
    *,
    self_aware_python: str,
    results_csv: Path,
    report_dir: Path,
    dry_run: bool,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    command = [
        self_aware_python,
        str(INTEGRATION_ROOT / "create_multisequence_degradation_report.py"),
        "--results-csv",
        str(results_csv),
        "--output-dir",
        str(report_dir),
    ]
    _run(command, dry_run)


def _update_row_metrics(
    row: Dict[str, object],
    *,
    baseline_vio_dir: Path,
    degraded_vio_dir: Path,
    baseline_self_aware_dir: Path,
    degraded_self_aware_dir: Path,
    comparison_dir: Path,
) -> Dict[str, object]:
    baseline_vio_summary = _read_summary(baseline_vio_dir / "summary.txt")
    degraded_vio_summary = _read_summary(degraded_vio_dir / "summary.txt")
    baseline_self_summary = _read_summary(baseline_self_aware_dir / "summary.txt")
    degraded_self_summary = _read_summary(degraded_self_aware_dir / "summary.txt")

    updated = dict(row)
    updated["comparison_gui_path"] = str((comparison_dir / "visual_demo.html").resolve())
    updated["comparison_summary_path"] = str((comparison_dir / "comparison_gui_summary.txt").resolve())
    updated["baseline_primary_risk_source"] = baseline_self_summary.get("primary_risk_source")
    updated["degraded_primary_risk_source"] = degraded_self_summary.get("primary_risk_source")

    metric_keys = [
        "mean_inlier_ratio",
        "tracking_success_ratio",
        "trajectory_length_m",
        "pose_error_mean",
        "failure_probability_mean",
        "confidence_mean",
    ]
    for key in metric_keys:
        baseline_value = baseline_vio_summary.get(key)
        degraded_value = degraded_vio_summary.get(key)
        if baseline_value is None:
            baseline_value = _summary_metric(baseline_self_summary, key)
        if degraded_value is None:
            degraded_value = _summary_metric(degraded_self_summary, key)
        updated.pop(f"{key}_delta", None)
        if baseline_value is not None:
            updated[f"baseline_{key}"] = baseline_value
        if degraded_value is not None:
            updated[f"degraded_{key}"] = degraded_value
        if baseline_value is not None and degraded_value is not None:
            try:
                updated[f"{key}_delta"] = float(degraded_value) - float(baseline_value)
            except (TypeError, ValueError):
                pass
    return updated


def main():
    parser = argparse.ArgumentParser(description="Refresh heuristic-primary outputs for an existing replay directory")
    parser.add_argument("--output-root", required=True, help="Existing replay output root with sweep_results.csv")
    parser.add_argument("--dataset-root", required=True, help="EuRoC sequence root, e.g. VIO-SLAM/data/sequences")
    parser.add_argument("--self-aware-python", default=str(ROOT_DIR / "self_aware_slam" / "venv" / "bin" / "python"))
    parser.add_argument("--self-aware-config", default=str(ROOT_DIR / "self_aware_slam" / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-stats", default=None)
    parser.add_argument("--sequences", default="", help="Optional comma-separated sequence filter")
    parser.add_argument("--skip-baseline-visual", action="store_true")
    parser.add_argument(
        "--rebuild-csv-report-only",
        action="store_true",
        help="Reuse existing self-aware outputs and only rebuild sweep_results.csv and report",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    results_csv = output_root / "sweep_results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"sweep_results.csv not found: {results_csv}")

    results = pd.read_csv(results_csv)
    sequence_filter = {item.strip() for item in args.sequences.split(",") if item.strip()}
    if sequence_filter:
        results = results[results["sequence"].isin(sequence_filter)].reset_index(drop=True)
        if results.empty:
            raise ValueError("No rows left after applying --sequences filter")

    updated_rows: List[Dict[str, object]] = []
    baseline_cache: set[str] = set()

    for row in results.to_dict(orient="records"):
        sequence = str(row["sequence"])
        sequence_dir = output_root / sequence
        data_path = dataset_root / sequence / "mav0"
        groundtruth_path = data_path / "state_groundtruth_estimate0" / "data.csv"
        if not groundtruth_path.exists():
            raise FileNotFoundError(f"Ground truth not found: {groundtruth_path}")

        baseline_vio_dir = sequence_dir / "baseline_vio"
        baseline_self_aware_dir = sequence_dir / "baseline_self_aware"
        baseline_visual_dir = sequence_dir / "baseline_visual_gui"
        if sequence not in baseline_cache and not args.rebuild_csv_report_only:
            _refresh_offline_demo(
                self_aware_python=args.self_aware_python,
                config_path=args.self_aware_config,
                checkpoint=args.checkpoint,
                dataset_stats=args.dataset_stats,
                metrics_path=baseline_vio_dir / "slam_metrics.csv",
                estimated_path=baseline_vio_dir / "estimated_tum.txt",
                groundtruth_path=groundtruth_path,
                output_dir=baseline_self_aware_dir,
                dry_run=args.dry_run,
            )
            if not args.skip_baseline_visual:
                _refresh_visual_demo(
                    self_aware_python=args.self_aware_python,
                    self_aware_dir=baseline_self_aware_dir,
                    output_dir=baseline_visual_dir,
                    dry_run=args.dry_run,
                )
            baseline_cache.add(sequence)

        comparison_path = Path(str(row["comparison_gui_path"])).resolve()
        scenario_root = comparison_path.parents[2]
        degraded_vio_dir = scenario_root / "degraded_vio"
        degraded_self_aware_dir = scenario_root / "degraded_self_aware"
        comparison_dir = scenario_root / "comparison" / "gui"

        if not args.rebuild_csv_report_only:
            _refresh_offline_demo(
                self_aware_python=args.self_aware_python,
                config_path=args.self_aware_config,
                checkpoint=args.checkpoint,
                dataset_stats=args.dataset_stats,
                metrics_path=degraded_vio_dir / "slam_metrics.csv",
                estimated_path=degraded_vio_dir / "estimated_tum.txt",
                groundtruth_path=groundtruth_path,
                output_dir=degraded_self_aware_dir,
                dry_run=args.dry_run,
            )
            _refresh_comparison_gui(
                self_aware_python=args.self_aware_python,
                baseline_dir=baseline_self_aware_dir,
                degraded_dir=degraded_self_aware_dir,
                output_dir=comparison_dir,
                dry_run=args.dry_run,
            )

        updated = _update_row_metrics(
            row,
            baseline_vio_dir=baseline_vio_dir,
            degraded_vio_dir=degraded_vio_dir,
            baseline_self_aware_dir=baseline_self_aware_dir,
            degraded_self_aware_dir=degraded_self_aware_dir,
            comparison_dir=comparison_dir,
        )
        updated["sequence_short"] = _normalize_name(sequence)
        updated_rows.append(updated)

    if args.dry_run:
        print(f"dry_run_rows: {len(updated_rows)}")
        return

    refreshed = pd.DataFrame(updated_rows)
    # If a sequence filter was used, merge refreshed rows back into the original full CSV.
    if sequence_filter:
        original = pd.read_csv(results_csv)
        keep = original[~original["sequence"].isin(sequence_filter)]
        refreshed = pd.concat([keep, refreshed], ignore_index=True)
        sort_columns = [column for column in ["sequence", "scenario"] if column in refreshed.columns]
        if sort_columns:
            refreshed = refreshed.sort_values(sort_columns).reset_index(drop=True)

    refreshed.to_csv(results_csv, index=False)
    print(f"refreshed_results_csv: {results_csv}")

    report_dir = output_root / "report"
    _refresh_report(
        self_aware_python=args.self_aware_python,
        results_csv=results_csv,
        report_dir=report_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
