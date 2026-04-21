#!/usr/bin/env python3
"""
Analyze unified self-aware outputs and save quick-look plots.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat([a, b], axis=1).dropna()
    if len(frame) < 2:
        return float("nan")
    return float(frame.iloc[:, 0].corr(frame.iloc[:, 1]))


def _display_score_columns(predictions: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    if 'primary_risk_score' in predictions.columns:
        risk = predictions['primary_risk_score']
        confidence = predictions.get('primary_confidence_score', 1.0 - risk)
        source = (
            str(predictions['primary_risk_source'].dropna().iloc[0])
            if 'primary_risk_source' in predictions.columns and predictions['primary_risk_source'].notna().any()
            else 'primary_risk_score'
        )
        return risk, confidence, source
    return predictions['failure_probability'], predictions['confidence_score'], 'learned_failure_probability'


def analyze_results(predictions_path: str, summary_path: str, output_dir: str) -> dict:
    predictions = pd.read_csv(predictions_path)
    os.makedirs(output_dir, exist_ok=True)
    risk_scores, confidence_scores, risk_source = _display_score_columns(predictions)

    has_actual = 'actual_pose_error' in predictions.columns
    correlation = _safe_corr(
        risk_scores,
        predictions['actual_pose_error'],
    ) if has_actual else float("nan")

    metrics = {
        'num_predictions': int(len(predictions)),
        'risk_score_source': risk_source,
        'failure_probability_mean': float(risk_scores.mean()),
        'confidence_mean': float(confidence_scores.mean()),
        'predicted_pose_error_mean': float(predictions['predicted_pose_error'].mean()),
        'predicted_failure_rate': float((risk_scores >= 0.5).mean()),
        'failure_vs_actual_corr': correlation,
    }
    if 'failure_probability' in predictions.columns:
        metrics['learned_failure_probability_mean'] = float(predictions['failure_probability'].mean())

    if has_actual:
        metrics['actual_pose_error_mean'] = float(predictions['actual_pose_error'].mean())

    analysis_txt = os.path.join(output_dir, 'analysis_summary.txt')
    with open(analysis_txt, 'w', encoding='utf-8') as handle:
        handle.write('Unified result analysis\n')
        if os.path.exists(summary_path):
            handle.write(f'source_summary: {summary_path}\n')
        for key, value in metrics.items():
            handle.write(f'{key}: {value}\n')

    # Plot 1: probability and confidence over time.
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(predictions['timestamp'], risk_scores, label='failure_probability', color='tab:red')
    ax.plot(predictions['timestamp'], confidence_scores, label='confidence_score', color='tab:blue')
    ax.set_title('Self-Awareness Scores Over Time')
    ax.set_xlabel('timestamp (s)')
    ax.set_ylabel('score')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'scores_over_time.png'))
    plt.close(fig)

    # Plot 2: predicted vs actual pose error if available.
    if has_actual:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(predictions['timestamp'], predictions['predicted_pose_error'], label='predicted_pose_error', color='tab:orange')
        axes[0].plot(predictions['timestamp'], predictions['actual_pose_error'], label='actual_pose_error', color='tab:green', alpha=0.8)
        axes[0].set_ylabel('pose error (m)')
        axes[0].set_title('Predicted vs Actual Pose Error')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].scatter(
            predictions['actual_pose_error'],
            risk_scores,
            s=10,
            alpha=0.6,
            color='tab:red',
        )
        axes[1].set_xlabel('actual_pose_error (m)')
        axes[1].set_ylabel('failure_probability')
        axes[1].set_title('Failure Probability vs Actual Pose Error')
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'pose_error_comparison.png'))
        plt.close(fig)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze unified self-aware outputs')
    parser.add_argument('--predictions', required=True, help='Path to reliability_predictions.csv')
    parser.add_argument('--summary', required=False, default='', help='Optional summary.txt path')
    parser.add_argument('--output-dir', required=True, help='Directory for analysis artifacts')
    args = parser.parse_args()

    metrics = analyze_results(
        predictions_path=args.predictions,
        summary_path=args.summary,
        output_dir=args.output_dir,
    )
    for key, value in metrics.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
