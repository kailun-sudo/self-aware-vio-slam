#!/usr/bin/env python3
"""
Create an interactive baseline-vs-degraded GUI demo for EuRoC playback comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parent.parent / ".mplcache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from create_visual_demo import build_demo_dataframe


def _load_demo(sequence_dir: str) -> pd.DataFrame:
    return build_demo_dataframe(
        metrics_path=os.path.join(sequence_dir, "slam_metrics.csv"),
        predictions_path=os.path.join(sequence_dir, "reliability_predictions.csv"),
        estimated_path=os.path.join(sequence_dir, "estimated.txt"),
    ).sort_values("timestamp").reset_index(drop=True)


def _align_pair(baseline: pd.DataFrame, degraded: pd.DataFrame) -> pd.DataFrame:
    left = baseline.add_prefix("baseline_").rename(columns={"baseline_timestamp": "timestamp"})
    right = degraded.add_prefix("degraded_").rename(columns={"degraded_timestamp": "timestamp_right"})
    aligned = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp_right"),
        left_on="timestamp",
        right_on="timestamp_right",
        direction="nearest",
        tolerance=0.02,
    )
    return aligned


def _write_summary(aligned: pd.DataFrame, output_dir: str):
    output_path = os.path.join(output_dir, "comparison_gui_summary.txt")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("Degradation comparison GUI summary\n")
        handle.write(f"rows: {len(aligned)}\n")
        if "baseline_risk_score_source" in aligned.columns:
            handle.write(f"baseline_risk_score_source: {aligned['baseline_risk_score_source'].iloc[0]}\n")
        if "degraded_risk_score_source" in aligned.columns:
            handle.write(f"degraded_risk_score_source: {aligned['degraded_risk_score_source'].iloc[0]}\n")
        handle.write(
            f"baseline_failure_probability_mean: {aligned['baseline_failure_probability'].mean():.6f}\n"
        )
        handle.write(
            f"degraded_failure_probability_mean: {aligned['degraded_failure_probability'].mean():.6f}\n"
        )
        handle.write(
            f"baseline_confidence_mean: {aligned['baseline_confidence_score'].mean():.6f}\n"
        )
        handle.write(
            f"degraded_confidence_mean: {aligned['degraded_confidence_score'].mean():.6f}\n"
        )
        if "baseline_actual_pose_error" in aligned.columns:
            handle.write(
                f"baseline_actual_pose_error_mean: {aligned['baseline_actual_pose_error'].mean():.6f}\n"
            )
        if "degraded_actual_pose_error" in aligned.columns:
            handle.write(
                f"degraded_actual_pose_error_mean: {aligned['degraded_actual_pose_error'].mean():.6f}\n"
            )


def _plot_static_overview(aligned: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(
        aligned["baseline_px"],
        aligned["baseline_py"],
        color="tab:blue",
        label="baseline",
    )
    axes[0, 0].plot(
        aligned["degraded_px"],
        aligned["degraded_py"],
        color="tab:red",
        label="degraded",
        alpha=0.8,
    )
    axes[0, 0].set_title("Trajectory Comparison")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(
        aligned["timestamp"],
        aligned["baseline_failure_probability"],
        color="tab:blue",
        label="baseline failure",
    )
    axes[0, 1].plot(
        aligned["timestamp"],
        aligned["degraded_failure_probability"],
        color="tab:red",
        label="degraded failure",
    )
    axes[0, 1].set_title("Failure Probability Over Time")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        aligned["timestamp"],
        aligned["baseline_confidence_score"],
        color="tab:blue",
        label="baseline confidence",
    )
    axes[1, 0].plot(
        aligned["timestamp"],
        aligned["degraded_confidence_score"],
        color="tab:red",
        label="degraded confidence",
    )
    axes[1, 0].set_title("Confidence Over Time")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    if "baseline_actual_pose_error" in aligned.columns and "degraded_actual_pose_error" in aligned.columns:
        axes[1, 1].plot(
            aligned["timestamp"],
            aligned["baseline_actual_pose_error"],
            color="tab:blue",
            label="baseline actual error",
        )
        axes[1, 1].plot(
            aligned["timestamp"],
            aligned["degraded_actual_pose_error"],
            color="tab:red",
            label="degraded actual error",
        )
        axes[1, 1].set_title("Actual Pose Error Over Time")
        axes[1, 1].legend()
    else:
        axes[1, 1].plot(
            aligned["timestamp"],
            aligned["baseline_predicted_pose_error"],
            color="tab:blue",
            label="baseline predicted error",
        )
        axes[1, 1].plot(
            aligned["timestamp"],
            aligned["degraded_predicted_pose_error"],
            color="tab:red",
            label="degraded predicted error",
        )
        axes[1, 1].set_title("Predicted Pose Error Over Time")
        axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_gui_overview.png"))
    plt.close(fig)


def _write_dashboard_json(aligned: pd.DataFrame, output_dir: str):
    payload = aligned.replace({np.nan: None}).to_dict(orient="records")
    output_path = os.path.join(output_dir, "comparison_dashboard_data.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "rows": payload,
                "summary": {
                    "num_rows": int(len(aligned)),
                    "baseline_failure_probability_mean": float(aligned["baseline_failure_probability"].mean()),
                    "degraded_failure_probability_mean": float(aligned["degraded_failure_probability"].mean()),
                    "baseline_confidence_mean": float(aligned["baseline_confidence_score"].mean()),
                    "degraded_confidence_mean": float(aligned["degraded_confidence_score"].mean()),
                },
            },
            handle,
            ensure_ascii=False,
        )


def _top_delta_rows(aligned: pd.DataFrame) -> list[dict]:
    scored = aligned.copy()
    scored["failure_delta"] = scored["degraded_failure_probability"] - scored["baseline_failure_probability"]
    columns = [
        "baseline_frame_id",
        "baseline_failure_probability",
        "degraded_failure_probability",
        "failure_delta",
        "baseline_inlier_ratio",
        "degraded_inlier_ratio",
        "baseline_actual_pose_error",
        "degraded_actual_pose_error",
    ]
    available = [column for column in columns if column in scored.columns]
    return scored.sort_values("failure_delta", ascending=False).head(12)[available].replace({np.nan: None}).to_dict(orient="records")


def _write_html(aligned: pd.DataFrame, output_dir: str):
    rows = aligned.replace({np.nan: None}).to_dict(orient="records")
    top_delta_rows = _top_delta_rows(aligned)
    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Degradation Comparison GUI</title>
  <style>
    :root {{
      --bg: #f5f1ea;
      --panel: #fffdfa;
      --ink: #17212b;
      --muted: #6a7280;
      --blue: #2a6f97;
      --red: #c44536;
      --line: #e6dccd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(42,111,151,0.12), transparent 24%),
        radial-gradient(circle at bottom right, rgba(196,69,54,0.12), transparent 28%),
        var(--bg);
    }}
    .page {{ max-width: 1520px; margin: 0 auto; padding: 28px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 14px 30px rgba(60,55,48,0.06);
    }}
    .hero {{ padding: 24px; margin-bottom: 20px; }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    .card {{
      padding: 14px 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.85);
    }}
    .card .value {{
      display: block;
      margin-top: 8px;
      font-size: 1.6rem;
      font-weight: 700;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
      margin-bottom: 20px;
    }}
    .controls button {{
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 10px 18px;
      cursor: pointer;
      color: white;
      font-weight: 600;
      background: linear-gradient(135deg, var(--red), #d98324);
    }}
    .controls button.secondary {{ background: #3f4b59; }}
    .controls input[type="range"] {{ width: 100%; }}
    .layout {{
      display: grid;
      grid-template-columns: 1.3fr 0.95fr;
      gap: 18px;
    }}
    .stack {{ display: grid; gap: 18px; }}
    .panel {{ padding: 18px; }}
    .dual {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(249,245,238,0.98));
      border: 1px solid var(--line);
    }}
    .frame-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 16px;
      margin-top: 14px;
    }}
    .item {{
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(23,33,43,0.03);
    }}
    .item strong {{
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .item span {{ font-size: 1.05rem; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
    }}
    .jump-btn {{
      border: none;
      background: rgba(42,111,151,0.12);
      color: var(--blue);
      padding: 6px 10px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 600;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .legend span::before {{
      content: "";
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 6px;
      vertical-align: middle;
    }}
    .legend .baseline::before {{ background: var(--blue); }}
    .legend .degraded::before {{ background: var(--red); }}
    .legend .delta::before {{ background: #6a994e; }}
    @media (max-width: 1120px) {{
      .layout, .dual {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>EuRoC Degradation Comparison GUI</h1>
      <p>这个页面把 baseline 和 degraded 两条 Self-Aware VIO-SLAM 结果放到同一屏里。你可以拖动时间轴、自动播放，并直接观察退化注入之后轨迹、failure probability、confidence 和 tracking 指标是如何变化的。</p>
      <div class="metrics">
        <div class="card"><strong>对齐帧数</strong><span class="value">{len(aligned)}</span></div>
        <div class="card"><strong>Baseline 平均失效概率</strong><span class="value">{aligned['baseline_failure_probability'].mean():.3f}</span></div>
        <div class="card"><strong>Degraded 平均失效概率</strong><span class="value">{aligned['degraded_failure_probability'].mean():.3f}</span></div>
        <div class="card"><strong>Baseline 平均置信度</strong><span class="value">{aligned['baseline_confidence_score'].mean():.3f}</span></div>
        <div class="card"><strong>Degraded 平均置信度</strong><span class="value">{aligned['degraded_confidence_score'].mean():.3f}</span></div>
        <div class="card"><strong>Baseline 风险来源</strong><span class="value">{aligned['baseline_risk_score_source'].iloc[0] if 'baseline_risk_score_source' in aligned.columns else 'learned_failure_probability'}</span></div>
        <div class="card"><strong>Degraded 风险来源</strong><span class="value">{aligned['degraded_risk_score_source'].iloc[0] if 'degraded_risk_score_source' in aligned.columns else 'learned_failure_probability'}</span></div>
      </div>
    </section>

    <section class="hero controls">
      <div>
        <button id="playButton">播放</button>
        <button id="resetButton" class="secondary">回到开头</button>
      </div>
      <input id="frameSlider" type="range" min="0" max="{max(len(aligned) - 1, 0)}" value="0">
      <div id="statusText">Frame 0 / {max(len(aligned) - 1, 0)}</div>
    </section>

    <section class="layout">
      <div class="stack">
        <div class="panel">
          <h2>1. Baseline vs Degraded Trajectory</h2>
          <p>左边是 baseline 轨迹，右边是 degraded 轨迹。当前时间点用白色圆环标出，点颜色表示该点的 failure probability。</p>
          <div class="dual">
            <div>
              <h3>Baseline</h3>
              <canvas id="baselineTrajectoryCanvas" width="700" height="420"></canvas>
            </div>
            <div>
              <h3>Degraded</h3>
              <canvas id="degradedTrajectoryCanvas" width="700" height="420"></canvas>
            </div>
          </div>
        </div>

        <div class="panel">
          <h2>2. Failure / Confidence 对比</h2>
          <p>这里把 baseline 和 degraded 的 `failure_probability`、`confidence_score` 放在同一时间线上，适合直接看退化是否让风险更高、置信度更低。</p>
          <canvas id="scoreCanvas" width="980" height="320"></canvas>
          <div class="legend">
            <span class="baseline">baseline</span>
            <span class="degraded">degraded</span>
          </div>
        </div>

        <div class="panel">
          <h2>3. Tracking / Error 对比</h2>
          <p>这部分帮助你解释“为什么会变差”。如果退化注入真的影响系统，你通常会看到 `inlier_ratio`、`actual_pose_error` 或 `pose_optimization_residual` 发生偏移。</p>
          <canvas id="metricsCanvas" width="980" height="320"></canvas>
          <div class="legend">
            <span class="baseline">baseline</span>
            <span class="degraded">degraded</span>
            <span class="delta">error / residual</span>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="panel">
          <h2>4. 当前帧状态对比</h2>
          <p id="frameSummary">拖动滑条或点击播放，查看同一时刻 baseline 与 degraded 的联合状态。</p>
          <div class="frame-grid" id="frameGrid"></div>
        </div>

        <div class="panel">
          <h2>5. Top Failure Delta Frames</h2>
          <p>这些是退化后相对 baseline 风险上升最明显的帧，点“跳转”可以直接定位。</p>
          <table>
            <thead>
              <tr><th>frame_id</th><th>baseline</th><th>degraded</th><th>delta</th><th>跳转</th></tr>
            </thead>
            <tbody id="topDeltaBody"></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
  <script>
    const rows = {json.dumps(rows, ensure_ascii=False)};
    const topDeltaRows = {json.dumps(top_delta_rows, ensure_ascii=False)};
    const slider = document.getElementById('frameSlider');
    const playButton = document.getElementById('playButton');
    const resetButton = document.getElementById('resetButton');
    const statusText = document.getElementById('statusText');
    const frameGrid = document.getElementById('frameGrid');
    const frameSummary = document.getElementById('frameSummary');
    const topDeltaBody = document.getElementById('topDeltaBody');

    const baselineTrajectoryCanvas = document.getElementById('baselineTrajectoryCanvas');
    const degradedTrajectoryCanvas = document.getElementById('degradedTrajectoryCanvas');
    const scoreCanvas = document.getElementById('scoreCanvas');
    const metricsCanvas = document.getElementById('metricsCanvas');

    const baselineTrajectoryCtx = baselineTrajectoryCanvas.getContext('2d');
    const degradedTrajectoryCtx = degradedTrajectoryCanvas.getContext('2d');
    const scoreCtx = scoreCanvas.getContext('2d');
    const metricsCtx = metricsCanvas.getContext('2d');

    let currentIndex = 0;
    let timer = null;

    function clamp01(v) {{
      return Math.max(0, Math.min(1, v));
    }}

    function riskColor(prob) {{
      const t = clamp01(prob || 0);
      const hue = 215 - t * 200;
      return `hsl(${{hue}}, 70%, 48%)`;
    }}

    function formatValue(value, digits = 3) {{
      if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
      return Number(value).toFixed(digits);
    }}

    function computeTrajectoryPoints(keyX, keyY, canvas) {{
      const padding = 32;
      const xs = rows.map(row => row[keyX] ?? 0);
      const ys = rows.map(row => row[keyY] ?? 0);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const spanX = Math.max(maxX - minX, 1e-6);
      const spanY = Math.max(maxY - minY, 1e-6);
      const width = canvas.width - padding * 2;
      const height = canvas.height - padding * 2;
      return rows.map(row => {{
        const x = padding + (((row[keyX] ?? 0) - minX) / spanX) * width;
        const y = canvas.height - padding - (((row[keyY] ?? 0) - minY) / spanY) * height;
        return {{ x, y }};
      }});
    }}

    const baselinePoints = computeTrajectoryPoints('baseline_px', 'baseline_py', baselineTrajectoryCanvas);
    const degradedPoints = computeTrajectoryPoints('degraded_px', 'degraded_py', degradedTrajectoryCanvas);

    function drawGrid(ctx, canvas, yTicks = 4) {{
      ctx.save();
      ctx.strokeStyle = 'rgba(60, 55, 48, 0.08)';
      for (let i = 0; i <= yTicks; i += 1) {{
        const y = 24 + (canvas.height - 48) * (i / yTicks);
        ctx.beginPath();
        ctx.moveTo(24, y);
        ctx.lineTo(canvas.width - 18, y);
        ctx.stroke();
      }}
      ctx.restore();
    }}

    function drawTrajectory(ctx, canvas, points, prefix, title) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawGrid(ctx, canvas, 5);
      ctx.beginPath();
      points.forEach((point, idx) => {{
        if (idx === 0) ctx.moveTo(point.x, point.y);
        else ctx.lineTo(point.x, point.y);
      }});
      ctx.strokeStyle = 'rgba(90,90,90,0.22)';
      ctx.lineWidth = 2;
      ctx.stroke();

      points.forEach((point, idx) => {{
        ctx.fillStyle = riskColor(rows[idx][`${{prefix}}_failure_probability`]);
        ctx.beginPath();
        ctx.arc(point.x, point.y, idx === currentIndex ? 5.8 : 3.2, 0, Math.PI * 2);
        ctx.fill();
      }});

      const current = points[currentIndex];
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(current.x, current.y, 9, 0, Math.PI * 2);
      ctx.stroke();

      ctx.fillStyle = '#17212b';
      ctx.font = '600 14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText(title, 24, 18);
    }}

    function normalizeSeries(values) {{
      const filtered = values.filter(v => Number.isFinite(v));
      const min = filtered.length ? Math.min(...filtered) : 0;
      const max = filtered.length ? Math.max(...filtered) : 1;
      const span = Math.max(max - min, 1e-6);
      return values.map(v => (Number.isFinite(v) ? (v - min) / span : 0));
    }}

    function drawSeriesCanvas(ctx, canvas, specs, title) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawGrid(ctx, canvas, 4);
      const left = 36;
      const right = canvas.width - 18;
      const top = 26;
      const bottom = canvas.height - 28;
      const width = right - left;
      const height = bottom - top;
      specs.forEach(spec => {{
        ctx.beginPath();
        spec.normalized.forEach((value, idx) => {{
          const x = left + (idx / Math.max(rows.length - 1, 1)) * width;
          const y = bottom - value * height;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.strokeStyle = spec.color;
        ctx.lineWidth = spec.width || 2;
        ctx.stroke();
      }});
      const cursorX = left + (currentIndex / Math.max(rows.length - 1, 1)) * width;
      ctx.strokeStyle = 'rgba(23,33,43,0.35)';
      ctx.beginPath();
      ctx.moveTo(cursorX, top);
      ctx.lineTo(cursorX, bottom);
      ctx.stroke();
      ctx.fillStyle = '#17212b';
      ctx.font = '600 14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText(title, 24, 18);
    }}

    function drawScoreCanvas() {{
      drawSeriesCanvas(scoreCtx, scoreCanvas, [
        {{ normalized: rows.map(r => r.baseline_failure_probability || 0), color: '#2a6f97', width: 2.4 }},
        {{ normalized: rows.map(r => r.degraded_failure_probability || 0), color: '#c44536', width: 2.4 }},
        {{ normalized: rows.map(r => r.baseline_confidence_score || 0), color: '#70a9d6', width: 2 }},
        {{ normalized: rows.map(r => r.degraded_confidence_score || 0), color: '#f4a261', width: 2 }},
      ], 'Failure / Confidence Comparison');
    }}

    function drawMetricsCanvas() {{
      const errorKey = rows[0]?.baseline_actual_pose_error !== undefined ? 'actual_pose_error' : 'predicted_pose_error';
      drawSeriesCanvas(metricsCtx, metricsCanvas, [
        {{ normalized: rows.map(r => r.baseline_inlier_ratio || 0), color: '#2a6f97', width: 2.4 }},
        {{ normalized: rows.map(r => r.degraded_inlier_ratio || 0), color: '#c44536', width: 2.4 }},
        {{ normalized: normalizeSeries(rows.map(r => r[`baseline_${{errorKey}}`] || 0)), color: '#6a994e', width: 2 }},
        {{ normalized: normalizeSeries(rows.map(r => r[`degraded_${{errorKey}}`] || 0)), color: '#90be6d', width: 2 }},
      ], 'Tracking / Error Comparison');
    }}

    function updateFramePanel() {{
      const row = rows[currentIndex];
      statusText.textContent = `Frame ${{currentIndex}} / ${{rows.length - 1}} | frame_id=${{row.baseline_frame_id}}`;
      frameSummary.textContent = `当前帧同时显示 baseline 与 degraded 的同一时间点状态，适合直接观察退化是否抬高风险、降低置信度或拉低 tracking 质量。`;
      const items = [
        ['baseline_failure_probability', formatValue(row.baseline_failure_probability)],
        ['degraded_failure_probability', formatValue(row.degraded_failure_probability)],
        ['failure_delta', formatValue((row.degraded_failure_probability || 0) - (row.baseline_failure_probability || 0))],
        ['baseline_confidence_score', formatValue(row.baseline_confidence_score)],
        ['degraded_confidence_score', formatValue(row.degraded_confidence_score)],
        ['baseline_inlier_ratio', formatValue(row.baseline_inlier_ratio)],
        ['degraded_inlier_ratio', formatValue(row.degraded_inlier_ratio)],
        ['baseline_actual_pose_error', formatValue(row.baseline_actual_pose_error)],
        ['degraded_actual_pose_error', formatValue(row.degraded_actual_pose_error)],
        ['baseline_pose_optimization_residual', formatValue(row.baseline_pose_optimization_residual)],
        ['degraded_pose_optimization_residual', formatValue(row.degraded_pose_optimization_residual)],
        ['baseline_num_matches', formatValue(row.baseline_num_matches, 0)],
        ['degraded_num_matches', formatValue(row.degraded_num_matches, 0)],
        ['baseline_num_inliers', formatValue(row.baseline_num_inliers, 0)],
        ['degraded_num_inliers', formatValue(row.degraded_num_inliers, 0)],
      ];
      frameGrid.innerHTML = items.map(([label, value]) => `
        <div class="item">
          <strong>${{label}}</strong>
          <span>${{value}}</span>
        </div>
      `).join('');
    }}

    function renderTopTable() {{
      topDeltaBody.innerHTML = topDeltaRows.map(row => {{
        const idx = rows.findIndex(candidate => candidate.baseline_frame_id === row.baseline_frame_id);
        return `
          <tr>
            <td>${{row.baseline_frame_id}}</td>
            <td>${{formatValue(row.baseline_failure_probability)}}</td>
            <td>${{formatValue(row.degraded_failure_probability)}}</td>
            <td>${{formatValue(row.failure_delta)}}</td>
            <td><button class="jump-btn" data-index="${{idx}}">跳转</button></td>
          </tr>
        `;
      }}).join('');
      topDeltaBody.querySelectorAll('.jump-btn').forEach(button => {{
        button.addEventListener('click', () => {{
          const idx = Number(button.getAttribute('data-index'));
          if (Number.isFinite(idx) && idx >= 0) setFrame(idx);
        }});
      }});
    }}

    function render() {{
      drawTrajectory(baselineTrajectoryCtx, baselineTrajectoryCanvas, baselinePoints, 'baseline', 'Baseline trajectory');
      drawTrajectory(degradedTrajectoryCtx, degradedTrajectoryCanvas, degradedPoints, 'degraded', 'Degraded trajectory');
      drawScoreCanvas();
      drawMetricsCanvas();
      updateFramePanel();
      slider.value = currentIndex;
    }}

    function setFrame(idx) {{
      currentIndex = Math.max(0, Math.min(rows.length - 1, idx));
      render();
    }}

    slider.addEventListener('input', event => setFrame(Number(event.target.value)));
    playButton.addEventListener('click', () => {{
      if (timer) {{
        window.clearInterval(timer);
        timer = null;
        playButton.textContent = '播放';
        return;
      }}
      playButton.textContent = '暂停';
      timer = window.setInterval(() => {{
        if (currentIndex >= rows.length - 1) {{
          window.clearInterval(timer);
          timer = null;
          playButton.textContent = '播放';
          return;
        }}
        setFrame(currentIndex + 1);
      }}, 180);
    }});
    resetButton.addEventListener('click', () => setFrame(0));

    renderTopTable();
    render();
  </script>
</body>
</html>
"""
    with open(os.path.join(output_dir, "visual_demo.html"), "w", encoding="utf-8") as handle:
        handle.write(html)


def main():
    parser = argparse.ArgumentParser(description="Create baseline-vs-degraded GUI demo")
    parser.add_argument("--baseline-dir", required=True, help="Directory containing baseline unified outputs")
    parser.add_argument("--degraded-dir", required=True, help="Directory containing degraded unified outputs")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison GUI artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    baseline = _load_demo(args.baseline_dir)
    degraded = _load_demo(args.degraded_dir)
    aligned = _align_pair(baseline, degraded)
    aligned.to_csv(os.path.join(args.output_dir, "comparison_aligned.csv"), index=False)
    _write_summary(aligned, args.output_dir)
    _plot_static_overview(aligned, args.output_dir)
    _write_dashboard_json(aligned, args.output_dir)
    _write_html(aligned, args.output_dir)
    print(f"comparison_gui: {os.path.join(args.output_dir, 'visual_demo.html')}")


if __name__ == "__main__":
    raise SystemExit(main())
