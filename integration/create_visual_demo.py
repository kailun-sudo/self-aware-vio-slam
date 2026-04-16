#!/usr/bin/env python3
"""
Create a richer visual demo for the unified VIO-SLAM + self-awareness pipeline.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_estimated(estimated_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        estimated_path,
        sep=' ',
        header=None,
        names=['timestamp', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'],
    )
    return df


def build_demo_dataframe(metrics_path: str,
                         predictions_path: str,
                         estimated_path: str) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path).sort_values('timestamp').reset_index(drop=True)
    predictions = pd.read_csv(predictions_path).sort_values('timestamp').reset_index(drop=True)
    estimated = _load_estimated(estimated_path).sort_values('timestamp').reset_index(drop=True)

    demo = pd.merge_asof(
        predictions,
        estimated[['timestamp', 'px', 'py', 'pz']],
        on='timestamp',
        direction='nearest',
        tolerance=0.02,
    )
    demo = pd.merge_asof(
        demo.sort_values('timestamp'),
        metrics[[
            'timestamp',
            'num_matches',
            'num_inliers',
            'inlier_ratio',
            'mean_epipolar_error',
            'pose_optimization_residual',
            'processing_time_ms',
            'trajectory_increment_norm',
        ]].sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=0.02,
    )
    return demo


def _write_summary(demo: pd.DataFrame, output_dir: str):
    summary_path = os.path.join(output_dir, 'visual_demo_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        handle.write('Unified visual demo summary\n')
        handle.write(f'rows: {len(demo)}\n')
        handle.write(f'failure_probability_mean: {demo["failure_probability"].mean():.6f}\n')
        handle.write(f'confidence_mean: {demo["confidence_score"].mean():.6f}\n')
        if 'actual_pose_error' in demo.columns:
            handle.write(f'actual_pose_error_mean: {demo["actual_pose_error"].mean():.6f}\n')
        handle.write(f'max_failure_probability: {demo["failure_probability"].max():.6f}\n')


def _plot_trajectory_risk(demo: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    scatter = axes[0].scatter(
        demo['px'],
        demo['py'],
        c=demo['failure_probability'],
        cmap='inferno',
        s=18,
    )
    axes[0].plot(demo['px'], demo['py'], color='lightgray', linewidth=1, alpha=0.6)
    axes[0].set_title('Trajectory Colored by Failure Probability')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(scatter, ax=axes[0], label='failure_probability')

    if 'actual_pose_error' in demo.columns:
        scatter2 = axes[1].scatter(
            demo['px'],
            demo['py'],
            c=demo['actual_pose_error'],
            cmap='viridis',
            s=18,
        )
        axes[1].plot(demo['px'], demo['py'], color='lightgray', linewidth=1, alpha=0.6)
        axes[1].set_title('Trajectory Colored by Actual Pose Error')
        axes[1].set_xlabel('x (m)')
        axes[1].set_ylabel('y (m)')
        axes[1].axis('equal')
        axes[1].grid(True, alpha=0.3)
        fig.colorbar(scatter2, ax=axes[1], label='actual_pose_error (m)')
    else:
        axes[1].plot(demo['px'], demo['py'], color='tab:blue')
        axes[1].set_title('Estimated Trajectory')
        axes[1].set_xlabel('x (m)')
        axes[1].set_ylabel('y (m)')
        axes[1].axis('equal')
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'trajectory_risk_map.png'))
    plt.close(fig)


def _plot_system_dashboard(demo: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex='col')

    axes[0, 0].plot(demo['timestamp'], demo['failure_probability'], color='tab:red', label='failure_probability')
    axes[0, 0].plot(demo['timestamp'], demo['confidence_score'], color='tab:blue', label='confidence_score')
    axes[0, 0].set_title('Self-Awareness Scores')
    axes[0, 0].set_ylabel('score')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    if 'actual_pose_error' in demo.columns:
        axes[0, 1].plot(demo['timestamp'], demo['actual_pose_error'], color='tab:green', label='actual_pose_error')
    axes[0, 1].plot(demo['timestamp'], demo['predicted_pose_error'], color='tab:orange', label='predicted_pose_error')
    axes[0, 1].set_title('Pose Error')
    axes[0, 1].set_ylabel('meters')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(demo['timestamp'], demo['inlier_ratio'], color='tab:purple', label='inlier_ratio')
    axes[1, 0].plot(demo['timestamp'], demo['num_inliers'] / np.maximum(demo['num_matches'], 1), color='tab:pink', alpha=0.6, label='num_inliers/num_matches')
    axes[1, 0].set_title('Tracking Quality')
    axes[1, 0].set_xlabel('timestamp (s)')
    axes[1, 0].set_ylabel('ratio')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(demo['timestamp'], demo['pose_optimization_residual'], color='tab:brown', label='pose_optimization_residual')
    axes[1, 1].plot(demo['timestamp'], demo['mean_epipolar_error'], color='tab:gray', alpha=0.8, label='mean_epipolar_error')
    axes[1, 1].set_title('Optimization / Geometry Residuals')
    axes[1, 1].set_xlabel('timestamp (s)')
    axes[1, 1].set_ylabel('residual')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'system_dashboard.png'))
    plt.close(fig)


def _save_top_risky_frames(demo: pd.DataFrame, output_dir: str) -> str:
    columns = [
        'timestamp',
        'frame_id',
        'failure_probability',
        'confidence_score',
        'predicted_pose_error',
        'actual_pose_error',
        'inlier_ratio',
        'num_matches',
        'num_inliers',
        'pose_optimization_residual',
    ]
    available = [column for column in columns if column in demo.columns]
    top_risky = demo.sort_values('failure_probability', ascending=False).head(15)[available].reset_index(drop=True)
    output_path = os.path.join(output_dir, 'top_risky_frames.csv')
    top_risky.to_csv(output_path, index=False)
    return output_path


def _write_dashboard_json(demo: pd.DataFrame, output_dir: str) -> str:
    payload_columns = [
        'timestamp',
        'frame_id',
        'failure_probability',
        'confidence_score',
        'predicted_pose_error',
        'predicted_localization_reliability',
        'predicted_failure',
        'actual_pose_error',
        'actual_rotation_error_deg',
        'px',
        'py',
        'pz',
        'num_matches',
        'num_inliers',
        'inlier_ratio',
        'mean_epipolar_error',
        'pose_optimization_residual',
        'processing_time_ms',
        'trajectory_increment_norm',
    ]
    available = [column for column in payload_columns if column in demo.columns]
    dashboard_rows = demo[available].copy()
    dashboard_rows = dashboard_rows.replace({np.nan: None})

    output_path = os.path.join(output_dir, 'dashboard_data.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'rows': dashboard_rows.to_dict(orient='records'),
                'summary': {
                    'num_rows': int(len(demo)),
                    'failure_probability_mean': float(demo['failure_probability'].mean()),
                    'confidence_mean': float(demo['confidence_score'].mean()),
                    'actual_pose_error_mean': float(demo['actual_pose_error'].mean()) if 'actual_pose_error' in demo.columns else None,
                    'max_failure_probability': float(demo['failure_probability'].max()),
                },
            },
            handle,
            ensure_ascii=False,
        )
    return output_path


def _write_html_report(demo: pd.DataFrame, output_dir: str):
    top_risky = demo.sort_values('failure_probability', ascending=False).head(10).reset_index(drop=True)
    dashboard_rows = demo.replace({np.nan: None}).to_dict(orient='records')
    top_risky_rows = top_risky.replace({np.nan: None}).to_dict(orient='records')
    has_actual_pose_error = 'actual_pose_error' in demo.columns and demo['actual_pose_error'].notna().any()
    actual_pose_error_display = f"{demo['actual_pose_error'].mean():.3f}" if has_actual_pose_error else "N/A"
    third_series_label = 'actual_pose_error' if has_actual_pose_error else 'predicted_pose_error'
    third_series_description = (
        "这里同步展示 `failure_probability`、`confidence_score` 和 `actual_pose_error`，便于看“模型判断”和“真实误差”是否同涨同跌。"
        if has_actual_pose_error else
        "这里同步展示 `failure_probability`、`confidence_score` 和 `predicted_pose_error`，适合观察在线运行时模型内部风险判断如何变化。"
    )

    html_content = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Unified Visual GUI Demo</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffdf9;
      --ink: #17202a;
      --muted: #69727d;
      --accent: #bf360c;
      --accent-2: #1e6091;
      --accent-3: #6a994e;
      --line: #e1d8c7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255, 209, 102, 0.25), transparent 24%),
        radial-gradient(circle at bottom right, rgba(30, 96, 145, 0.12), transparent 30%),
        var(--bg);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2, h3 {{ margin: 0 0 10px 0; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .hero {{
      background: linear-gradient(135deg, rgba(191, 54, 12, 0.08), rgba(30, 96, 145, 0.08));
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      margin-bottom: 22px;
      box-shadow: 0 18px 40px rgba(60, 55, 48, 0.08);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 12px 24px rgba(60, 55, 48, 0.05);
    }}
    .card .value {{
      display: block;
      margin-top: 8px;
      font-size: 1.7rem;
      font-weight: 700;
      color: var(--accent);
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 12px;
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      margin-bottom: 20px;
    }}
    .controls button {{
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 10px 18px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      cursor: pointer;
    }}
    .controls button.secondary {{
      background: #39424e;
    }}
    .controls input[type="range"] {{
      width: 100%;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.25fr 0.95fr;
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 14px 30px rgba(60, 55, 48, 0.06);
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    canvas {{
      width: 100%;
      height: auto;
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,244,236,0.98));
      border: 1px solid var(--line);
    }}
    .frame-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 16px;
      margin-top: 14px;
    }}
    .frame-grid .item {{
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(23, 32, 42, 0.03);
    }}
    .item strong {{
      display: block;
      font-size: 0.78rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .item span {{
      font-size: 1.05rem;
      font-weight: 600;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
    }}
    tr:hover {{ background: rgba(191, 54, 12, 0.04); }}
    .jump-btn {{
      border: none;
      background: rgba(30, 96, 145, 0.12);
      color: var(--accent-2);
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
    .legend .risk::before {{ background: #c1121f; }}
    .legend .conf::before {{ background: #1e6091; }}
    .legend .err::before {{ background: #6a994e; }}
    .legend .inlier::before {{ background: #7b2cbf; }}
    .note {{
      margin-top: 16px;
      padding: 14px;
      border-radius: 14px;
      background: rgba(106, 153, 78, 0.08);
      color: #3a4a2c;
    }}
    @media (max-width: 1100px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>ORB-SLAM with Self-Awareness GUI Demo</h1>
      <p>这不是静态报告，而是一个可交互的本地 GUI 页面。你可以拖动时间轴、自动播放、跳到高风险帧，同时观察轨迹位置、失效概率、置信度、真实误差和 tracking 指标如何一起变化。</p>

      <div class="metrics">
        <div class="card"><strong>预测帧数</strong><span class="value">{len(demo)}</span></div>
        <div class="card"><strong>平均失效概率</strong><span class="value">{demo['failure_probability'].mean():.3f}</span></div>
        <div class="card"><strong>平均置信度</strong><span class="value">{demo['confidence_score'].mean():.3f}</span></div>
        <div class="card"><strong>平均真实误差</strong><span class="value">{actual_pose_error_display}</span></div>
      </div>
    </section>

    <section class="controls">
      <div>
        <button id="playButton">播放</button>
        <button id="resetButton" class="secondary">回到开头</button>
      </div>
      <input id="frameSlider" type="range" min="0" max="{max(len(demo) - 1, 0)}" value="0">
      <div id="statusText">Frame 0 / {max(len(demo) - 1, 0)}</div>
    </section>

    <section class="layout">
      <div class="stack">
        <div class="panel">
          <h2>1. 轨迹 + 风险</h2>
          <p>所有轨迹点会按失效概率着色，白色圆环表示当前时间点。颜色越接近红色，self-aware 模块越认为这一帧附近存在定位风险。</p>
          <canvas id="trajectoryCanvas" width="820" height="520"></canvas>
        </div>

        <div class="panel">
          <h2>2. 分数时间线</h2>
          <p>{third_series_description}</p>
          <canvas id="scoreCanvas" width="820" height="320"></canvas>
          <div class="legend">
            <span class="risk">failure_probability</span>
            <span class="conf">confidence_score</span>
            <span class="err">{third_series_label}</span>
          </div>
        </div>
      </div>

      <div class="stack">
        <div class="panel">
          <h2>3. 当前帧状态</h2>
          <p id="frameSummary">拖动滑条或点击播放，查看任意时刻主 SLAM 和 self-aware 的联合状态。</p>
          <div class="frame-grid" id="frameGrid"></div>
        </div>

        <div class="panel">
          <h2>4. Tracking 指标时间线</h2>
          <p>这里帮助你理解“为什么模型会觉得危险”，比如 `inlier_ratio` 降低、`pose_optimization_residual` 抬高时，风险往往会同步升高。</p>
          <canvas id="metricsCanvas" width="620" height="260"></canvas>
          <div class="legend">
            <span class="inlier">inlier_ratio</span>
            <span class="err">pose_optimization_residual</span>
          </div>
        </div>

        <div class="panel">
          <h2>5. Top Risky Frames</h2>
          <p>点“跳转”可以直接把 GUI 定位到那一帧。</p>
          <table>
            <thead>
              <tr><th>frame_id</th><th>failure_probability</th><th>actual_pose_error</th><th>inlier_ratio</th><th>跳转</th></tr>
            </thead>
            <tbody id="topRiskyBody"></tbody>
          </table>
          <div class="note">
            当前这版 demo 主要用于直观理解系统闭环，不是论文级评估界面。它的重点是把
            “轨迹位置、SLAM 内部指标、self-aware 输出”放到同一屏里。
          </div>
        </div>
      </div>
    </section>
  </div>
  <script>
    const rows = {json.dumps(dashboard_rows, ensure_ascii=False)};
    const topRiskyRows = {json.dumps(top_risky_rows, ensure_ascii=False)};

    const statusText = document.getElementById('statusText');
    const slider = document.getElementById('frameSlider');
    const playButton = document.getElementById('playButton');
    const resetButton = document.getElementById('resetButton');
    const frameGrid = document.getElementById('frameGrid');
    const frameSummary = document.getElementById('frameSummary');
    const topRiskyBody = document.getElementById('topRiskyBody');

    const trajectoryCanvas = document.getElementById('trajectoryCanvas');
    const scoreCanvas = document.getElementById('scoreCanvas');
    const metricsCanvas = document.getElementById('metricsCanvas');
    const trajectoryCtx = trajectoryCanvas.getContext('2d');
    const scoreCtx = scoreCanvas.getContext('2d');
    const metricsCtx = metricsCanvas.getContext('2d');

    let currentIndex = 0;
    let timer = null;

    function lerp(a, b, t) {{
      return a + (b - a) * t;
    }}

    function clamp01(v) {{
      return Math.max(0, Math.min(1, v));
    }}

    function riskColor(prob) {{
      const t = clamp01(prob);
      const hue = lerp(215, 8, t);
      const sat = lerp(55, 78, t);
      const light = lerp(48, 46, t);
      return `hsl(${{hue}}, ${{sat}}%, ${{light}}%)`;
    }}

    function computeTrajectoryPoints() {{
      const padding = 42;
      const xs = rows.map(row => row.px ?? 0);
      const ys = rows.map(row => row.py ?? 0);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const width = trajectoryCanvas.width - padding * 2;
      const height = trajectoryCanvas.height - padding * 2;
      const spanX = Math.max(maxX - minX, 1e-6);
      const spanY = Math.max(maxY - minY, 1e-6);
      return rows.map(row => {{
        const x = padding + ((row.px - minX) / spanX) * width;
        const y = trajectoryCanvas.height - padding - ((row.py - minY) / spanY) * height;
        return {{ x, y }};
      }});
    }}

    const trajectoryPoints = computeTrajectoryPoints();

    function drawGrid(ctx, canvas, yTicks = 4) {{
      ctx.save();
      ctx.strokeStyle = 'rgba(60, 55, 48, 0.08)';
      ctx.lineWidth = 1;
      for (let i = 0; i <= yTicks; i += 1) {{
        const y = 24 + (canvas.height - 48) * (i / yTicks);
        ctx.beginPath();
        ctx.moveTo(28, y);
        ctx.lineTo(canvas.width - 18, y);
        ctx.stroke();
      }}
      ctx.restore();
    }}

    function drawTrajectory() {{
      trajectoryCtx.clearRect(0, 0, trajectoryCanvas.width, trajectoryCanvas.height);
      drawGrid(trajectoryCtx, trajectoryCanvas, 5);

      trajectoryCtx.lineWidth = 2;
      trajectoryCtx.strokeStyle = 'rgba(90, 90, 90, 0.24)';
      trajectoryCtx.beginPath();
      trajectoryPoints.forEach((point, idx) => {{
        if (idx === 0) trajectoryCtx.moveTo(point.x, point.y);
        else trajectoryCtx.lineTo(point.x, point.y);
      }});
      trajectoryCtx.stroke();

      trajectoryPoints.forEach((point, idx) => {{
        const row = rows[idx];
        trajectoryCtx.fillStyle = riskColor(row.failure_probability ?? 0);
        trajectoryCtx.beginPath();
        trajectoryCtx.arc(point.x, point.y, idx === currentIndex ? 5.4 : 3.1, 0, Math.PI * 2);
        trajectoryCtx.fill();
      }});

      const current = trajectoryPoints[currentIndex];
      trajectoryCtx.strokeStyle = 'white';
      trajectoryCtx.lineWidth = 3;
      trajectoryCtx.beginPath();
      trajectoryCtx.arc(current.x, current.y, 9, 0, Math.PI * 2);
      trajectoryCtx.stroke();

      trajectoryCtx.fillStyle = '#17202a';
      trajectoryCtx.font = '600 14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      trajectoryCtx.fillText('轨迹位置', 28, 24);
      trajectoryCtx.fillText('颜色 = failure probability', trajectoryCanvas.width - 210, 24);
    }}

    function normalizeSeries(values) {{
      const filtered = values.filter(v => Number.isFinite(v));
      const min = filtered.length ? Math.min(...filtered) : 0;
      const max = filtered.length ? Math.max(...filtered) : 1;
      const span = Math.max(max - min, 1e-6);
      return values.map(v => (Number.isFinite(v) ? (v - min) / span : 0));
    }}

    function drawSeriesCanvas(ctx, canvas, seriesSpecs, title) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawGrid(ctx, canvas, 4);
      const chartLeft = 36;
      const chartRight = canvas.width - 18;
      const chartTop = 26;
      const chartBottom = canvas.height - 28;
      const chartWidth = chartRight - chartLeft;
      const chartHeight = chartBottom - chartTop;

      seriesSpecs.forEach(spec => {{
        const normalized = spec.normalized;
        ctx.beginPath();
        normalized.forEach((value, idx) => {{
          const x = chartLeft + (idx / Math.max(rows.length - 1, 1)) * chartWidth;
          const y = chartBottom - value * chartHeight;
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.strokeStyle = spec.color;
        ctx.lineWidth = spec.width ?? 2;
        ctx.stroke();
      }});

      const cursorX = chartLeft + (currentIndex / Math.max(rows.length - 1, 1)) * chartWidth;
      ctx.strokeStyle = 'rgba(23, 32, 42, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cursorX, chartTop);
      ctx.lineTo(cursorX, chartBottom);
      ctx.stroke();

      ctx.fillStyle = '#17202a';
      ctx.font = '600 14px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif';
      ctx.fillText(title, 28, 18);
    }}

    function drawScoreCanvas() {{
      drawSeriesCanvas(
        scoreCtx,
        scoreCanvas,
        [
          {{
            normalized: rows.map(row => row.failure_probability ?? 0),
            color: '#c1121f',
            width: 2.5,
          }},
          {{
            normalized: rows.map(row => row.confidence_score ?? 0),
            color: '#1e6091',
            width: 2.5,
          }},
          {{
            normalized: normalizeSeries(rows.map(row => row.{third_series_label} ?? 0)),
            color: '#6a994e',
            width: 2,
          }},
        ],
        '风险 / 置信度 / 误差'
      );
    }}

    function drawMetricsCanvas() {{
      drawSeriesCanvas(
        metricsCtx,
        metricsCanvas,
        [
          {{
            normalized: rows.map(row => row.inlier_ratio ?? 0),
            color: '#7b2cbf',
            width: 2.5,
          }},
          {{
            normalized: normalizeSeries(rows.map(row => row.pose_optimization_residual ?? 0)),
            color: '#6a994e',
            width: 2.2,
          }},
        ],
        'Tracking 指标'
      );
    }}

    function formatValue(value, digits = 3) {{
      if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
      return Number(value).toFixed(digits);
    }}

    function updateFramePanel() {{
      const row = rows[currentIndex];
      statusText.textContent = `Frame ${{currentIndex}} / ${{rows.length - 1}} | frame_id=${{row.frame_id}} | failure=${{formatValue(row.failure_probability)}}`;
      frameSummary.textContent =
        `当前帧 frame_id=${{row.frame_id}}，时间戳=${{formatValue(row.timestamp, 3)}}。这里可以同时看到主 SLAM 内部 tracking 指标和 self-aware 输出。`;

      const items = [
        ['failure_probability', formatValue(row.failure_probability)],
        ['confidence_score', formatValue(row.confidence_score)],
        ['predicted_pose_error', formatValue(row.predicted_pose_error)],
        ['actual_pose_error', formatValue(row.actual_pose_error)],
        ['inlier_ratio', formatValue(row.inlier_ratio)],
        ['num_matches', formatValue(row.num_matches, 0)],
        ['num_inliers', formatValue(row.num_inliers, 0)],
        ['pose_optimization_residual', formatValue(row.pose_optimization_residual)],
        ['mean_epipolar_error', formatValue(row.mean_epipolar_error)],
        ['processing_time_ms', formatValue(row.processing_time_ms, 1)],
      ];

      frameGrid.innerHTML = items.map(([label, value]) => `
        <div class="item">
          <strong>${{label}}</strong>
          <span>${{value}}</span>
        </div>
      `).join('');
    }}

    function renderTopRiskyTable() {{
      topRiskyBody.innerHTML = topRiskyRows.map(row => {{
        const idx = rows.findIndex(candidate => candidate.frame_id === row.frame_id);
        return `
          <tr>
            <td>${{row.frame_id}}</td>
            <td>${{formatValue(row.failure_probability)}}</td>
            <td>${{formatValue(row.actual_pose_error)}}</td>
            <td>${{formatValue(row.inlier_ratio)}}</td>
            <td><button class="jump-btn" data-index="${{idx}}">跳转</button></td>
          </tr>
        `;
      }}).join('');

      topRiskyBody.querySelectorAll('.jump-btn').forEach(button => {{
        button.addEventListener('click', () => {{
          const idx = Number(button.getAttribute('data-index'));
          if (Number.isFinite(idx) && idx >= 0) {{
            setFrame(idx);
          }}
        }});
      }});
    }}

    function render() {{
      drawTrajectory();
      drawScoreCanvas();
      drawMetricsCanvas();
      updateFramePanel();
    }}

    function setFrame(index) {{
      currentIndex = Math.max(0, Math.min(rows.length - 1, index));
      slider.value = String(currentIndex);
      render();
    }}

    function togglePlay() {{
      if (timer) {{
        clearInterval(timer);
        timer = null;
        playButton.textContent = '播放';
        return;
      }}

      playButton.textContent = '暂停';
      timer = setInterval(() => {{
        if (currentIndex >= rows.length - 1) {{
          togglePlay();
          return;
        }}
        setFrame(currentIndex + 1);
      }}, 120);
    }}

    slider.addEventListener('input', event => {{
      setFrame(Number(event.target.value));
    }});
    playButton.addEventListener('click', togglePlay);
    resetButton.addEventListener('click', () => setFrame(0));

    renderTopRiskyTable();
    setFrame(0);
  </script>
</body>
</html>"""

    with open(os.path.join(output_dir, 'visual_demo.html'), 'w', encoding='utf-8') as handle:
        handle.write(html_content)


def create_visual_demo(metrics_path: str,
                       predictions_path: str,
                       estimated_path: str,
                       output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    demo = build_demo_dataframe(metrics_path, predictions_path, estimated_path)
    demo.to_csv(os.path.join(output_dir, 'demo_aligned.csv'), index=False)

    _write_summary(demo, output_dir)
    _plot_trajectory_risk(demo, output_dir)
    _plot_system_dashboard(demo, output_dir)
    top_risky_path = _save_top_risky_frames(demo, output_dir)
    dashboard_json_path = _write_dashboard_json(demo, output_dir)
    _write_html_report(demo, output_dir)

    return {
        'rows': len(demo),
        'output_dir': output_dir,
        'top_risky_path': top_risky_path,
        'dashboard_json_path': dashboard_json_path,
    }


def main():
    parser = argparse.ArgumentParser(description='Create a visual demo for unified outputs')
    parser.add_argument('--metrics', required=True, help='Path to slam_metrics.csv')
    parser.add_argument('--predictions', required=True, help='Path to reliability_predictions.csv')
    parser.add_argument('--estimated', required=True, help='Path to estimated_tum.txt')
    parser.add_argument('--output-dir', required=True, help='Directory for visual demo artifacts')
    args = parser.parse_args()

    result = create_visual_demo(
        metrics_path=args.metrics,
        predictions_path=args.predictions,
        estimated_path=args.estimated,
        output_dir=args.output_dir,
    )
    for key, value in result.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
