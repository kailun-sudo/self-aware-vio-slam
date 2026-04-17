#!/usr/bin/env python3
"""
Create an aggregate multi-sequence degradation report and interactive HTML summary.
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


def _write_summary(results: pd.DataFrame, output_dir: Path):
    summary_path = output_dir / "multi_sequence_summary.txt"
    scenario_group = (
        results.groupby("scenario", dropna=False)[
            [
                "mean_inlier_ratio_delta",
                "pose_error_mean_delta",
                "failure_probability_mean_delta",
                "confidence_mean_delta",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    sequence_group = (
        results.groupby("sequence_short", dropna=False)[
            [
                "mean_inlier_ratio_delta",
                "pose_error_mean_delta",
                "failure_probability_mean_delta",
                "confidence_mean_delta",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    scenario_group.to_csv(output_dir / "scenario_aggregate.csv", index=False)
    sequence_group.to_csv(output_dir / "sequence_aggregate.csv", index=False)

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Multi-sequence degradation sweep summary\n")
        handle.write(f"num_sequences: {results['sequence'].nunique()}\n")
        handle.write(f"num_scenarios: {results['scenario'].nunique()}\n")
        handle.write(f"num_runs: {len(results)}\n")
        handle.write(f"sequences: {','.join(sorted(results['sequence'].unique()))}\n")
        handle.write(f"scenarios: {','.join(sorted(results['scenario'].unique()))}\n\n")

        if not scenario_group.empty:
            worst_failure = scenario_group.sort_values(
                "failure_probability_mean_delta", ascending=False
            ).iloc[0]
            worst_pose = scenario_group.sort_values(
                "pose_error_mean_delta", ascending=False
            ).iloc[0]
            handle.write(
                f"largest_failure_delta_scenario: {worst_failure['scenario']} ({worst_failure['failure_probability_mean_delta']:.6f})\n"
            )
            handle.write(
                f"largest_pose_error_delta_scenario: {worst_pose['scenario']} ({worst_pose['pose_error_mean_delta']:.6f})\n\n"
            )

        handle.write("[Scenario averages]\n")
        for _, row in scenario_group.iterrows():
            handle.write(
                f"{row['scenario']}: "
                f"inlier_delta={row.get('mean_inlier_ratio_delta', float('nan')):.6f}, "
                f"pose_error_delta={row.get('pose_error_mean_delta', float('nan')):.6f}, "
                f"failure_delta={row.get('failure_probability_mean_delta', float('nan')):.6f}, "
                f"confidence_delta={row.get('confidence_mean_delta', float('nan')):.6f}\n"
            )

        handle.write("\n[Sequence averages]\n")
        for _, row in sequence_group.iterrows():
            handle.write(
                f"{row['sequence_short']}: "
                f"inlier_delta={row.get('mean_inlier_ratio_delta', float('nan')):.6f}, "
                f"pose_error_delta={row.get('pose_error_mean_delta', float('nan')):.6f}, "
                f"failure_delta={row.get('failure_probability_mean_delta', float('nan')):.6f}, "
                f"confidence_delta={row.get('confidence_mean_delta', float('nan')):.6f}\n"
            )


def _plot_overview(results: pd.DataFrame, output_dir: Path):
    labels = [f"{row.sequence_short}\n{row.scenario}" for row in results.itertuples()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plots = [
        ("failure_probability_mean_delta", "Failure Probability Delta", "tab:red"),
        ("confidence_mean_delta", "Confidence Delta", "tab:blue"),
        ("pose_error_mean_delta", "Pose Error Delta", "tab:orange"),
        ("mean_inlier_ratio_delta", "Inlier Ratio Delta", "tab:green"),
    ]
    for axis, (column, title, color) in zip(axes.flatten(), plots):
        if column not in results.columns:
            axis.axis("off")
            continue
        axis.bar(x, results[column], color=color, alpha=0.85)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "multi_sequence_overview.png")
    plt.close(fig)


def _top_findings(results: pd.DataFrame) -> dict[str, dict[str, object]]:
    findings: dict[str, dict[str, object]] = {}
    for column, key in [
        ("failure_probability_mean_delta", "largest_failure_delta"),
        ("pose_error_mean_delta", "largest_pose_error_delta"),
        ("confidence_mean_delta", "largest_confidence_delta"),
    ]:
        if column not in results.columns or results[column].dropna().empty:
            continue
        row = results.sort_values(column, ascending=False).iloc[0]
        findings[key] = {
            "sequence": row["sequence_short"],
            "scenario": row["scenario"],
            "value": float(row[column]),
            "gui_path": row.get("comparison_gui_path"),
        }
    return findings


def _write_dashboard_json(results: pd.DataFrame, output_dir: Path):
    payload = results.replace({np.nan: None}).to_dict(orient="records")
    summary = {
        "num_sequences": int(results["sequence"].nunique()),
        "num_scenarios": int(results["scenario"].nunique()),
        "num_runs": int(len(results)),
        "top_findings": _top_findings(results),
    }
    with open(output_dir / "dashboard_data.json", "w", encoding="utf-8") as handle:
        json.dump({"rows": payload, "summary": summary}, handle, ensure_ascii=False)


def _write_html(results: pd.DataFrame, output_dir: Path):
    rows = results.replace({np.nan: None}).to_dict(orient="records")
    summary = {
        "num_sequences": int(results["sequence"].nunique()),
        "num_scenarios": int(results["scenario"].nunique()),
        "num_runs": int(len(results)),
        "top_findings": _top_findings(results),
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Multi-Sequence Degradation Sweep</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --panel: #fffdfa;
      --ink: #17212b;
      --muted: #69707d;
      --line: #e3d7c6;
      --blue: #2a6f97;
      --red: #c44536;
      --gold: #d17b0f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(42,111,151,0.10), transparent 25%),
        radial-gradient(circle at bottom right, rgba(196,69,54,0.10), transparent 30%),
        var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .page {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 14px 30px rgba(60,55,48,0.06);
      padding: 20px;
      margin-bottom: 18px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.84);
    }}
    .card strong {{
      display: block;
      color: var(--muted);
      text-transform: uppercase;
      font-size: 0.78rem;
      margin-bottom: 8px;
    }}
    .value {{ font-size: 1.6rem; font-weight: 700; }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      align-items: end;
    }}
    label {{
      display: block;
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
    }}
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: white;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.94rem; }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); text-transform: uppercase; font-size: 0.78rem; }}
    a {{
      color: var(--blue);
      text-decoration: none;
      font-weight: 600;
    }}
    a:hover {{ text-decoration: underline; }}
    .findings {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .finding {{
      border-radius: 16px;
      padding: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,241,232,0.98));
      border: 1px solid var(--line);
    }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <div class="page">
    <section class="panel">
      <h1>Multi-Sequence Degradation Sweep</h1>
      <p class="muted">
        Aggregate view of representative EuRoC degradation scenarios across multiple sequences.
        Use the filters below to inspect scenario-level deltas and jump into per-run comparison GUIs.
      </p>
      <div class="cards">
        <div class="card"><strong>Sequences</strong><span class="value">{summary["num_sequences"]}</span></div>
        <div class="card"><strong>Scenarios</strong><span class="value">{summary["num_scenarios"]}</span></div>
        <div class="card"><strong>Runs</strong><span class="value">{summary["num_runs"]}</span></div>
      </div>
      <div class="findings" id="findings"></div>
    </section>

    <section class="panel">
      <div class="controls">
        <div>
          <label for="sequenceFilter">Sequence</label>
          <select id="sequenceFilter"></select>
        </div>
        <div>
          <label for="scenarioFilter">Scenario</label>
          <select id="scenarioFilter"></select>
        </div>
        <div>
          <label for="cameraFilter">Camera Degradation</label>
          <select id="cameraFilter"></select>
        </div>
        <div>
          <label for="imuFilter">IMU Degradation</label>
          <select id="imuFilter"></select>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Runs</h2>
      <table>
        <thead>
          <tr>
            <th>Sequence</th>
            <th>Scenario</th>
            <th>Camera</th>
            <th>IMU</th>
            <th>Severity</th>
            <th>Failure Δ</th>
            <th>Confidence Δ</th>
            <th>Pose Error Δ</th>
            <th>Inlier Δ</th>
            <th>GUI</th>
          </tr>
        </thead>
        <tbody id="runsTable"></tbody>
      </table>
    </section>
  </div>

  <script>
    const rows = {json.dumps(rows, ensure_ascii=False)};
    const summary = {json.dumps(summary, ensure_ascii=False)};

    const filters = {{
      sequence: document.getElementById('sequenceFilter'),
      scenario: document.getElementById('scenarioFilter'),
      camera: document.getElementById('cameraFilter'),
      imu: document.getElementById('imuFilter'),
    }};
    const runsTable = document.getElementById('runsTable');
    const findings = document.getElementById('findings');

    function uniqueValues(key) {{
      return ['All'].concat([...new Set(rows.map((row) => row[key]).filter(Boolean))].sort());
    }}

    function populateSelect(select, values) {{
      select.innerHTML = values.map((value) => `<option value="${{value}}">${{value}}</option>`).join('');
    }}

    function formatDelta(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) return '—';
      const numeric = Number(value);
      const sign = numeric > 0 ? '+' : '';
      return `${{sign}}${{numeric.toFixed(4)}}`;
    }}

    function renderFindings() {{
      const entries = Object.entries(summary.top_findings || {{}});
      findings.innerHTML = entries.map(([key, item]) => `
        <div class="finding">
          <strong>${{key.replaceAll('_', ' ')}}</strong>
          <div style="font-size: 1.15rem; font-weight: 700; margin: 8px 0;">${{item.sequence}} / ${{item.scenario}}</div>
          <div class="muted">value: ${{Number(item.value).toFixed(4)}}</div>
          ${{item.gui_path ? `<div style="margin-top: 10px;"><a href="file://${{item.gui_path}}">Open per-run GUI</a></div>` : ''}}
        </div>
      `).join('');
    }}

    function filteredRows() {{
      return rows.filter((row) => {{
        return (
          (filters.sequence.value === 'All' || row.sequence_short === filters.sequence.value) &&
          (filters.scenario.value === 'All' || row.scenario === filters.scenario.value) &&
          (filters.camera.value === 'All' || row.camera_degradation === filters.camera.value) &&
          (filters.imu.value === 'All' || row.imu_degradation === filters.imu.value)
        );
      }});
    }}

    function renderTable() {{
      const filtered = filteredRows();
      runsTable.innerHTML = filtered.map((row) => `
        <tr>
          <td>${{row.sequence_short}}</td>
          <td><strong>${{row.scenario}}</strong><div class="muted">${{row.description || ''}}</div></td>
          <td>${{row.camera_degradation}}</td>
          <td>${{row.imu_degradation}}</td>
          <td>${{Number(row.severity).toFixed(2)}}</td>
          <td>${{formatDelta(row.failure_probability_mean_delta)}}</td>
          <td>${{formatDelta(row.confidence_mean_delta)}}</td>
          <td>${{formatDelta(row.pose_error_mean_delta)}}</td>
          <td>${{formatDelta(row.mean_inlier_ratio_delta)}}</td>
          <td>${{row.comparison_gui_path ? `<a href="file://${{row.comparison_gui_path}}">Open GUI</a>` : '—'}}</td>
        </tr>
      `).join('');
    }}

    populateSelect(filters.sequence, uniqueValues('sequence_short'));
    populateSelect(filters.scenario, uniqueValues('scenario'));
    populateSelect(filters.camera, uniqueValues('camera_degradation'));
    populateSelect(filters.imu, uniqueValues('imu_degradation'));
    Object.values(filters).forEach((select) => select.addEventListener('change', renderTable));
    renderFindings();
    renderTable();
  </script>
</body>
</html>
"""
    (output_dir / "visual_demo.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create aggregate multisequence degradation report")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_summary(results, output_dir)
    _plot_overview(results, output_dir)
    _write_dashboard_json(results, output_dir)
    _write_html(results, output_dir)
    print(f"report_dir: {output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
