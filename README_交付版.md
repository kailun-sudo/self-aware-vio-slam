# Self-Aware VIO-SLAM

## 项目目标

这个 unified project 的目标不是分别维护两个独立工程，而是构建一条完整系统链路：

1. `VIO-SLAM` 作为主 SLAM 系统
2. `self_aware_slam` 作为可靠性/失效预测模块
3. 从主 SLAM 中导出内部 tracking metrics
4. 将 metrics 输入 self-aware 模型
5. 输出：
   - `confidence_score`
   - `failure_probability`
   - `predicted_localization_reliability`
6. 进一步支持：
   - 结果分析
   - 训练数据打包

## 当前系统结构

```text
/Users/kailunwang/Desktop/ossa
├── VIO-SLAM
├── self_aware_slam
├── self_aware_slam_legacy
├── integration
├── outputs
├── README_运行指南.md
├── README_教学指南.md
└── README_交付版.md
```

### VIO-SLAM

主入口：

```text
VIO-SLAM/run_pipeline.py
```

作用：

- 读取 EuRoC `mav0`
- 运行 notebook-derived sliding-window VIO
- 导出：
  - `slam_metrics.csv`
  - `estimated_tum.txt`
  - `trajectory.txt`
  - `trajectory.pkl`

### self_aware_slam

主要负责：

- 读取 `slam_metrics.csv`
- 特征工程与特征映射
- 时间窗构建
- 调用已训练 reliability/failure model
- 输出预测结果

### integration

桥接层脚本：

- `run_offline_unified_demo.py`
  - 轨迹对齐
  - pose error 生成
  - self-aware 推理

- `analyze_unified_results.py`
  - 预测结果分析
  - 图表输出

- `run_batch_unified_pipeline.sh`
  - 多序列批量运行

- `run_euroc_degradation_demo.py`
  - 单序列 baseline vs degraded 仿真与 comparison GUI

- `run_multisequence_degradation_sweep.py`
  - 多序列、多场景 degradation sweep

- `create_multisequence_degradation_report.py`
  - 跨序列汇总报告与总 GUI

- `run_model_validity_benchmark.py`
  - 模型有效性验证
  - 相关性 / ROC / 校准 / heuristic 对比

## 当前系统输入输出

### 输入

- EuRoC `mav0` 数据
- 已训练好的 self-aware 模型与统计文件

### 输出

以 `MH_01` 为例：

主 SLAM 输出：

```text
outputs/mh01/
├── slam_metrics.csv
├── estimated_tum.txt
├── trajectory.txt
├── trajectory.pkl
└── summary.txt
```

self-aware 输出：

```text
outputs/mh01_self_aware/
├── slam_metrics.csv
├── groundtruth.csv
├── estimated.txt
├── pose_errors.csv
├── reliability_predictions.csv
└── summary.txt
```

分析输出：

```text
outputs/mh01_analysis/
├── analysis_summary.txt
├── scores_over_time.png
└── pose_error_comparison.png
```

训练打包输出：

```text
self_aware_slam/slam_metrics_dataset/MH_01_unified/
├── slam_metrics.csv
├── pose_errors.csv
├── groundtruth.csv
├── estimated.csv
└── packaging_summary.txt
```

多序列 degradation sweep 输出：

```text
outputs/multisequence_degradation_grid/
├── sweep_results.csv
├── MH_01_easy/
├── MH_02_easy/
├── MH_03_medium/
├── MH_04_difficult/
├── MH_05_difficult/
└── report/
    ├── multi_sequence_summary.txt
    ├── scenario_aggregate.csv
    ├── sequence_aggregate.csv
    ├── benchmark_runs.csv
    ├── benchmark_scenario_severity.csv
    ├── benchmark_failure_delta_pivot.csv
    ├── benchmark_failure_delta_pivot.md
    ├── multi_sequence_overview.png
    └── visual_demo.html
```

Model validity benchmark 输出：

```text
outputs/multisequence_degradation_grid/model_validity/
├── frame_level_validity_data.csv
├── run_level_correlations.csv
├── sequence_validity_summary.csv
├── scenario_validity_summary.csv
├── threshold_metrics.csv
├── validity_summary.txt
├── model_vs_actual_scatter.png
├── sequence_correlation_overview.png
├── roc_comparison_t1p0.png
├── roc_comparison_t3p0.png
├── calibration_t1p0.png
└── calibration_t3p0.png
```

v2 训练数据输出：

```text
self_aware_slam/results/
└── train_dataset_v2.pkl
```

这份 v2 数据集的定义已经和旧版训练缓存分开：

- 输入：22 维 trend-aware learning features
- 回归目标：未来 10 帧内的 `future_max_pose_error`
- 分类目标：`future_max_pose_error > 0.18m` 或未来 tracking lost
- 数据源：长 baseline 序列 + degraded replay runs
- degraded split：按 `(sequence, base_scenario)` 的 replay family 切分，避免同一 replay family 跨 split
- source_mode=auto：优先 `hybrid`，否则 `sweep_runs`，最后才回退到 `sequence_dirs`

## 最小运行流程

### 1. 跑主 SLAM

```bash
cd /Users/kailunwang/Desktop/ossa/VIO-SLAM
./.venv/bin/python run_pipeline.py \
  --output /Users/kailunwang/Desktop/ossa/outputs/mh01
```

### 2. 跑 unified demo

```bash
cd /Users/kailunwang/Desktop/ossa
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_offline_unified_demo.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --groundtruth /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0/state_groundtruth_estimate0/data.csv \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware \
  --config /Users/kailunwang/Desktop/ossa/self_aware_slam/configs/config.yaml
```

### 3. 分析结果

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/analyze_unified_results.py \
  --predictions /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/reliability_predictions.csv \
  --summary /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/summary.txt \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_analysis
```

### 4. 打包训练序列

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/self_aware_slam/scripts/package_unified_sequence.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --groundtruth /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0/state_groundtruth_estimate0/data.csv \
  --sequence-name MH_01_unified \
  --dataset-root /Users/kailunwang/Desktop/ossa/self_aware_slam/slam_metrics_dataset
```

### 5. 构建 v2 训练数据集

```bash
cd /Users/kailunwang/Desktop/ossa/self_aware_slam
./venv/bin/python -m src.data.dataset_builder
```

当前这一步会生成：

- `self_aware_slam/results/train_dataset_v2.pkl`

并且已经验证过：

- `source_mode = hybrid`
- `window_size = 10`
- `feature_dim = 22`
- `train failure rate ≈ 9.5%`
- `val failure rate ≈ 20.5%`
- `test failure rate ≈ 20.6%`
- `train y_error range ≈ [0.116, 9.079]`
- `val y_error range ≈ [0.115, 9.994]`
- `test y_error range ≈ [0.116, 9.063]`

## 批量运行

如果你要一次跑多个序列：

```bash
bash /Users/kailunwang/Desktop/ossa/integration/run_batch_unified_pipeline.sh \
  --dataset-root /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences \
  --sequences MH_01_easy,MH_02_easy,MH_03_medium
```

如果你要展示更完整的 stress test 能力，可以再跑：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_multisequence_degradation_sweep.py \
  --dataset-root /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences \
  --sequences MH_01_easy,MH_02_easy,MH_03_medium,MH_04_difficult,MH_05_difficult \
  --scenarios blur_bias,noise_amp,lighting_dropout,dropout_bias \
  --severity-grid 0.45,0.70 \
  --output-root /Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid
```

如果你要进一步验证“模型到底对不对”，继续跑：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_model_validity_benchmark.py \
  --sweep-results /Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid/sweep_results.csv \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid/model_validity \
  --failure-thresholds 0.3,1.0,3.0 \
  --summary-threshold 3.0
```

目录约定是：

```text
<dataset-root>/<sequence>/mav0
```

例如：

```text
/Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences/MH_01_easy/mav0
/Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences/MH_02_easy/mav0
```

## 当前完成度

当前系统已经具备：

- 单序列主 SLAM 运行
- SLAM metrics 导出
- TUM 轨迹导出
- unified self-aware 推理
- 结果分析
- 训练序列打包
- 多序列批处理入口
- 跨序列 benchmark 表和 severity 网格 sweep
- 模型 validity benchmark（相关性 / ROC / 校准 / heuristic 对比）
- v2 self-awareness 训练数据路径（统一 future target + trend-aware features）

## 当前限制

- 当前主 SLAM 是 notebook-derived 纯 Python VIO，不是完整 C++ ORB-SLAM3
- self-aware 模型已能接入，但还存在域偏移，结果可用于系统展示，不适合当成最终精度结论
- 当前 validity benchmark 已说明：模型目前“有反应”，但“还没有被证明是对的”
- 因此仓库里已经新增 v2 训练线，用统一 future target 和 22 维 trend-aware features 来替换旧任务定义
- 当前主集成是离线式，不是在线每帧推理
- 当前公开 benchmark 已覆盖 `MH_01 ~ MH_05`，但 `V1 / V2` 还未纳入同一套公开 sweep

## 当前建议

如果你的目标是“把系统做出来”，下一步优先做：

1. 把 `V1 / V2` 序列接进现有 severity-grid benchmark
2. 继续积累更多 `*_unified` 训练序列
3. 只做最低限度模型校准，不深挖调参
4. 把项目作为完整 pipeline + benchmark package 展示，而不是只展示单个模型指标
