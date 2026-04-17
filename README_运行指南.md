# 运行指南

这份文档只保留当前有效主线，不再描述已经归档到 `legacy` 的旧 `src/vio_slam` 入口。

## 1. 当前结构

工作区根目录：

```text
/Users/kailunwang/Desktop/ossa
├── VIO-SLAM
├── self_aware_slam
├── self_aware_slam_legacy
├── integration
├── README.md
└── README_运行指南.md
```

其中：

- `VIO-SLAM/`：当前主 SLAM 系统
- `VIO-SLAM/run_pipeline.py`：当前唯一推荐入口
- `VIO-SLAM/legacy_python_vio_slam/`：旧 Python 实现归档
- `VIO-SLAM/reference/slam_reference.ipynb`：原始已跑通 notebook，仅作参考
- `self_aware_slam/`：当前正版 self-aware 模块
- `integration/`：统一离线整合脚本

## 2. 当前主流程

统一项目的主流程现在是：

1. 用 `VIO-SLAM/run_pipeline.py` 跑出：
   - `slam_metrics.csv`
   - `estimated_tum.txt`
2. 用 `integration/run_offline_unified_demo.py` 做：
   - ground truth 对齐
   - `pose_errors.csv`
   - self-aware inference
3. 得到：
   - `confidence_score`
   - `failure_probability`
   - `predicted_localization_reliability`

## 3. 数据目录

`VIO-SLAM` 读取的是 EuRoC `mav0` 目录。

标准结构：

```text
/path/to/MH_01_easy/mav0
├── cam0
│   ├── data
│   └── sensor.yaml
├── cam1
├── imu0
│   ├── data.csv
│   └── sensor.yaml
└── state_groundtruth_estimate0
    └── data.csv
```

如果你想沿用之前 notebook 的习惯，最省事的放法是：

```text
/Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0
```

这样运行时可以不写 `--data_path`。

## 4. Python 环境

### 4.1 VIO-SLAM

建议使用：

```text
/Users/kailunwang/Desktop/ossa/VIO-SLAM/.venv/bin/python
```

依赖在：

```text
/Users/kailunwang/Desktop/ossa/VIO-SLAM/requirements.txt
```

### 4.2 self-aware

建议使用：

```text
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python
```

## 4.3 一键下载 EuRoC 单序列

如果你本地没有 `mav0`，可以直接用下面的脚本下载并整理成：

```text
/Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0
```

命令：

```bash
bash /Users/kailunwang/Desktop/ossa/integration/download_euroc_mh.sh
```

默认会下载 `MH_01_easy`。如果你想换成别的序列，例如 `MH_02_easy`：

```bash
bash /Users/kailunwang/Desktop/ossa/integration/download_euroc_mh.sh \
  --sequence MH_02_easy
```

现在这个脚本已经改成适配 **EuRoC 新官方托管源** 的版本：

- 优先从 ETH Research Collection 下载对应大类压缩包
- `Machine Hall` 大包里再自动抽取你指定的单条序列
- 不再依赖老的单序列直链

如果你要准备多序列实验，不建议都放到 `VIO-SLAM/data/mav0`，而是按下面的结构整理：

```text
/Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences/MH_01_easy/mav0
/Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences/MH_02_easy/mav0
/Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences/MH_03_medium/mav0
```

例如下载 `MH_02_easy` 到多序列目录：

```bash
bash /Users/kailunwang/Desktop/ossa/integration/download_euroc_mh.sh \
  --sequence MH_02_easy \
  --output-root /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences/MH_02_easy \
  --keep-zip
```

## 5. 第一步：运行 VIO-SLAM

如果数据在默认位置 `VIO-SLAM/data/mav0`：

```bash
cd /Users/kailunwang/Desktop/ossa/VIO-SLAM
./.venv/bin/python run_pipeline.py \
  --output /Users/kailunwang/Desktop/ossa/outputs/mh01
```

如果数据在别的地方：

```bash
cd /Users/kailunwang/Desktop/ossa/VIO-SLAM
./.venv/bin/python run_pipeline.py \
  --data_path /path/to/EuRoC/MH_01_easy/mav0 \
  --output /Users/kailunwang/Desktop/ossa/outputs/mh01
```

### 5.1 这一步会生成什么

输出目录例如：

```text
/Users/kailunwang/Desktop/ossa/outputs/mh01
```

关键文件：

- `slam_metrics.csv`
- `estimated_tum.txt`
- `trajectory.txt`
- `trajectory.pkl`
- `summary.txt`

其中最关键的是前两个：

- `slam_metrics.csv`：给 self-aware 模块做推理
- `estimated_tum.txt`：给 ground truth 对齐和误差计算

## 6. 第二步：运行 unified demo

回到工作区根目录：

```bash
cd /Users/kailunwang/Desktop/ossa
```

执行：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_offline_unified_demo.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --groundtruth /path/to/EuRoC/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware \
  --config /Users/kailunwang/Desktop/ossa/self_aware_slam/configs/config.yaml
```

### 6.1 这一步会生成什么

输出目录例如：

```text
/Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware
```

关键文件：

- `pose_errors.csv`
- `reliability_predictions.csv`
- `summary.txt`

## 7. 最小可运行命令序列

如果你已经准备好了 EuRoC `mav0` 数据，直接按这个顺序执行：

### Step 1

```bash
cd /Users/kailunwang/Desktop/ossa/VIO-SLAM
./.venv/bin/python run_pipeline.py \
  --data_path /path/to/EuRoC/MH_01_easy/mav0 \
  --output /Users/kailunwang/Desktop/ossa/outputs/mh01
```

### Step 2

```bash
cd /Users/kailunwang/Desktop/ossa
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_offline_unified_demo.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --groundtruth /path/to/EuRoC/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware \
  --config /Users/kailunwang/Desktop/ossa/self_aware_slam/configs/config.yaml
```

### Step 3

看结果：

```bash
head -n 20 /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/reliability_predictions.csv
cat /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/summary.txt
```

## 7A. 在线 self-aware 版本

如果你想在 `VIO-SLAM` 主循环运行时就同步得到 self-aware 预测，而不是等结束后再离线推理：

```bash
cd /Users/kailunwang/Desktop/ossa/VIO-SLAM
./.venv/bin/python run_pipeline.py \
  --output /Users/kailunwang/Desktop/ossa/outputs/mh01_online \
  --enable_online_self_aware
```

这条命令会自动启动 `self_aware_slam/venv/bin/python` 里的 streaming predictor sidecar，
在主 VIO 每个滑窗生成 metrics 后立刻做一次预测。

会额外生成：

- `/Users/kailunwang/Desktop/ossa/outputs/mh01_online/online_predictions.csv`

查看在线预测结果：

```bash
head -n 20 /Users/kailunwang/Desktop/ossa/outputs/mh01_online/online_predictions.csv
cat /Users/kailunwang/Desktop/ossa/outputs/mh01_online/summary.txt
```

## 8. 结果分析

如果你想把 unified demo 的结果画成图，并输出更详细的统计：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/analyze_unified_results.py \
  --predictions /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/reliability_predictions.csv \
  --summary /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/summary.txt \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_analysis
```

会生成：

- `analysis_summary.txt`
- `scores_over_time.png`
- `pose_error_comparison.png`

## 8A. 可视化 Demo

如果你想把轨迹风险、真实误差和 tracking 指标放到一个更直观的页面里：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/create_visual_demo.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --predictions /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/reliability_predictions.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_visual_demo
```

会生成：

- `demo_aligned.csv`
- `trajectory_risk_map.png`
- `system_dashboard.png`
- `top_risky_frames.csv`
- `visual_demo.html`
- `visual_demo_summary.txt`

建议优先打开：

- `visual_demo.html`
- `trajectory_risk_map.png`
- `system_dashboard.png`

如果你想要的是更像 GUI 的交互版本：

- 现在生成的 `visual_demo.html` 已经是可交互页面
- 页面内支持拖动时间轴、自动播放、跳转到高风险帧
- 会同步显示轨迹位置、风险分数、真实误差和 tracking 指标

例如：

```text
/Users/kailunwang/Desktop/ossa/outputs/mh01_visual_demo_gui/visual_demo.html
```

双击或用浏览器打开即可。

## 9. 打包成训练序列

如果你想把这次 unified 输出打包成 self-aware 训练可复用的序列目录：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/self_aware_slam/scripts/package_unified_sequence.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --groundtruth /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0/state_groundtruth_estimate0/data.csv \
  --sequence-name MH_01_unified \
  --dataset-root /Users/kailunwang/Desktop/ossa/self_aware_slam/slam_metrics_dataset
```

会生成：

```text
/Users/kailunwang/Desktop/ossa/self_aware_slam/slam_metrics_dataset/MH_01_unified
├── slam_metrics.csv
├── pose_errors.csv
├── groundtruth.csv
├── estimated.csv
└── packaging_summary.txt
```

## 10. Linux VM 一键跑法

Linux VM 上可以继续用：

```bash
bash /Users/kailunwang/Desktop/ossa/integration/check_env.sh \
  --workspace /Users/kailunwang/Desktop/ossa \
  --dataset /path/to/EuRoC/MH_01_easy/mav0
```

然后：

```bash
bash /Users/kailunwang/Desktop/ossa/integration/run_linux_vm_pipeline.sh \
  --workspace /Users/kailunwang/Desktop/ossa \
  --dataset /path/to/EuRoC/MH_01_easy/mav0 \
  --output-name mh01
```

这两个脚本现在也已经切到新的 `VIO-SLAM/run_pipeline.py` 主入口。

## 11. 当前限制

- 当前主 SLAM 以 notebook 中已验证的纯 Python sliding-window VIO 为主，不再以旧 `src/vio_slam` 包为主
- `legacy_python_vio_slam/` 仍然保留，但不再是当前运行入口
- 如果 Mac 本地没有 EuRoC 数据，`run_pipeline.py` 只能先做参数验证，不能跑真实序列

## 12. EuRoC 回放 + 退化注入仿真

如果你想演示“正常回放 vs 退化回放”对主 VIO 和 self-aware 结果的影响，可以直接运行：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_euroc_degradation_demo.py \
  --data-path /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0 \
  --camera-degradation motion_blur \
  --imu-degradation bias_drift \
  --severity 0.6 \
  --output-root /Users/kailunwang/Desktop/ossa/outputs/euroc_degradation_demo
```

如果你只想先快速演示，可以加大 `downsample`：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_euroc_degradation_demo.py \
  --data-path /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0 \
  --camera-degradation motion_blur \
  --imu-degradation bias_drift \
  --severity 0.6 \
  --downsample 120 \
  --output-root /Users/kailunwang/Desktop/ossa/outputs/euroc_degradation_quick
```

可选退化类型：

- 相机：`motion_blur`、`gaussian_noise`、`brightness_change`、`image_dropout`
- IMU：`bias_drift`、`noise_amplification`

脚本会自动完成：

1. baseline EuRoC 回放
2. degraded EuRoC 回放
3. 两条结果分别做 offline self-aware inference
4. 输出对比摘要和图

关键输出：

- `baseline_vio/`
- `degraded_vio/`
- `baseline_self_aware/`
- `degraded_self_aware/`
- `comparison/comparison_summary.txt`
- `comparison/comparison_metrics.csv`
- `comparison/comparison_overview.png`
- `comparison/gui/visual_demo.html`

对比 GUI 页面会自动生成在：

```text
/Users/kailunwang/Desktop/ossa/outputs/euroc_degradation_quick/comparison/gui/visual_demo.html
```

这个页面支持：

- baseline / degraded 双轨迹同步查看
- failure probability 与 confidence 双路时间线对比
- tracking / error 指标对比
- 跳转到退化后风险抬升最明显的帧

## 13. 多序列 degradation sweep

如果你要一次性跑多条 EuRoC 序列，并在每条序列上复用 baseline、批量比较多个代表性退化场景与 severity 网格，直接运行：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_multisequence_degradation_sweep.py \
  --dataset-root /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences \
  --sequences MH_01_easy,MH_02_easy,MH_03_medium,MH_04_difficult,MH_05_difficult \
  --scenarios blur_bias,noise_amp,lighting_dropout,dropout_bias \
  --severity-grid 0.45,0.70 \
  --output-root /Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid
```

当前公开 benchmark 默认覆盖：

- 5 条序列：`MH_01_easy`、`MH_02_easy`、`MH_03_medium`、`MH_04_difficult`、`MH_05_difficult`
- 4 个复合场景：`blur_bias`、`noise_amp`、`lighting_dropout`、`dropout_bias`
- 2 个 severity：`0.45`、`0.70`

默认这 4 个场景分别代表：

- `blur_bias`: `motion_blur + bias_drift`
- `noise_amp`: `gaussian_noise + noise_amplification`
- `lighting_dropout`: `brightness_change`
- `dropout_bias`: `image_dropout + bias_drift`

脚本会自动完成：

1. 每条序列 baseline 只跑一次
2. 每个退化场景单独跑 degraded VIO
3. 每个 severity 自动展开成单独 run
4. 每个 run 分别做 offline self-aware inference
5. 自动生成每条序列 / 每个 run 的 comparison GUI
6. 自动汇总成跨序列总表、benchmark 表和总 GUI

关键输出：

- `outputs/multisequence_degradation_grid/sweep_results.csv`
- `outputs/multisequence_degradation_grid/report/multi_sequence_summary.txt`
- `outputs/multisequence_degradation_grid/report/scenario_aggregate.csv`
- `outputs/multisequence_degradation_grid/report/sequence_aggregate.csv`
- `outputs/multisequence_degradation_grid/report/benchmark_runs.csv`
- `outputs/multisequence_degradation_grid/report/benchmark_scenario_severity.csv`
- `outputs/multisequence_degradation_grid/report/benchmark_failure_delta_pivot.csv`
- `outputs/multisequence_degradation_grid/report/benchmark_failure_delta_pivot.md`
- `outputs/multisequence_degradation_grid/report/multi_sequence_overview.png`
- `outputs/multisequence_degradation_grid/report/visual_demo.html`

总 GUI 页面位置：

```text
/Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid/report/visual_demo.html
```

这个页面支持：

- 按 sequence / scenario / severity / camera / imu 过滤
- 查看每个 run 的 failure / confidence / pose error / inlier ratio 变化量
- 直接跳转到单条 run 的 comparison GUI
- 快速定位“哪个退化场景整体最伤系统”

当前这轮公开结果摘要：

- 覆盖 `5` 条 MH 序列、`4` 个 base 场景、`2` 个 severity，总计 `40` 组 run
- 平均 `failure delta` 最大场景：`blur_bias_s45`
- 平均 `pose error delta` 最大场景：`noise_amp_s45`
- 平均看，`MH_05` 对 `failure/confidence` 变化最敏感，`MH_04`、`MH_05` 的脆弱性高于前 3 条 easy/medium 序列

## 14. Model Validity Benchmark

如果你现在要回答的问题不是“模型会不会动”，而是“模型到底准不准”，就跑这一步。

这个 benchmark 直接建立在多序列 sweep 结果上，不需要重跑主 VIO，只会读取每个 run 的：

- `reliability_predictions.csv`
- `slam_metrics.csv`
- `actual_pose_error`

然后输出：

- `failure_probability` 和真实 `pose_error` 的 Pearson / Spearman 相关性
- `predicted_pose_error` 和真实 `pose_error` 的相关性
- 多个 failure threshold 下的 ROC-AUC / AP / F1
- 概率校准图
- 和简单 heuristic（如 `inlier_ratio`、`pose_optimization_residual`、`num_inliers`、`mean_epipolar_error`）的对比

运行命令：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_model_validity_benchmark.py \
  --sweep-results /Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid/sweep_results.csv \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid/model_validity \
  --failure-thresholds 0.3,1.0,3.0 \
  --summary-threshold 3.0
```

关键输出：

- `outputs/multisequence_degradation_grid/model_validity/validity_summary.txt`
- `outputs/multisequence_degradation_grid/model_validity/threshold_metrics.csv`
- `outputs/multisequence_degradation_grid/model_validity/run_level_correlations.csv`
- `outputs/multisequence_degradation_grid/model_validity/sequence_validity_summary.csv`
- `outputs/multisequence_degradation_grid/model_validity/scenario_validity_summary.csv`
- `outputs/multisequence_degradation_grid/model_validity/model_vs_actual_scatter.png`
- `outputs/multisequence_degradation_grid/model_validity/sequence_correlation_overview.png`
- `outputs/multisequence_degradation_grid/model_validity/roc_comparison_t1p0.png`
- `outputs/multisequence_degradation_grid/model_validity/roc_comparison_t3p0.png`
- `outputs/multisequence_degradation_grid/model_validity/calibration_t1p0.png`
- `outputs/multisequence_degradation_grid/model_validity/calibration_t3p0.png`

当前这轮 benchmark 的第一版结论：

- 全局 `failure_probability` vs `actual_pose_error` Pearson 只有约 `-0.010`
- 全局 `predicted_pose_error` vs `actual_pose_error` Pearson 只有约 `-0.028`
- 在 `3.0m` failure threshold 下，`model_failure_probability` 的 ROC-AUC 约 `0.511`
- 同一阈值下，最好 heuristic `heuristic_epipolar_error_risk` 的 ROC-AUC 约 `0.542`

这说明：

- 模型目前**对退化有反应**
- 但还**没有证明它稳定地预测了真实误差**

也就是说，现在更准确的结论是：

- “model reacts”
- 不是 “model is already valid”
