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

### 4.3 当前 self-aware 有两条线

- **runtime 推理线**
  - 继续使用当前已训练 checkpoint
  - 输入还是现有 `7` 维 runtime features
  - 不影响 online/offline demo
- **v2 训练线**
  - 使用统一的 future target 定义
  - 训练特征扩展为 `22` 维 trend-aware learning features
  - 数据集输出路径：
    `/Users/kailunwang/Desktop/ossa/self_aware_slam/results/train_dataset_v2.pkl`

当前 v2 训练任务定义是：

- regression target:
  `future_max_pose_error` over next `10` frames
- classification target:
  `future_max_pose_error > 0.18m` or future tracking lost

也就是说，v2 现在解决的是：

```text
看最近 10 帧内部状态 -> 预测接下来 10 帧内最坏会坏到什么程度
```

当前 v2 数据构建默认是 `source_mode = auto`，选择优先级是：

1. 如果 `slam_metrics_dataset/MH_*` 和 `outputs/multisequence_degradation_grid/` 都存在：
   使用 **hybrid**
2. 如果只有 replay runs：
   使用 **sweep_runs**
3. 如果只有长 baseline 序列：
   回退到 **sequence_dirs**

也就是说，当前默认会优先构建：

- 长 baseline 序列：`self_aware_slam/slam_metrics_dataset/MH_*`
- degraded replay runs：`outputs/multisequence_degradation_grid/`

然后做一个 **hybrid source**。

其中 degraded replay runs 不再只是普通 run-level shuffle，而是按
`(sequence, base_scenario)` 作为 **replay family** 做 split，保证同一 replay family 不跨 train / val / test。

默认 `split_protocol = family_aware_dev`，它适合：

- 先验证问题是否可学
- 控制 replay family 泄漏
- 保留更多训练样本

如果你要做严格的跨 sequence 泛化 benchmark，可以显式改成：

```bash
cd /Users/kailunwang/Desktop/ossa/self_aware_slam
./venv/bin/python -m src.data.dataset_builder \
  --split-protocol sequence_held_out \
  --output-path results/train_dataset_v2_sequence_held_out.pkl
```

这会保证同一个 sequence 的 baseline + degraded runs 全部进入同一个 split。

注意：基于当前公开的 `multisequence_degradation_grid` 结果，`MH_04_difficult` 和 `MH_05_difficult` 的 degraded runs 由于 downsample 后长度太短，很多会在 `window_size=10` 和 `prediction_horizon=10` 下被跳过。

所以当前 `sequence_held_out` 更准确地说是：

- 结构上严格避免 sequence leakage
- 但 `val/test` 里可用的 degraded 样本还偏少

如果你要做更强的 strict benchmark，下一步应当重跑 `MH_04/MH_05` 的 degradation replay，并降低 downsample。

builder 现在会显式打印：

- `data source selected`
- `replay runs found`
- `fallback activated`

这样可以直接看出当前到底在用 hybrid / sweep_runs / sequence_dirs，而不是 silent fallback。

## 4.4 一键下载 EuRoC 单序列

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

## 5.2 构建 v2 训练数据集

如果你要走新的 learnable training path，执行：

```bash
cd /Users/kailunwang/Desktop/ossa/self_aware_slam
./venv/bin/python -m src.data.dataset_builder
```

这一步会生成：

```text
/Users/kailunwang/Desktop/ossa/self_aware_slam/results/train_dataset_v2.pkl
```

当前这份数据集已经验证过，核心元数据是：

- `source_mode = hybrid`
- `window_size = 10`
- `feature_dim = 22`
- `target.mode = future_window_max`
- `prediction_horizon = 10`
- `classification_error_threshold = 0.18`

当前一版 sanity check 结果大致是：

- `train failure rate ≈ 9.5%`
- `val failure rate ≈ 20.5%`
- `test failure rate ≈ 20.6%`
- `train y_error range ≈ [0.116, 9.079]`
- `val y_error range ≈ [0.115, 9.994]`
- `test y_error range ≈ [0.116, 9.063]`
- split run counts:
  - `train = 23`
  - `val = 11`
  - `test = 11`

这一步的作用不是跑 demo，而是为后续重新训练一个更合理的 self-aware predictor 做准备。

### 5.3 用新 checkpoint 跑推理 / benchmark

如果你训练出了新的 v2 checkpoint，不需要替换仓库默认 runtime 模型，也可以直接通过参数显式指定：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/run_offline_unified_demo.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --groundtruth /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0/state_groundtruth_estimate0/data.csv \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware_v2 \
  --config /Users/kailunwang/Desktop/ossa/self_aware_slam/configs/config.yaml \
  --checkpoint /Users/kailunwang/Desktop/ossa/self_aware_slam/results/models/transformer_failure_predictor.pt \
  --dataset-stats /Users/kailunwang/Desktop/ossa/self_aware_slam/results/train_dataset_v2.pkl
```

同样，多序列 degradation sweep 也支持这两个参数，所以你可以用新模型重跑完整 benchmark，而不是继续评估旧 checkpoint。

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
