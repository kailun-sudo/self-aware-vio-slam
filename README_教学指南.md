# Self-Aware VIO-SLAM 教学指南

这份文档不是简单的命令清单，而是带你**理解整条 unified pipeline 链路**。

如果你现在想单独把 self-aware 预测模型看透，直接配合这份文档一起看：

- `README_自感知模型详解.md`

你现在手上的系统已经不是两个分离项目，而是一条完整的数据流：

1. `VIO-SLAM` 读取 EuRoC `mav0`
2. 运行 notebook 提炼出来的纯 Python VIO 主线
3. 导出 `slam_metrics.csv` 和 `estimated_tum.txt`
4. `self_aware_slam` 读取这些输出
5. 结合 ground truth 生成 `pose_errors.csv`
6. 运行 reliability / failure prediction
7. 输出 `confidence_score`、`failure_probability`
8. 进一步做结果分析和训练集打包

---

## 1. 先理解整条链路

可以把它看成 4 层：

### 第 1 层：主 SLAM 层

主入口：

```text
/Users/kailunwang/Desktop/ossa/VIO-SLAM/run_pipeline.py
```

它做的事情：

- 读取 EuRoC 的 `mav0` 数据
- 按 notebook 验证过的 sliding-window VIO 逻辑跑完整序列
- 估计一条轨迹
- 同时导出给 self-aware 用的内部指标

这一层的核心输出是：

- `slam_metrics.csv`
- `estimated_tum.txt`

### 第 2 层：对齐与误差层

主入口：

```text
/Users/kailunwang/Desktop/ossa/integration/run_offline_unified_demo.py
```

它做的事情：

- 读取 `slam_metrics.csv`
- 读取 `estimated_tum.txt`
- 读取 ground truth
- 做时间戳对齐
- 做轨迹对齐
- 计算每帧 `pose_error`

这一层新增的关键输出是：

- `pose_errors.csv`

### 第 3 层：self-aware 推理层

主入口内部调用的是：

```text
/Users/kailunwang/Desktop/ossa/self_aware_slam/src/models/inference.py
```

它做的事情：

- 读取 `slam_metrics.csv`
- 把字段映射成模型需要的 canonical features
- 做 normalization
- 构建时间窗口
- 调用已训练模型
- 输出故障预测和可靠性评分

这一层的关键输出是：

- `reliability_predictions.csv`

### 第 4 层：分析和训练准备层

两个补充入口：

- 结果分析：
  `/Users/kailunwang/Desktop/ossa/integration/analyze_unified_results.py`
- 训练打包：
  `/Users/kailunwang/Desktop/ossa/self_aware_slam/scripts/package_unified_sequence.py`

它们做的事情：

- 把预测结果画图
- 计算预测和真实误差的关系
- 把这次跑出来的数据整理成可继续训练的序列目录

---

## 2. 你现在工作区里的关键目录

```text
/Users/kailunwang/Desktop/ossa
├── VIO-SLAM
├── self_aware_slam
├── self_aware_slam_legacy
├── integration
├── outputs
├── README_运行指南.md
└── README_教学指南.md
```

重点理解这几个：

- `VIO-SLAM/`
  当前主 SLAM 系统

- `VIO-SLAM/vio_pipeline.py`
  从原始 notebook 提炼出来的主算法逻辑

- `VIO-SLAM/run_pipeline.py`
  当前唯一推荐运行入口

- `VIO-SLAM/data/mav0`
  当前主 SLAM 默认读取的数据目录

- `self_aware_slam/`
  当前正版 self-aware 项目

- `integration/`
  放统一项目的桥接脚本

- `outputs/`
  统一项目跑出来的结果

---

## 3. 整条链路的“输入”和“输出”

### 输入

最原始输入其实只有两类：

- EuRoC `mav0` 数据
- 已训练好的 self-aware 模型和归一化统计

你现在已经有：

- `VIO-SLAM/data/mav0`
- `self_aware_slam/results/models/...`
- `self_aware_slam/results/train_dataset.pkl`
- `self_aware_slam/results/train_dataset_v2.pkl`

### 中间产物

跑完主 SLAM 后，会有：

- `outputs/mh01/slam_metrics.csv`
- `outputs/mh01/estimated_tum.txt`
- `outputs/mh01/trajectory.txt`
- `outputs/mh01/trajectory.pkl`

跑完 unified demo 后，会有：

- `outputs/mh01_self_aware/pose_errors.csv`
- `outputs/mh01_self_aware/reliability_predictions.csv`
- `outputs/mh01_self_aware/summary.txt`

跑完分析后，会有：

- `outputs/mh01_analysis/analysis_summary.txt`
- `outputs/mh01_analysis/scores_over_time.png`
- `outputs/mh01_analysis/pose_error_comparison.png`

跑完训练打包后，会有：

- `self_aware_slam/slam_metrics_dataset/MH_01_unified/slam_metrics.csv`
- `self_aware_slam/slam_metrics_dataset/MH_01_unified/pose_errors.csv`
- `self_aware_slam/slam_metrics_dataset/MH_01_unified/groundtruth.csv`
- `self_aware_slam/slam_metrics_dataset/MH_01_unified/estimated.csv`

如果你走新的 self-aware 训练线，还会有：

- `self_aware_slam/results/train_dataset_v2.pkl`

这份 v2 数据集和旧训练缓存的核心区别是：

- 旧版更偏“当前诊断量 + 混合标签”
- v2 明确统一成：
  - 输入：22 维 trend-aware learning features
  - 回归目标：未来 10 帧内的 `future_max_pose_error`
  - 分类目标：`future_max_pose_error > 0.18m` 或未来 tracking lost
  - 数据源：长 baseline 序列 + degraded replay runs
  - split 方式：run-level split，而不是只按整条 sequence 切

---

## 4. 手把手跑一遍主链路

下面这部分就是“你自己重新跑一遍时”的标准动作。

### Step 1：跑主 SLAM

命令：

```bash
cd /Users/kailunwang/Desktop/ossa/VIO-SLAM
./.venv/bin/python run_pipeline.py \
  --output /Users/kailunwang/Desktop/ossa/outputs/mh01
```

这一步输入：

- `VIO-SLAM/data/mav0`

这一步内部做了什么：

- 读取 `cam0` 图像和 `imu0/data.csv`
- 做 IMU 预积分
- 做 ORB 特征匹配
- 做 sliding-window 优化
- 生成轨迹
- 记录窗口级 metrics

这一步输出：

- `outputs/mh01/slam_metrics.csv`
- `outputs/mh01/estimated_tum.txt`
- `outputs/mh01/trajectory.txt`
- `outputs/mh01/trajectory.pkl`
- `outputs/mh01/summary.txt`

你怎么判断它成功了：

- 终端里会看到 `Pipeline finished`
- `outputs/mh01/slam_metrics.csv` 存在
- `outputs/mh01/estimated_tum.txt` 存在

### Step 1.5：构建 v2 学习数据集

如果你现在的目标是“把 self-aware 训练问题重新定义正确”，就执行：

```bash
cd /Users/kailunwang/Desktop/ossa/self_aware_slam
./venv/bin/python -m src.data.dataset_builder
```

这一步不是做推理，而是把训练任务收敛成一个更可学的问题。

当前这版 v2 的 sanity check 结果是：

- `source_mode = hybrid`
- `window_size = 10`
- `feature_dim = 22`
- `train failure rate ≈ 9.5%`
- `val failure rate ≈ 20.5%`
- `test failure rate ≈ 20.6%`
- split run counts:
  - `train = 23`
  - `val = 11`
  - `test = 11`

这里最关键的变化不是“多了特征”，而是：

- train / val / test 现在用了同一种 target 定义
- classification 和 regression 现在都围绕同一个 future target
- 不再是 train 学 current failure、eval 看 predictive failure 那种错位设置
- train 现在也真正看到了 degraded replay 里的 failure 模式

### Step 2：跑 unified demo

命令：

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

这一步输入：

- 主 SLAM 导出的 `slam_metrics.csv`
- 主 SLAM 导出的 `estimated_tum.txt`
- EuRoC ground truth

这一步内部做了什么：

- 对齐估计轨迹和 ground truth
- 计算真实 `pose_error`
- 运行 self-aware 推理
- 把真实误差并回预测结果

这一步输出：

- `outputs/mh01_self_aware/pose_errors.csv`
- `outputs/mh01_self_aware/reliability_predictions.csv`
- `outputs/mh01_self_aware/summary.txt`

你怎么判断它成功了：

- `summary.txt` 出现
- `reliability_predictions.csv` 里有：
  - `failure_probability`
  - `confidence_score`
  - `actual_pose_error`

### Step 3：看结果

命令：

```bash
cat /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/summary.txt
head -n 20 /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/reliability_predictions.csv
```

你现在这次跑出来的核心结果是：

- `matched_poses: 364`
- `pose_error_mean: 3.533206`
- `confidence_mean: 0.217112`
- `failure_probability_mean: 0.782888`

这表示什么：

- 主链路已经打通
- 模型已经能对当前 `VIO-SLAM` 输出做推理
- 但当前这版模型对这条序列整体判断偏“不可靠”

---

## 5. 每个关键文件到底代表什么

### `slam_metrics.csv`

这是“主 SLAM 运行过程中的内部状态记录”。

它不是轨迹，也不是 ground truth，而是给 self-aware 模型看的“体征数据”。

典型列包括：

- `num_keypoints`
- `num_matches`
- `num_inliers`
- `inlier_ratio`
- `tracking_state`
- `mean_epipolar_error`
- `pose_optimization_residual`
- `trajectory_increment_norm`

理解方式：

- 它回答的是：`这一帧或这一窗口，SLAM 当前看起来稳不稳？`

### `estimated_tum.txt`

这是主 SLAM 估计出来的轨迹。

理解方式：

- 它回答的是：`SLAM 认为自己走到了哪里？`

---

## 6. EuRoC 回放 + 退化注入仿真怎么理解

现在这套系统除了正常 EuRoC 回放，还支持一个非常适合展示和实验的模式：

- 同一条 EuRoC `mav0`
- 一次跑 baseline
- 一次跑 degraded
- 然后把两条结果一起送进 self-aware 模块做对比

这条链的主入口是：

```text
/Users/kailunwang/Desktop/ossa/integration/run_euroc_degradation_demo.py
```

它内部会做 5 件事：

1. 跑 baseline `VIO-SLAM/run_pipeline.py`
2. 跑 degraded `VIO-SLAM/run_pipeline.py`
3. 对两条结果分别跑 `run_offline_unified_demo.py`
4. 输出比较摘要和图
5. 自动生成 baseline vs degraded 对比 GUI

### 退化到底注入到哪里

不是只在 ML 层伪造数值，而是直接在主 VIO 回放时注入退化：

- 图像退化：
  - `motion_blur`
  - `gaussian_noise`
  - `brightness_change`
  - `image_dropout`
- IMU 退化：
  - `bias_drift`
  - `noise_amplification`

也就是说，你现在可以讲得更准确一点：

**这不是纯离线表格增强，而是作用在 EuRoC 回放执行过程中的 sensor degradation simulation。**

### 为什么这一步重要

因为它把你的项目从“我能预测风险”推进到了：

- 我能主动构造退化场景
- 我能比较退化前后系统状态变化
- 我能观察 self-aware 输出是否随系统退化同步变化

这在面试、汇报和后续实验设计里都很有用。

### 这一步会产出什么

如果你跑：

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

你会同时得到：

- `baseline_vio/`
- `degraded_vio/`
- `baseline_self_aware/`
- `degraded_self_aware/`
- `comparison/comparison_summary.txt`
- `comparison/comparison_metrics.csv`
- `comparison/comparison_overview.png`
- `comparison/gui/visual_demo.html`

### 这个 comparison GUI 是干什么的

它不是重复单路 GUI，而是专门回答：

- 退化后轨迹有没有明显变化
- failure probability 有没有抬高
- confidence 有没有降低
- 哪些帧的风险增量最大
- tracking 质量指标有没有同步恶化

所以它更像一个：

**stress test dashboard**

而不是普通 demo 页面。

### 再往前一步：为什么要做 multi-sequence sweep

单条序列的 comparison GUI 解决的是：

- 某一种退化会不会让系统更危险

但如果你要更像实验，而不是只像 demo，还需要回答：

- 这个现象是不是只在 `MH_01` 上成立
- 不同退化类型谁更伤系统
- 哪种退化更容易让 failure probability 抬高
- 哪种退化更容易让 pose error 变大

所以我又加了一条更高层的主线：

```text
/Users/kailunwang/Desktop/ossa/integration/run_multisequence_degradation_sweep.py
```

它做的事是：

1. 选多条 EuRoC 序列
2. 每条序列 baseline 只跑一次
3. 对每条序列跑多个代表性退化场景
4. 为每个 run 自动生成 comparison GUI
5. 最后再汇总成一个跨序列总表和总 GUI

这样你得到的就不是：

- “一个退化 demo”

而是：

- “一个小型 stress-test experiment package”

### 这条多序列 sweep 现在默认做什么

默认序列：

- `MH_01_easy`
- `MH_02_easy`
- `MH_03_medium`
- `MH_04_difficult`
- `MH_05_difficult`

默认代表性退化：

- `blur_bias`
  - `motion_blur + bias_drift`
- `noise_amp`
  - `gaussian_noise + noise_amplification`
- `lighting_dropout`
  - `brightness_change`
- `dropout_bias`
  - `image_dropout + bias_drift`

默认 severity 网格：

- `0.45`
- `0.70`

### 最后会产出什么

除了每条 run 自己的 comparison GUI，还会有一个总报告目录：

```text
/Users/kailunwang/Desktop/ossa/outputs/multisequence_degradation_grid/report
```

里面有：

- `sweep_results.csv`
- `multi_sequence_summary.txt`
- `scenario_aggregate.csv`
- `sequence_aggregate.csv`
- `benchmark_runs.csv`
- `benchmark_scenario_severity.csv`
- `benchmark_failure_delta_pivot.csv`
- `benchmark_failure_delta_pivot.md`
- `multi_sequence_overview.png`
- `visual_demo.html`

这个总 GUI 更像一个：

**cross-sequence experiment dashboard**

它回答的是：

- 哪个退化场景平均最危险
- 哪个序列最脆弱
- 哪个 run 的 failure delta 最大
- 哪些单条 comparison GUI 最值得单独点进去看

### 这轮公开 benchmark 现在做到哪一步

目前已经实际跑完：

- 5 条 Machine Hall 序列
- 4 个复合退化场景
- 2 个 severity
- 共 `40` 组 degraded run

所以它不再只是：

- “我拿 3 条序列试了一下”

而是已经更接近：

- “我有一套稳定的跨序列 stress-test benchmark”

这也意味着你后面要继续扩展时，最自然的下一步不是再重复 MH，而是补：

- `V1_*`
- `V2_*`

把 benchmark 从 `Machine Hall only` 扩到 `cross-room / cross-motion regime`。

### 但这里有一个非常关键的区分

做到 multi-sequence sweep 以后，你已经能证明：

- 模型对退化有反应
- 风险分数、confidence 和 tracking 指标会跟着变化

但这**还不等于**证明模型真的预测对了。

因为一个模型可能：

- 对退化很敏感
- 输出变化很明显

但它和真实 `pose_error` 的关系仍然不稳定。

这就是为什么我又补了这条专门的 benchmark：

```text
/Users/kailunwang/Desktop/ossa/integration/run_model_validity_benchmark.py
```

它要回答的问题不是：

- “模型有没有动？”

而是：

- “模型的输出到底和真实误差有没有稳定对应关系？”
- “它比简单 heuristic 强不强？”
- “这个 failure probability 有没有概率意义？”

### 这条 validity benchmark 会看什么

它主要看 4 类证据：

1. **相关性**
   - `failure_probability` vs `actual_pose_error`
   - `predicted_pose_error` vs `actual_pose_error`

2. **二分类有效性**
   - 把真实 `pose_error` 过阈值当 failure label
   - 看 ROC-AUC / AP / F1

3. **校准**
   - 看 `failure_probability` 和真实失败率是否对得上

4. **heuristic 对比**
   - 和 `inlier_ratio`
   - `pose_optimization_residual`
   - `num_inliers`
   - `mean_epipolar_error`
   做对比

### 当前第一版 validity benchmark 的结论

目前这套系统的结论是：

- 模型**有反应**
- 但**还没有被证明是对的**

原因很直接：

- 全局 `failure_probability` 和 `actual_pose_error` 的相关性接近 `0`
- 全局 `predicted_pose_error` 和 `actual_pose_error` 的相关性也接近 `0`
- 在 `3.0m` failure threshold 下，模型的 ROC-AUC 只有大约 `0.51`
- 最好的简单 heuristic 反而能到大约 `0.54`

所以这一步很重要，因为它把结论从：

- “系统做出来了”

推进到：

- “我们现在知道模型哪里还没站住”

### `pose_errors.csv`

这是把 `estimated_tum.txt` 和 ground truth 对齐后得到的真实误差。

理解方式：

- 它回答的是：`SLAM 实际错了多少？`

### `reliability_predictions.csv`

这是 self-aware 模型的最终输出。

关键列：

- `failure_probability`
- `confidence_score`
- `predicted_pose_error`
- `predicted_localization_reliability`
- `actual_pose_error`

理解方式：

- 它回答的是：`模型觉得这一刻会不会失败？`
- 同时还能和真实误差对照看：`它判断得准不准？`

---

## 6. 为什么现在 failure probability 偏高

你现在看到很多行都是：

- `failure_probability ~ 0.78`
- `confidence_score ~ 0.22`
- `predicted_failure = 1`

这不代表 pipeline 坏了。

更合理的解释是：

- 这版 notebook-derived VIO 在这条序列上的真实误差确实不小
- 当前 self-aware 模型原本并不是专门按你这版 `VIO-SLAM` 的指标分布训练的
- 所以推理是能跑，但存在**域偏移**

怎么判断是不是域偏移：

- 看 `predicted_pose_error` 是否变化很小
- 看 `failure_probability` 与 `actual_pose_error` 的相关性是否很高

你这次分析结果里：

- `failure_vs_actual_corr ≈ 0.24`

说明它有一定关系，但还不够强。

---

## 7. 如何做结果分析

命令：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/integration/analyze_unified_results.py \
  --predictions /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/reliability_predictions.csv \
  --summary /Users/kailunwang/Desktop/ossa/outputs/mh01_self_aware/summary.txt \
  --output-dir /Users/kailunwang/Desktop/ossa/outputs/mh01_analysis
```

会得到：

- `analysis_summary.txt`
- `scores_over_time.png`
- `pose_error_comparison.png`

你应该怎么读这两张图：

- `scores_over_time.png`
  看模型是不是整段都在报高风险，还是只有局部高风险

- `pose_error_comparison.png`
  看 `failure_probability` 和 `actual_pose_error` 是否同步变化

---

## 8. 如何把这次结果沉淀成训练数据

命令：

```bash
/Users/kailunwang/Desktop/ossa/self_aware_slam/venv/bin/python \
  /Users/kailunwang/Desktop/ossa/self_aware_slam/scripts/package_unified_sequence.py \
  --metrics /Users/kailunwang/Desktop/ossa/outputs/mh01/slam_metrics.csv \
  --estimated /Users/kailunwang/Desktop/ossa/outputs/mh01/estimated_tum.txt \
  --groundtruth /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/mav0/state_groundtruth_estimate0/data.csv \
  --sequence-name MH_01_unified \
  --dataset-root /Users/kailunwang/Desktop/ossa/self_aware_slam/slam_metrics_dataset
```

这一步的意义非常重要：

- 你不是只跑一次 demo
- 你是在把当前这版 `VIO-SLAM` 输出，变成 self-aware 可继续训练的数据资产

输出目录：

```text
/Users/kailunwang/Desktop/ossa/self_aware_slam/slam_metrics_dataset/MH_01_unified
```

这就是以后继续扩充训练集时要累积的格式。

---

## 9. 如果你以后要继续训练

你的长期目标应该是：

1. 多跑几个 EuRoC 序列
2. 每个序列都生成：
   - `slam_metrics.csv`
   - `pose_errors.csv`
   - `groundtruth.csv`
   - `estimated.csv`
3. 都放进：
   - `self_aware_slam/slam_metrics_dataset/<sequence_name>/`
4. 然后重建训练集，再重新训练模型

也就是说，真正的闭环是：

`VIO-SLAM -> unified demo -> sequence packaging -> dataset accumulation -> retraining -> better self-awareness`

---

## 10. 你现在已经完成了什么

截至目前，这个 unified project 已经具备：

- 一个从 notebook 提炼出的纯 Python 主 SLAM 入口
- 主 SLAM metrics 导出
- ground truth 对齐
- self-aware 推理
- 结果分析
- 训练数据打包

换句话说，你现在不是只有“一个 demo”。

你已经有了一条可以持续迭代的工程链路。

---

## 11. 最后给你一个最短记忆版

如果你只想记住最少的 4 条命令，记这个：

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

### 3. 做分析

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

---

## 12. 你下一步最值得做什么

我给你的建议是按这个顺序推进：

1. 再跑 `MH_02`、`MH_03`
2. 每条序列都打包成 `*_unified`
3. 看分析图，确认不同序列的 failure pattern
4. 开始做一版专门针对当前 `VIO-SLAM` 指标分布的再训练

如果你愿意，下一步我可以继续直接帮你写：

- 一个“批量跑多个 EuRoC 序列”的脚本
- 一个“批量打包训练集并触发重训练”的脚本
