# Self-Aware VIO-SLAM 教学指南

这份文档不是简单的命令清单，而是带你**理解整条 unified pipeline 链路**。

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
