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

## 批量运行

如果你要一次跑多个序列：

```bash
bash /Users/kailunwang/Desktop/ossa/integration/run_batch_unified_pipeline.sh \
  --dataset-root /Users/kailunwang/Desktop/ossa/VIO-SLAM/data/sequences \
  --sequences MH_01_easy,MH_02_easy,MH_03_medium
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

## 当前限制

- 当前主 SLAM 是 notebook-derived 纯 Python VIO，不是完整 C++ ORB-SLAM3
- self-aware 模型已能接入，但还存在域偏移，结果可用于系统展示，不适合当成最终精度结论
- 当前主集成是离线式，不是在线每帧推理

## 当前建议

如果你的目标是“把系统做出来”，下一步优先做：

1. 跑通 `MH_02`、`MH_03`
2. 用批处理脚本积累更多 `*_unified` 序列
3. 只做最低限度模型校准，不深挖调参
4. 把项目作为完整 pipeline 展示，而不是只展示单个模型指标
