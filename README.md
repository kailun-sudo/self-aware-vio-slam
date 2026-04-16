# ORB-SLAM 与 Self-Awareness 统一项目

详细运行方式请直接看：

- [README_运行指南.md](/Users/kailunwang/Desktop/ossa/README_运行指南.md)
- [README_教学指南.md](/Users/kailunwang/Desktop/ossa/README_教学指南.md)
- [README_交付版.md](/Users/kailunwang/Desktop/ossa/README_交付版.md)

当前目录语义：

- `self_aware_slam/`：当前正版主目录，以 VM 上可跑通版本为基线整理
- `self_aware_slam_legacy/`：旧版本保留目录，只做对照和回溯，不再作为主运行入口

当前项目已经实现：

- `VIO-SLAM/run_pipeline.py` 导出 `slam_metrics.csv`
- 导出带时间戳的 `estimated_tum.txt`
- `self_aware_slam` 读取 metrics 做离线推理
- 输出 `confidence_score`、`failure_probability`、`predicted_localization_reliability`

如果你要快速开始，直接按运行指南里的“推荐的最小执行顺序”执行即可。
