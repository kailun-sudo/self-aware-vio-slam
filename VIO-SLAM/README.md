# VIO-SLAM

这是当前统一项目里的主 SLAM 系统。

## 当前主线

当前 `VIO-SLAM` 已经切换为一条清晰主线：

- `run_pipeline.py`：唯一推荐的运行入口
- `vio_pipeline.py`：从已验证 notebook 提炼出的纯 Python 主 pipeline
- `config/default.yaml`：当前主配置

这条主线直接完成：

- 读取 EuRoC `mav0` 数据
- 运行 notebook 版本的 sliding-window VIO
- 导出 `slam_metrics.csv`
- 导出 `estimated_tum.txt`
- 导出 `trajectory.txt` / `trajectory.pkl`

## 目录说明

```text
VIO-SLAM/
├── config/
│   └── default.yaml
├── legacy_python_vio_slam/
├── reference/
│   └── slam_reference.ipynb
├── requirements.txt
├── run_pipeline.py
└── vio_pipeline.py
```

- `legacy_python_vio_slam/`：原来的打包版 Python 实现，已经降级为 legacy
- `reference/`：保留原始 notebook，仅作来源参考，不再作为运行入口

## 运行

如果数据放在默认位置：

```text
VIO-SLAM/data/mav0
```

直接执行：

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

输出重点看：

- `slam_metrics.csv`
- `estimated_tum.txt`

这两个文件会被 `self_aware_slam` 和 `integration/run_offline_unified_demo.py` 继续使用。
