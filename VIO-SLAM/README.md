# VIO-SLAM

This directory contains the main Python **visual-inertial SLAM runtime** used by the project.

It is the front end of the full self-aware pipeline described in the repository root `README.md`.

## Purpose

`VIO-SLAM` is responsible for:

- loading EuRoC-style `mav0` data,
- running the notebook-derived sliding-window VIO pipeline,
- exporting trajectory estimates,
- and exporting internal SLAM metrics for downstream self-awareness analysis.

The recommended runtime entrypoint is:

- `run_pipeline.py`

The main pipeline implementation lives in:

- `vio_pipeline.py`

## Directory Layout

```text
VIO-SLAM/
├── config/
│   └── default.yaml
├── reference/
│   └── README.md
├── requirements.txt
├── run_pipeline.py
└── vio_pipeline.py
```

## Recommended Usage

If your EuRoC sequence is stored at a custom location:

```bash
cd /path/to/ossa/VIO-SLAM
./.venv/bin/python run_pipeline.py \
  --data_path /path/to/EuRoC/MH_01_easy/mav0 \
  --output ../outputs/mh01
```

If your local setup uses the default repository layout, you can point to a sequence under `VIO-SLAM/data/sequences/...`.

## Main Outputs

The most important exported files are:

- `slam_metrics.csv`
- `estimated_tum.txt`
- `trajectory.txt`
- `trajectory.pkl`

These files are consumed downstream by:

- `integration/run_offline_unified_demo.py`
- the self-awareness inference stack in `self_aware_slam/`

## Role in the Full Project

At the repository level, the runtime stack is:

```text
EuRoC mav0
  -> VIO-SLAM/run_pipeline.py
  -> slam_metrics.csv + estimated_tum.txt
  -> integration/run_offline_unified_demo.py
  -> pose_errors.csv + reliability_predictions.csv
```

In other words:

- `VIO-SLAM/` produces motion estimates and internal SLAM signals
- `self_aware_slam/` turns those signals into runtime reliability outputs
- `integration/` provides replay, benchmarking, and visualization scripts

## Notes

- This is a **research runtime**, not a production C++ SLAM system.
- The public repository keeps the runnable extracted pipeline here and avoids shipping large local datasets or private experimental artifacts.
