#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_DEFAULT="${HOME}/workspace/ossa"
OUTPUT_NAME_DEFAULT="mh01"

usage() {
  cat <<'EOF'
用法:
  bash integration/run_linux_vm_pipeline.sh --dataset /path/to/mav0 [可选参数]

必填参数:
  --dataset PATH             EuRoC mav0 数据目录

可选参数:
  --workspace PATH           工程根目录，默认: ~/workspace/ossa
  --groundtruth PATH         Ground truth 文件路径，默认自动取 <dataset>/state_groundtruth_estimate0/data.csv
  --output-name NAME         输出目录名，默认: mh01
  --skip-setup               跳过虚拟环境创建和依赖安装
  --skip-slam                跳过 VIO-SLAM，仅运行 unified demo
  --skip-demo                跳过 unified demo，仅运行 VIO-SLAM
  --vio-python PATH          指定 VIO-SLAM 的 Python 可执行文件
  --aware-python PATH        指定 self-aware 的 Python 可执行文件

示例:
  bash integration/run_linux_vm_pipeline.sh \
    --workspace ~/workspace/ossa \
    --dataset /data/EuRoC/MH_01_easy/mav0 \
    --output-name mh01
EOF
}

WORKSPACE="${WORKSPACE_DEFAULT}"
DATASET_PATH=""
GROUNDTRUTH_PATH=""
OUTPUT_NAME="${OUTPUT_NAME_DEFAULT}"
SKIP_SETUP=0
SKIP_SLAM=0
SKIP_DEMO=0
VIO_PYTHON=""
AWARE_PYTHON=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --groundtruth)
      GROUNDTRUTH_PATH="$2"
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --skip-setup)
      SKIP_SETUP=1
      shift
      ;;
    --skip-slam)
      SKIP_SLAM=1
      shift
      ;;
    --skip-demo)
      SKIP_DEMO=1
      shift
      ;;
    --vio-python)
      VIO_PYTHON="$2"
      shift 2
      ;;
    --aware-python)
      AWARE_PYTHON="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET_PATH}" ]]; then
  echo "错误: 必须提供 --dataset" >&2
  usage
  exit 1
fi

if [[ -z "${GROUNDTRUTH_PATH}" ]]; then
  GROUNDTRUTH_PATH="${DATASET_PATH}/state_groundtruth_estimate0/data.csv"
fi

VIO_ROOT="${WORKSPACE}/VIO-SLAM"
AWARE_ROOT="${WORKSPACE}/self_aware_slam"
INTEGRATION_ROOT="${WORKSPACE}/integration"
OUTPUT_ROOT="${WORKSPACE}/outputs/${OUTPUT_NAME}"
DEMO_OUTPUT_ROOT="${WORKSPACE}/outputs/${OUTPUT_NAME}_self_aware"

if [[ ! -d "${VIO_ROOT}" ]]; then
  echo "错误: 找不到 VIO-SLAM 目录: ${VIO_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${AWARE_ROOT}" ]]; then
  echo "错误: 找不到 self_aware_slam 目录: ${AWARE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "错误: 找不到数据集目录: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ ! -f "${GROUNDTRUTH_PATH}" ]]; then
  echo "错误: 找不到 ground truth 文件: ${GROUNDTRUTH_PATH}" >&2
  exit 1
fi

if [[ -z "${VIO_PYTHON}" ]]; then
  VIO_PYTHON="${VIO_ROOT}/.venv/bin/python"
fi

if [[ -z "${AWARE_PYTHON}" ]]; then
  AWARE_PYTHON="${AWARE_ROOT}/venv/bin/python"
fi

setup_vio_env() {
  if [[ ! -x "${VIO_PYTHON}" ]]; then
    python3 -m venv "${VIO_ROOT}/.venv"
  fi

  "${VIO_PYTHON}" -m pip install --upgrade pip setuptools wheel

  if [[ -f "${VIO_ROOT}/requirements.txt" ]]; then
    "${VIO_PYTHON}" -m pip install -r "${VIO_ROOT}/requirements.txt"
  else
    "${VIO_PYTHON}" -m pip install opencv-python numpy scipy matplotlib tqdm pyyaml
  fi
}

setup_aware_env() {
  if [[ ! -x "${AWARE_PYTHON}" ]]; then
    python3 -m venv "${AWARE_ROOT}/venv"
  fi

  "${AWARE_PYTHON}" -m pip install --upgrade pip setuptools wheel
  "${AWARE_PYTHON}" -m pip install -r "${AWARE_ROOT}/requirements.txt"
}

run_slam() {
  mkdir -p "${OUTPUT_ROOT}"

  (
    cd "${VIO_ROOT}"
    "${VIO_PYTHON}" run_pipeline.py \
      --data_path "${DATASET_PATH}" \
      --output "${OUTPUT_ROOT}" \
      --config "${VIO_ROOT}/config/default.yaml"
  )
}

run_demo() {
  "${AWARE_PYTHON}" "${INTEGRATION_ROOT}/run_offline_unified_demo.py" \
    --metrics "${OUTPUT_ROOT}/slam_metrics.csv" \
    --estimated "${OUTPUT_ROOT}/estimated_tum.txt" \
    --groundtruth "${GROUNDTRUTH_PATH}" \
    --output-dir "${DEMO_OUTPUT_ROOT}" \
    --config "${AWARE_ROOT}/configs/config.yaml"
}

echo "== Unified Offline Pipeline =="
echo "workspace:    ${WORKSPACE}"
echo "dataset:      ${DATASET_PATH}"
echo "groundtruth:  ${GROUNDTRUTH_PATH}"
echo "slam output:  ${OUTPUT_ROOT}"
echo "demo output:  ${DEMO_OUTPUT_ROOT}"

if [[ "${SKIP_SETUP}" -eq 0 ]]; then
  echo
  echo "== [1/3] 配置环境 =="
  setup_vio_env
  setup_aware_env
fi

if [[ "${SKIP_SLAM}" -eq 0 ]]; then
  echo
  echo "== [2/3] 运行 VIO-SLAM =="
  run_slam
fi

if [[ "${SKIP_DEMO}" -eq 0 ]]; then
  echo
  echo "== [3/3] 运行 unified demo =="
  run_demo
fi

echo
echo "完成。关键输出如下:"
echo "  ${OUTPUT_ROOT}/slam_metrics.csv"
echo "  ${OUTPUT_ROOT}/estimated_tum.txt"
echo "  ${DEMO_OUTPUT_ROOT}/pose_errors.csv"
echo "  ${DEMO_OUTPUT_ROOT}/reliability_predictions.csv"
echo "  ${DEMO_OUTPUT_ROOT}/summary.txt"
