#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_DEFAULT="${HOME}/workspace/ossa"

usage() {
  cat <<'EOF'
用法:
  bash integration/check_env.sh --dataset /path/to/mav0 [可选参数]

必填参数:
  --dataset PATH             EuRoC mav0 数据目录

可选参数:
  --workspace PATH           工程根目录，默认: ~/workspace/ossa
  --groundtruth PATH         Ground truth 文件路径，默认自动取 <dataset>/state_groundtruth_estimate0/data.csv
  --vio-python PATH          指定 VIO-SLAM Python 可执行文件
  --aware-python PATH        指定 self-aware Python 可执行文件

示例:
  bash integration/check_env.sh \
    --workspace ~/workspace/ossa \
    --dataset /data/EuRoC/MH_01_easy/mav0
EOF
}

WORKSPACE="${WORKSPACE_DEFAULT}"
DATASET_PATH=""
GROUNDTRUTH_PATH=""
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

if [[ -z "${VIO_PYTHON}" ]]; then
  VIO_PYTHON="${VIO_ROOT}/.venv/bin/python"
fi

if [[ -z "${AWARE_PYTHON}" ]]; then
  AWARE_PYTHON="${AWARE_ROOT}/venv/bin/python"
fi

print_check() {
  local label="$1"
  local result="$2"
  printf "%-36s %s\n" "${label}" "${result}"
}

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    print_check "${label}" "OK    ${path}"
  else
    print_check "${label}" "MISS  ${path}"
  fi
}

check_python_imports() {
  local label="$1"
  local python_bin="$2"
  local modules_csv="$3"

  if [[ ! -x "${python_bin}" ]]; then
    print_check "${label}" "MISS  ${python_bin}"
    return
  fi

  local output
  if output="$("${python_bin}" - <<PY
import importlib
modules = "${modules_csv}".split(",")
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    print("MISSING:" + ",".join(missing))
else:
    print("OK")
PY
)"; then
    if [[ "${output}" == OK ]]; then
      print_check "${label}" "OK    ${python_bin}"
    else
      print_check "${label}" "${output}"
    fi
  else
    print_check "${label}" "FAIL  ${python_bin}"
  fi
}

echo "== Linux VM 环境检查 =="
echo "workspace:    ${WORKSPACE}"
echo "dataset:      ${DATASET_PATH}"
echo "groundtruth:  ${GROUNDTRUTH_PATH}"
echo

check_path "VIO-SLAM 根目录" "${VIO_ROOT}"
check_path "VIO-SLAM 主入口" "${VIO_ROOT}/run_pipeline.py"
check_path "self_aware_slam 根目录" "${AWARE_ROOT}"
check_path "数据集目录" "${DATASET_PATH}"
check_path "ground truth 文件" "${GROUNDTRUTH_PATH}"
check_path "cam0 数据目录" "${DATASET_PATH}/cam0/data"
check_path "cam0 配置" "${DATASET_PATH}/cam0/sensor.yaml"
check_path "imu0 数据" "${DATASET_PATH}/imu0/data.csv"
check_path "imu0 配置" "${DATASET_PATH}/imu0/sensor.yaml"

echo
check_python_imports "VIO-SLAM Python 依赖" "${VIO_PYTHON}" "cv2,numpy,scipy,matplotlib,tqdm,yaml"
check_python_imports "self-aware Python 依赖" "${AWARE_PYTHON}" "numpy,pandas,torch,sklearn,scipy,yaml"

echo
if [[ -d "${DATASET_PATH}/cam0/data" ]]; then
  image_count="$(find "${DATASET_PATH}/cam0/data" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')"
  print_check "cam0 PNG 数量" "${image_count}"
fi

if [[ -f "${DATASET_PATH}/imu0/data.csv" ]]; then
  imu_lines="$(wc -l < "${DATASET_PATH}/imu0/data.csv" | tr -d ' ')"
  print_check "imu0 CSV 行数" "${imu_lines}"
fi

echo
echo "检查完成。"
echo "如果上面仍有 MISS 或 MISSING，请先修复，再运行统一 pipeline。"
