#!/usr/bin/env bash

set -euo pipefail

WORKSPACE_DEFAULT="/Users/kailunwang/Desktop/ossa"
SEQUENCE_DEFAULTS="MH_01_easy,MH_02_easy,MH_03_medium"
DATASET_ROOT_DEFAULT="${WORKSPACE_DEFAULT}/VIO-SLAM/data/sequences"
PACKAGED_ROOT_DEFAULT="${WORKSPACE_DEFAULT}/self_aware_slam/slam_metrics_dataset"

usage() {
  cat <<'EOF'
用法:
  bash integration/run_batch_unified_pipeline.sh [可选参数]

可选参数:
  --workspace PATH         工程根目录，默认: /Users/kailunwang/Desktop/ossa
  --dataset-root PATH      EuRoC 序列根目录，默认: VIO-SLAM/data/sequences
  --sequences CSV          逗号分隔序列名，默认: MH_01_easy,MH_02_easy,MH_03_medium
  --vio-python PATH        VIO-SLAM Python，可选
  --aware-python PATH      self-aware Python，可选
  --output-prefix NAME     输出目录前缀，默认: unified
  --packaged-root PATH     打包训练序列根目录，默认: self_aware_slam/slam_metrics_dataset
  --skip-slam              跳过主 SLAM
  --skip-demo              跳过 unified demo
  --skip-analysis          跳过结果分析
  --skip-package           跳过训练集打包
  --dry-run                只打印将要执行的步骤
  -h, --help               显示帮助

默认约定:
  每个序列目录应位于:
    <dataset-root>/<sequence>/mav0

示例:
  bash integration/run_batch_unified_pipeline.sh \
    --dataset-root /data/euroc \
    --sequences MH_01_easy,MH_02_easy
EOF
}

WORKSPACE="${WORKSPACE_DEFAULT}"
DATASET_ROOT="${DATASET_ROOT_DEFAULT}"
SEQUENCES_CSV="${SEQUENCE_DEFAULTS}"
VIO_PYTHON=""
AWARE_PYTHON=""
OUTPUT_PREFIX="unified"
PACKAGED_ROOT="${PACKAGED_ROOT_DEFAULT}"
SKIP_SLAM=0
SKIP_DEMO=0
SKIP_ANALYSIS=0
SKIP_PACKAGE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --sequences)
      SEQUENCES_CSV="$2"
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
    --output-prefix)
      OUTPUT_PREFIX="$2"
      shift 2
      ;;
    --packaged-root)
      PACKAGED_ROOT="$2"
      shift 2
      ;;
    --skip-slam)
      SKIP_SLAM=1
      shift
      ;;
    --skip-demo)
      SKIP_DEMO=1
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=1
      shift
      ;;
    --skip-package)
      SKIP_PACKAGE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

VIO_ROOT="${WORKSPACE}/VIO-SLAM"
AWARE_ROOT="${WORKSPACE}/self_aware_slam"
INTEGRATION_ROOT="${WORKSPACE}/integration"

if [[ -z "${VIO_PYTHON}" ]]; then
  VIO_PYTHON="${VIO_ROOT}/.venv/bin/python"
fi
if [[ -z "${AWARE_PYTHON}" ]]; then
  AWARE_PYTHON="${AWARE_ROOT}/venv/bin/python"
fi

run_or_echo() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

normalize_name() {
  local sequence="$1"
  case "${sequence}" in
    MH_01_easy|MH_01) echo "MH_01" ;;
    MH_02_easy|MH_02) echo "MH_02" ;;
    MH_03_medium|MH_03) echo "MH_03" ;;
    MH_04_difficult|MH_04) echo "MH_04" ;;
    MH_05_difficult|MH_05) echo "MH_05" ;;
    V1_01_easy|V1_01) echo "V1_01" ;;
    V1_02_medium|V1_02) echo "V1_02" ;;
    V1_03_difficult|V1_03) echo "V1_03" ;;
    V2_01_easy|V2_01) echo "V2_01" ;;
    V2_02_medium|V2_02) echo "V2_02" ;;
    V2_03_difficult|V2_03) echo "V2_03" ;;
    *) echo "${sequence}" ;;
  esac
}

IFS=',' read -r -a SEQUENCES <<< "${SEQUENCES_CSV}"

echo "== Batch Unified Pipeline =="
echo "workspace:      ${WORKSPACE}"
echo "dataset root:   ${DATASET_ROOT}"
echo "sequences:      ${SEQUENCES_CSV}"
echo "output prefix:  ${OUTPUT_PREFIX}"
echo "packaged root:  ${PACKAGED_ROOT}"
echo

for sequence in "${SEQUENCES[@]}"; do
  sequence="$(echo "${sequence}" | xargs)"
  short_name="$(normalize_name "${sequence}")"
  dataset_path="${DATASET_ROOT}/${sequence}/mav0"
  groundtruth_path="${dataset_path}/state_groundtruth_estimate0/data.csv"
  slam_output="${WORKSPACE}/outputs/${OUTPUT_PREFIX}_${short_name}"
  demo_output="${WORKSPACE}/outputs/${OUTPUT_PREFIX}_${short_name}_self_aware"
  analysis_output="${WORKSPACE}/outputs/${OUTPUT_PREFIX}_${short_name}_analysis"

  echo "------------------------------------------------------------"
  echo "Sequence:       ${sequence}"
  echo "Short name:     ${short_name}"
  echo "Dataset path:   ${dataset_path}"
  echo "SLAM output:    ${slam_output}"
  echo "Demo output:    ${demo_output}"
  echo "Analysis out:   ${analysis_output}"
  echo "Packaged out:   ${PACKAGED_ROOT}/${short_name}_unified"

  if [[ ! -d "${dataset_path}" ]]; then
    echo "跳过: 数据集目录不存在 ${dataset_path}" >&2
    continue
  fi
  if [[ ! -f "${groundtruth_path}" ]]; then
    echo "跳过: ground truth 不存在 ${groundtruth_path}" >&2
    continue
  fi

  if [[ "${SKIP_SLAM}" -eq 0 ]]; then
    echo "== [1/4] 运行主 SLAM =="
    run_or_echo "${VIO_PYTHON}" "${VIO_ROOT}/run_pipeline.py" \
      --data_path "${dataset_path}" \
      --output "${slam_output}"
  fi

  if [[ "${SKIP_DEMO}" -eq 0 ]]; then
    echo "== [2/4] 运行 unified demo =="
    run_or_echo "${AWARE_PYTHON}" "${INTEGRATION_ROOT}/run_offline_unified_demo.py" \
      --metrics "${slam_output}/slam_metrics.csv" \
      --estimated "${slam_output}/estimated_tum.txt" \
      --groundtruth "${groundtruth_path}" \
      --output-dir "${demo_output}" \
      --config "${AWARE_ROOT}/configs/config.yaml"
  fi

  if [[ "${SKIP_ANALYSIS}" -eq 0 ]]; then
    echo "== [3/4] 结果分析 =="
    run_or_echo "${AWARE_PYTHON}" "${INTEGRATION_ROOT}/analyze_unified_results.py" \
      --predictions "${demo_output}/reliability_predictions.csv" \
      --summary "${demo_output}/summary.txt" \
      --output-dir "${analysis_output}"
  fi

  if [[ "${SKIP_PACKAGE}" -eq 0 ]]; then
    echo "== [4/4] 训练序列打包 =="
    run_or_echo "${AWARE_PYTHON}" "${AWARE_ROOT}/scripts/package_unified_sequence.py" \
      --metrics "${slam_output}/slam_metrics.csv" \
      --estimated "${slam_output}/estimated_tum.txt" \
      --groundtruth "${groundtruth_path}" \
      --sequence-name "${short_name}_unified" \
      --dataset-root "${PACKAGED_ROOT}"
  fi

  echo
done

echo "批处理完成。"
