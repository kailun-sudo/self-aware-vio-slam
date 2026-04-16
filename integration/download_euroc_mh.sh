#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_OUTPUT_ROOT="${WORKSPACE_ROOT}/VIO-SLAM/data"
DEFAULT_DOWNLOAD_ROOT="${WORKSPACE_ROOT}/downloads/euroc"
DEFAULT_SEQUENCE="MH_01_easy"

usage() {
  cat <<EOF
用法:
  bash integration/download_euroc_mh.sh [可选参数]

可选参数:
  --sequence NAME         要下载的 EuRoC 序列，默认: ${DEFAULT_SEQUENCE}
  --output-root PATH      数据输出根目录，默认: ${DEFAULT_OUTPUT_ROOT}
  --download-root PATH    zip 缓存目录，默认: ${DEFAULT_DOWNLOAD_ROOT}
  --keep-zip              下载完成后保留 zip
  -h, --help              显示帮助

支持的序列命名:
  - Machine Hall: MH_01_easy, MH_02_easy, MH_03_medium, MH_04_difficult, MH_05_difficult
  - Vicon Room 1: V1_01_easy, V1_02_medium, V1_03_difficult
  - Vicon Room 2: V2_01_easy, V2_02_medium, V2_03_difficult

脚本行为:
  1. 直接下载单序列 zip，而不是整包
  2. 解压出其中的 mav0
  3. 整理成 <output-root>/mav0
EOF
}

SEQUENCE="${DEFAULT_SEQUENCE}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
DOWNLOAD_ROOT="${DEFAULT_DOWNLOAD_ROOT}"
KEEP_ZIP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sequence)
      SEQUENCE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --download-root)
      DOWNLOAD_ROOT="$2"
      shift 2
      ;;
    --keep-zip)
      KEEP_ZIP=1
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

sequence_url() {
  local sequence="$1"
  local base="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
  if [[ "${sequence}" == MH_* ]]; then
    echo "${base}/machine_hall/${sequence}/${sequence}.zip"
  elif [[ "${sequence}" == V1_* ]]; then
    echo "${base}/vicon_room1/${sequence}/${sequence}.zip"
  elif [[ "${sequence}" == V2_* ]]; then
    echo "${base}/vicon_room2/${sequence}/${sequence}.zip"
  else
    echo ""
  fi
}

ZIP_URL="$(sequence_url "${SEQUENCE}")"
if [[ -z "${ZIP_URL}" ]]; then
  echo "错误: 不支持的序列名 ${SEQUENCE}" >&2
  usage
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${DOWNLOAD_ROOT}"

ZIP_PATH="${DOWNLOAD_ROOT}/${SEQUENCE}.zip"
TMP_DIR="$(mktemp -d "${DOWNLOAD_ROOT}/extract.XXXXXX")"
TARGET_DIR="${OUTPUT_ROOT}/mav0"

cleanup() {
  rm -rf "${TMP_DIR}"
  if [[ "${KEEP_ZIP}" -eq 0 && -f "${ZIP_PATH}" ]]; then
    rm -f "${ZIP_PATH}"
  fi
}
trap cleanup EXIT

download_zip() {
  local url="$1"
  local output_path="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -c --content-disposition -O "${output_path}" "${url}"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --connect-timeout 20 "${url}" -o "${output_path}"
  else
    echo "错误: 系统里既没有 wget 也没有 curl，无法下载数据" >&2
    exit 1
  fi
}

echo "== 下载 EuRoC 单序列数据 =="
echo "sequence:      ${SEQUENCE}"
echo "zip url:       ${ZIP_URL}"
echo "output root:   ${OUTPUT_ROOT}"
echo "download root: ${DOWNLOAD_ROOT}"

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo
  echo "== [1/3] 下载 zip =="
  download_zip "${ZIP_URL}" "${ZIP_PATH}"
else
  echo
  echo "== [1/3] 复用已下载 zip =="
  echo "${ZIP_PATH}"
fi

echo
echo "== [2/3] 解压数据 =="
unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"

EXTRACTED_MAV0="$(find "${TMP_DIR}" -type d -name mav0 | head -n 1)"
if [[ -z "${EXTRACTED_MAV0}" ]]; then
  echo "错误: 解压后未找到 mav0 目录" >&2
  exit 1
fi

echo
echo "== [3/3] 安装到 ${TARGET_DIR} =="
if [[ -e "${TARGET_DIR}" ]]; then
  BACKUP_DIR="${OUTPUT_ROOT}/mav0_backup_$(date +%Y%m%d_%H%M%S)"
  echo "检测到已有 ${TARGET_DIR}"
  echo "自动备份到 ${BACKUP_DIR}"
  mv "${TARGET_DIR}" "${BACKUP_DIR}"
fi

mv "${EXTRACTED_MAV0}" "${TARGET_DIR}"

echo
echo "完成。当前数据目录："
echo "  ${TARGET_DIR}"
echo
echo "可以直接运行："
echo "  cd ${WORKSPACE_ROOT}/VIO-SLAM"
echo "  ./.venv/bin/python run_pipeline.py --output ${WORKSPACE_ROOT}/outputs/mh01"
