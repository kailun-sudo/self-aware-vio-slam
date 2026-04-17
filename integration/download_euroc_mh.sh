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
  1. 从新的 EuRoC 官方托管源下载对应大类压缩包
  2. 只解压目标序列对应的 mav0
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

sequence_family() {
  local sequence="$1"
  if [[ "${sequence}" == MH_* ]]; then
    echo "machine_hall"
  elif [[ "${sequence}" == V1_* ]]; then
    echo "vicon_room1"
  elif [[ "${sequence}" == V2_* ]]; then
    echo "vicon_room2"
  else
    echo ""
  fi
}

package_url() {
  local family="$1"
  case "${family}" in
    machine_hall)
      echo "https://www.research-collection.ethz.ch/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/download"
      ;;
    vicon_room1)
      echo "https://www.research-collection.ethz.ch/bitstreams/02ecda9a-298f-498b-970c-b7c44334d880/download"
      ;;
    vicon_room2)
      echo "https://www.research-collection.ethz.ch/bitstreams/ea12bc01-3677-4b4c-853d-87c7870b8c44/download"
      ;;
    *)
      echo ""
      ;;
  esac
}

PACKAGE_FAMILY="$(sequence_family "${SEQUENCE}")"
ZIP_URL="$(package_url "${PACKAGE_FAMILY}")"
if [[ -z "${ZIP_URL}" || -z "${PACKAGE_FAMILY}" ]]; then
  echo "错误: 不支持的序列名 ${SEQUENCE}" >&2
  usage
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}" "${DOWNLOAD_ROOT}"

ZIP_PATH="${DOWNLOAD_ROOT}/${PACKAGE_FAMILY}.zip"
TMP_DIR="$(mktemp -d "/tmp/ossa_euroc_extract.XXXXXX")"
TARGET_DIR="${OUTPUT_ROOT}/mav0"
INNER_ZIP_PATH="${TMP_DIR}/${SEQUENCE}.zip"

cleanup() {
  if [[ -d "${TMP_DIR}" ]]; then
    for _ in 1 2 3; do
      /bin/rm -rf "${TMP_DIR}" 2>/dev/null && break
      sleep 1
    done
    if [[ -d "${TMP_DIR}" ]]; then
      echo "警告: 未能完全清理临时目录 ${TMP_DIR}" >&2
    fi
  fi
  if [[ "${KEEP_ZIP}" -eq 0 && -f "${ZIP_PATH}" ]]; then
    rm -f "${ZIP_PATH}"
  fi
}
trap cleanup EXIT

download_zip() {
  local url="$1"
  local output_path="$2"
  local urls=("${url}")

  if [[ "${url}" == https://* ]]; then
    urls+=("${url/https:\/\//http://}")
  elif [[ "${url}" == http://* ]]; then
    urls+=("${url/http:\/\//https://}")
  fi

  for candidate_url in "${urls[@]}"; do
    echo "尝试下载: ${candidate_url}"

    if command -v wget >/dev/null 2>&1; then
      if wget \
        -c \
        --content-disposition \
        --tries=3 \
        --timeout=30 \
        --waitretry=2 \
        -O "${output_path}" \
        "${candidate_url}"; then
        return 0
      fi
    elif command -v curl >/dev/null 2>&1; then
      if curl \
        -fL \
        --retry 3 \
        --retry-delay 2 \
        --connect-timeout 20 \
        --max-time 0 \
        "${candidate_url}" \
        -o "${output_path}"; then
        return 0
      fi
    else
      echo "错误: 系统里既没有 wget 也没有 curl，无法下载数据" >&2
      exit 1
    fi

    echo "下载失败，继续尝试下一个候选地址..."
  done

  echo "错误: 所有下载地址都失败了" >&2
  return 1
}

echo "== 下载 EuRoC 单序列数据 =="
echo "sequence:      ${SEQUENCE}"
echo "package:       ${PACKAGE_FAMILY}"
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
INNER_ZIP_MEMBER="$(zipinfo -1 "${ZIP_PATH}" | grep -E "/${SEQUENCE}/${SEQUENCE}\\.zip$" | head -n 1 || true)"

if [[ -n "${INNER_ZIP_MEMBER}" ]]; then
  echo "检测到官方新格式：外层大包内嵌 ${SEQUENCE}.zip"
  unzip -q -p "${ZIP_PATH}" "${INNER_ZIP_MEMBER}" > "${INNER_ZIP_PATH}"
  unzip -q "${INNER_ZIP_PATH}" -d "${TMP_DIR}"
  EXTRACTED_MAV0="$(find "${TMP_DIR}" -type d -path "*/mav0" | head -n 1)"
else
  SEQUENCE_PREFIX="$(zipinfo -1 "${ZIP_PATH}" | grep -E "(^|/+)${SEQUENCE}/mav0/" | head -n 1 | sed 's#^\(.*'"${SEQUENCE}"'/mav0/.*\)$#\1#')"

  if [[ -n "${SEQUENCE_PREFIX}" ]]; then
    echo "检测到压缩包内路径前缀: ${SEQUENCE_PREFIX}"
    unzip -q "${ZIP_PATH}" "${SEQUENCE_PREFIX}*" -d "${TMP_DIR}"
  else
    echo "未能直接定位序列前缀，回退到全量解压后查找目标序列..."
    unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"
  fi

  EXTRACTED_MAV0="$(find "${TMP_DIR}" -type d -path "*/${SEQUENCE}/mav0" | head -n 1)"
  if [[ -z "${EXTRACTED_MAV0}" ]]; then
    EXTRACTED_MAV0="$(find "${TMP_DIR}" -type d -name mav0 | head -n 1)"
  fi
fi

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
