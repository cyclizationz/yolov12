#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${DEPLOYMENT_DIR}/build"
ORT_VERSION="1.22.0"
ORT_DIR="${DEPLOYMENT_DIR}/onnxruntime-linux-x64-gpu-${ORT_VERSION}"

echo "[setup] RESPAWN client environment check"
echo "[setup] deployment dir: ${DEPLOYMENT_DIR}"

missing=0

check_cmd() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    echo "[ok] ${name}: $(command -v "${name}")"
  else
    echo "[missing] ${name}"
    missing=$((missing + 1))
  fi
}

check_cmd cmake
check_cmd g++
check_cmd ffmpeg
check_cmd pkg-config

if pkg-config --exists opencv4; then
  echo "[ok] opencv4: $(pkg-config --modversion opencv4)"
else
  echo "[warn] opencv4 was not found via pkg-config; CMake may still find OpenCV through package config paths."
fi

if [[ -d "${ORT_DIR}" ]]; then
  echo "[ok] ONNX Runtime: ${ORT_DIR}"
else
  echo "[warn] ONNX Runtime GPU ${ORT_VERSION} not found at ${ORT_DIR}"
  echo "       Server builds require it. See deployment/download_ort.sh or install the expected ORT bundle."
fi

mkdir -p "${BUILD_DIR}"
echo "[ok] build dir: ${BUILD_DIR}"

cat <<EOF

Next commands:
  cd "${BUILD_DIR}"
  cmake ..
  cmake --build . --target RespawnOnlineClient -j

Optional server target:
  cmake --build . --target RespawnOnlineServer -j

Optional template-delivery target:
  cmake --build . --target RespawnTemplateServer -j

Client smoke test after a server run:
  "${BUILD_DIR}/RespawnOnlineClient" --input "${DEPLOYMENT_DIR}/outputs/online_server" --cache-mode partial-warm
EOF

if [[ "${missing}" -gt 0 ]]; then
  echo "[setup] ${missing} required command(s) were missing."
  exit 1
fi

echo "[setup] client environment check completed."
