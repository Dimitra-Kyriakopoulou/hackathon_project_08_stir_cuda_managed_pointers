#!/usr/bin/env bash
set -euo pipefail

: "${STIR_PREFIX:?set STIR_PREFIX to the patched STIR install prefix}"
: "${SIRF_SOURCE:?set SIRF_SOURCE to the patched SIRF source tree}"
: "${SIRF_BUILD:?set SIRF_BUILD to the patched SIRF build tree}"
: "${PHANTOM_HV:?set PHANTOM_HV to the PET phantom .hv file}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/build"
OVERLAY_DIR="${BUILD_DIR}/sirf_overlay"
OUTPUT_DIR="${1:-${ROOT}/output}"

mkdir -p "${BUILD_DIR}" "${OVERLAY_DIR}/sirf" "${OUTPUT_DIR}"

c++ -std=c++17 -DSTIR_WITH_CUDA -fPIC -shared \
  "${ROOT}/managed_sirf_python_factory.cpp" \
  -o "${BUILD_DIR}/libmanaged_sirf_python_factory.so" \
  -I"${ROOT}/include" \
  -I"${STIR_PREFIX}/include" \
  -I"${SIRF_SOURCE}/src/common/include" \
  -I"${SIRF_SOURCE}/src/iUtilities/include" \
  -I"${SIRF_SOURCE}/src/xSTIR/cSTIR/include" \
  -I/usr/local/cuda/include \
  -L"${STIR_PREFIX}/lib" \
  -L/usr/local/cuda/lib64 \
  -Wl,-rpath,"${STIR_PREFIX}/lib" \
  -Wl,-rpath,/usr/local/cuda/lib64 \
  -lstir_buildblock -lcudart -lgomp

ln -sf "${SIRF_SOURCE}/src/common/SIRF.py" "${OVERLAY_DIR}/sirf/SIRF.py"
ln -sf "${SIRF_SOURCE}/src/common/Utilities.py" "${OVERLAY_DIR}/sirf/Utilities.py"
ln -sf "${SIRF_BUILD}/cmake/config.py" "${OVERLAY_DIR}/sirf/config.py"
ln -sf "${SIRF_BUILD}/src/iUtilities/pyiutilities.py" "${OVERLAY_DIR}/sirf/pyiutilities.py"
ln -sf "${SIRF_BUILD}/src/iUtilities/_pyiutilities.so" "${OVERLAY_DIR}/sirf/_pyiutilities.so"
ln -sf "${SIRF_BUILD}/src/common/pysirf.py" "${OVERLAY_DIR}/sirf/pysirf.py"
ln -sf "${SIRF_BUILD}/src/common/_pysirf.so" "${OVERLAY_DIR}/sirf/_pysirf.so"
ln -sf "${SIRF_BUILD}/src/xSTIR/pSTIR/pystir.py" "${OVERLAY_DIR}/sirf/pystir.py"
ln -sf "${SIRF_BUILD}/src/xSTIR/pSTIR/_pystir.so" "${OVERLAY_DIR}/sirf/_pystir.so"
ln -sf "${SIRF_SOURCE}/src/xSTIR/pSTIR/STIR.py" "${OVERLAY_DIR}/sirf/STIR.py"
ln -sf "${SIRF_SOURCE}/src/xSTIR/pSTIR/STIR_params.py" "${OVERLAY_DIR}/sirf/STIR_params.py"
printf "__version__ = \"0.0\"\n" > "${OVERLAY_DIR}/sirf/__init__.py"
ln -sf "${SIRF_BUILD}/src/iUtilities/pyiutilities.py" "${OVERLAY_DIR}/pyiutilities.py"
ln -sf "${SIRF_BUILD}/src/iUtilities/_pyiutilities.so" "${OVERLAY_DIR}/_pyiutilities.so"
ln -sf "${SIRF_BUILD}/src/common/pysirf.py" "${OVERLAY_DIR}/pysirf.py"
ln -sf "${SIRF_BUILD}/src/common/_pysirf.so" "${OVERLAY_DIR}/_pysirf.so"
ln -sf "${SIRF_BUILD}/src/xSTIR/pSTIR/pystir.py" "${OVERLAY_DIR}/pystir.py"
ln -sf "${SIRF_BUILD}/src/xSTIR/pSTIR/_pystir.so" "${OVERLAY_DIR}/_pystir.so"

LD_LIBRARY_PATH="${STIR_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
SIRF_OVERLAY="${OVERLAY_DIR}" \
PYTHONPATH="${OVERLAY_DIR}" \
python3 "${ROOT}/managed_phantom_compare.py" \
  --phantom "${PHANTOM_HV}" \
  --factory "${BUILD_DIR}/libmanaged_sirf_python_factory.so" \
  --output "${OUTPUT_DIR}"
