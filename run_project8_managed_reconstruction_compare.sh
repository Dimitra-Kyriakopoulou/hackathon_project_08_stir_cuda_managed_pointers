#!/usr/bin/env bash
set -euo pipefail

VM_USER_HOST="kyriakopouloudimitra@34.140.209.173"
SSH_KEY="/home/fotis/.ssh/petric_gcp"
REMOTE_ROOT="/home/kyriakopouloudimitra/hackathon_vm"
REMOTE_PROBE_DIR="${REMOTE_ROOT}/project_08_probe"
REMOTE_DATA_DIR="${REMOTE_PROBE_DIR}/pet_data"
REMOTE_OUTPUT_DIR="${REMOTE_ROOT}/project_08_outputs/project_08_managed_reconstruction_compare"
LOCAL_OUTPUT_DIR="/home/fotis/hackathon/prior_10.3.26/outputs/project_08_managed_reconstruction_compare"

mkdir -p "${LOCAL_OUTPUT_DIR}"

ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=5 -o LogLevel=ERROR \
  "${VM_USER_HOST}" \
  "mkdir -p ${REMOTE_PROBE_DIR}/include/stir ${REMOTE_DATA_DIR} ${REMOTE_OUTPUT_DIR}"

scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  /home/fotis/hackathon/prior_10.3.26/project_08_stir_cuda_managed_pointers/vm_prototype/managed_sirf_python_factory.cpp \
  /home/fotis/hackathon/prior_10.3.26/project_08_stir_cuda_managed_pointers/vm_prototype/managed_sirf_acquisition_factory.cpp \
  /home/fotis/hackathon/prior_10.3.26/project_08_stir_cuda_managed_pointers/vm_prototype/managed_reconstruction_compare_raw.py \
  "${VM_USER_HOST}:${REMOTE_PROBE_DIR}/"

scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  /home/fotis/hackathon/prior_10.3.26/project_08_stir_cuda_managed_pointers/vm_prototype/cuda_managed_memory.h \
  "${VM_USER_HOST}:${REMOTE_PROBE_DIR}/include/stir/cuda_managed_memory.h"

scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  /home/fotis/hackathon/11.3.26/SIRF/data/examples/PET/Utahscat600k_ca_seg4.hs \
  /home/fotis/hackathon/11.3.26/SIRF/data/examples/PET/Utahscat600k_ca_seg4.s \
  "${VM_USER_HOST}:${REMOTE_DATA_DIR}/"

ssh -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=5 -o LogLevel=ERROR \
  "${VM_USER_HOST}" \
  "sudo docker exec project8-build bash -lc '
    apt-get update >/tmp/apt.log
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-dev python3-numpy python3-matplotlib python3-deprecation libboost-all-dev >/tmp/apt-install.log
    mkdir -p /work/project_08_probe/include/stir /work/project_08_probe/pet_data /work/project_08_outputs/project_08_managed_reconstruction_compare
    g++ -std=c++17 -DSTIR_WITH_CUDA -fPIC -shared /work/project_08_probe/managed_sirf_python_factory.cpp \
      -o /work/project_08_probe/libmanaged_sirf_python_factory.so \
      -I/work/project_08_probe/include \
      -I/work/STIR/src/include \
      -I/work/STIR-install-full-cuda/include \
      -I/work/SIRF/src/common/include \
      -I/work/SIRF/src/iUtilities/include \
      -I/work/SIRF/src/xSTIR/cSTIR/include \
      -I/usr/local/cuda/include \
      -L/work/STIR-install-full-cuda/lib \
      -L/usr/local/cuda/lib64 \
      -Wl,-rpath,/work/STIR-install-full-cuda/lib \
      -Wl,-rpath,/usr/local/cuda/lib64 \
      -lstir_buildblock -lcudart -lgomp
    g++ -std=c++17 -DSTIR_WITH_CUDA -fPIC -shared /work/project_08_probe/managed_sirf_acquisition_factory.cpp \
      -o /work/project_08_probe/libmanaged_sirf_acquisition_factory.so \
      -I/work/project_08_probe/include \
      -I/work/STIR/src/include \
      -I/work/STIR-install-full-cuda/include \
      -I/work/SIRF/src/common/include \
      -I/work/SIRF/src/iUtilities/include \
      -I/work/SIRF/src/xSTIR/cSTIR/include \
      -I/usr/local/cuda/include \
      -L/work/STIR-install-full-cuda/lib \
      -L/usr/local/cuda/lib64 \
      -Wl,-rpath,/work/STIR-install-full-cuda/lib \
      -Wl,-rpath,/usr/local/cuda/lib64 \
      -lstir_buildblock -lcudart -lgomp
    rm -rf /work/sirf_overlay
    mkdir -p /work/sirf_overlay/sirf
    ln -sf /work/SIRF/src/common/SIRF.py /work/sirf_overlay/sirf/SIRF.py
    ln -sf /work/SIRF/src/common/Utilities.py /work/sirf_overlay/sirf/Utilities.py
    ln -sf /work/SIRF-build-pystir/cmake/config.py /work/sirf_overlay/sirf/config.py
    ln -sf /work/SIRF-build-pystir/src/iUtilities/pyiutilities.py /work/sirf_overlay/sirf/pyiutilities.py
    ln -sf /work/SIRF-build-pystir/src/iUtilities/_pyiutilities.so /work/sirf_overlay/sirf/_pyiutilities.so
    ln -sf /work/SIRF-build-pystir/src/common/pysirf.py /work/sirf_overlay/sirf/pysirf.py
    ln -sf /work/SIRF-build-pystir/src/common/_pysirf.so /work/sirf_overlay/sirf/_pysirf.so
    ln -sf /work/SIRF-build-pystir/src/xSTIR/pSTIR/pystir.py /work/sirf_overlay/sirf/pystir.py
    ln -sf /work/SIRF-build-pystir/src/xSTIR/pSTIR/_pystir.so /work/sirf_overlay/sirf/_pystir.so
    ln -sf /work/SIRF/src/xSTIR/pSTIR/STIR.py /work/sirf_overlay/sirf/STIR.py
    ln -sf /work/SIRF/src/xSTIR/pSTIR/STIR_params.py /work/sirf_overlay/sirf/STIR_params.py
    printf \"__version__ = \\\"0.0\\\"\\n\" > /work/sirf_overlay/sirf/__init__.py
    ln -sf /work/SIRF-build-pystir/src/iUtilities/pyiutilities.py /work/sirf_overlay/pyiutilities.py
    ln -sf /work/SIRF-build-pystir/src/iUtilities/_pyiutilities.so /work/sirf_overlay/_pyiutilities.so
    ln -sf /work/SIRF-build-pystir/src/common/pysirf.py /work/sirf_overlay/pysirf.py
    ln -sf /work/SIRF-build-pystir/src/common/_pysirf.so /work/sirf_overlay/_pysirf.so
    ln -sf /work/SIRF-build-pystir/src/xSTIR/pSTIR/pystir.py /work/sirf_overlay/pystir.py
    ln -sf /work/SIRF-build-pystir/src/xSTIR/pSTIR/_pystir.so /work/sirf_overlay/_pystir.so
    LD_LIBRARY_PATH=/work/STIR-install-full-cuda/lib:${LD_LIBRARY_PATH:-} PYTHONPATH=/work/sirf_overlay \
      python3 /work/project_08_probe/managed_reconstruction_compare_raw.py \
        --acq /work/project_08_probe/pet_data/Utahscat600k_ca_seg4.hs \
        --image-factory /work/project_08_probe/libmanaged_sirf_python_factory.so \
        --acq-factory /work/project_08_probe/libmanaged_sirf_acquisition_factory.so \
        --output /work/project_08_outputs/project_08_managed_reconstruction_compare
  '"

scp -r -i "${SSH_KEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "${VM_USER_HOST}:${REMOTE_OUTPUT_DIR}/." \
  "${LOCAL_OUTPUT_DIR}/"

echo "copied outputs to ${LOCAL_OUTPUT_DIR}"
