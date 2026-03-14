#!/usr/bin/env python3
import argparse
import ctypes
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIRF_OVERLAY = Path(os.environ.get("SIRF_OVERLAY", "/work/sirf_overlay"))

ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pystir.so"), mode=ctypes.RTLD_GLOBAL)
raw_pyi = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pyiutilities.so"), mode=ctypes.RTLD_GLOBAL)
raw_pysirf = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pysirf.so"), mode=ctypes.RTLD_GLOBAL)
raw_pystir = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pystir.so"), mode=ctypes.RTLD_GLOBAL)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="Path to the managed-acquisition factory shared library")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def expect_ok(status_handle):
    if raw_pyi.executionStatus(status_handle) != 0:
        msg = raw_pyi.executionError(status_handle)
        text = msg.decode("utf-8", errors="replace") if msg else "unknown engine error"
        raw_pyi.deleteDataHandle(status_handle)
        raise RuntimeError(text)
    raw_pyi.deleteDataHandle(status_handle)


def make_managed_acquisition(factory_lib: Path):
    factory = ctypes.CDLL(str(factory_lib), mode=ctypes.RTLD_GLOBAL)
    factory.make_managed_stir_acquisition_handle.restype = ctypes.c_void_p
    handle = factory.make_managed_stir_acquisition_handle()
    if not handle:
        raise RuntimeError("factory returned a null managed acquisition handle")
    return handle


raw_pyi.executionStatus.argtypes = [ctypes.c_void_p]
raw_pyi.executionStatus.restype = ctypes.c_int
raw_pyi.executionError.argtypes = [ctypes.c_void_p]
raw_pyi.executionError.restype = ctypes.c_char_p
raw_pyi.deleteDataHandle.argtypes = [ctypes.c_void_p]
raw_pyi.intDataFromHandle.argtypes = [ctypes.c_void_p]
raw_pyi.intDataFromHandle.restype = ctypes.c_int
raw_pyi.size_tDataFromHandle.argtypes = [ctypes.c_void_p]
raw_pyi.size_tDataFromHandle.restype = ctypes.c_size_t
raw_pysirf.cSIRF_supportsArrayView.argtypes = [ctypes.c_void_p]
raw_pysirf.cSIRF_supportsArrayView.restype = ctypes.c_void_p
raw_pystir.cSTIR_parameter.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
raw_pystir.cSTIR_parameter.restype = ctypes.c_void_p
raw_pystir.cSTIR_getAcquisitionDataDimensions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
raw_pystir.cSTIR_getAcquisitionDataDimensions.restype = ctypes.c_void_p
raw_pystir.cSTIR_getAcquisitionData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
raw_pystir.cSTIR_getAcquisitionData.restype = ctypes.c_void_p


def get_array(handle):
    dims = np.zeros(16, dtype=np.int32)
    expect_ok(raw_pystir.cSTIR_getAcquisitionDataDimensions(handle, dims.ctypes.data))
    shape = tuple(int(v) for v in dims[:4][::-1])
    data = np.empty(shape, dtype=np.float32)
    expect_ok(raw_pystir.cSTIR_getAcquisitionData(handle, data.ctypes.data))
    return data


def get_flags(handle):
    supports_h = raw_pysirf.cSIRF_supportsArrayView(handle)
    is_managed_h = raw_pystir.cSTIR_parameter(handle, b"AcquisitionData", b"supports_cuda_array_view")
    cuda_addr_h = raw_pystir.cSTIR_parameter(handle, b"AcquisitionData", b"cuda_address")
    supports_view = bool(raw_pyi.intDataFromHandle(supports_h))
    is_managed = bool(raw_pyi.intDataFromHandle(is_managed_h))
    cuda_addr = int(raw_pyi.size_tDataFromHandle(cuda_addr_h))
    raw_pyi.deleteDataHandle(supports_h)
    raw_pyi.deleteDataHandle(is_managed_h)
    raw_pyi.deleteDataHandle(cuda_addr_h)
    return supports_view, is_managed, cuda_addr


def save_panel(expected_arr, managed_arr, error_arr, output_dir: Path):
    tof = 0
    sino = expected_arr.shape[1] // 2
    slice_min = float(np.min(expected_arr[tof, sino]))
    slice_max = float(np.max(expected_arr[tof, sino]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(expected_arr[tof, sino], cmap="gray", vmin=slice_min, vmax=slice_max)
    axes[0].set_title("Expected Ramp")
    axes[1].imshow(managed_arr[tof, sino], cmap="gray", vmin=slice_min, vmax=slice_max)
    axes[1].set_title("Managed Acquisition")
    axes[2].imshow(error_arr[tof, sino], cmap="magma")
    axes[2].set_title("Absolute Error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "acquisition_error_strip.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for name, arr, cmap, kwargs in (
        ("expected_slice.png", expected_arr[tof, sino], "gray", {"vmin": slice_min, "vmax": slice_max}),
        ("managed_slice.png", managed_arr[tof, sino], "gray", {"vmin": slice_min, "vmax": slice_max}),
        ("abs_error_slice.png", error_arr[tof, sino], "magma", {}),
    ):
        plt.figure(figsize=(4, 4))
        plt.imshow(arr, cmap=cmap, **kwargs)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=160, bbox_inches="tight", pad_inches=0)
        plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = make_managed_acquisition(Path(args.factory))

    managed_arr = get_array(handle)
    expected_arr = np.arange(managed_arr.size, dtype=np.float32).reshape(managed_arr.shape)
    error_arr = np.abs(expected_arr - managed_arr)
    supports_view, is_managed, cuda_addr = get_flags(handle)

    rmse = float(np.sqrt(np.mean((expected_arr - managed_arr) ** 2)))
    mae = float(np.mean(error_arr))
    max_abs_error = float(np.max(error_arr))
    corr = float(np.corrcoef(expected_arr.ravel(), managed_arr.ravel())[0, 1])

    metrics = {
        "shape": list(managed_arr.shape),
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "correlation": corr,
        "supports_array_view": supports_view,
        "supports_cuda_array_view": is_managed,
        "cuda_address": cuda_addr,
        "tested_slice": {"tof": 0, "sinogram": int(managed_arr.shape[1] // 2)},
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_panel(expected_arr, managed_arr, error_arr, output_dir)
    print(json.dumps(metrics, indent=2))
    raw_pyi.deleteDataHandle(handle)


if __name__ == "__main__":
    main()
