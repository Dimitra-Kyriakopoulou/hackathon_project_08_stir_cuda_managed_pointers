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

SIRF_OVERLAY = Path(os.environ["SIRF_OVERLAY"])

ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pystir.so"), mode=ctypes.RTLD_GLOBAL)
raw_pyi = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pyiutilities.so"), mode=ctypes.RTLD_GLOBAL)
raw_pysirf = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pysirf.so"), mode=ctypes.RTLD_GLOBAL)
raw_pystir = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pystir.so"), mode=ctypes.RTLD_GLOBAL)

import sirf.STIR as STIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phantom", required=True, help="Path to the PET phantom .hv file")
    parser.add_argument("--factory", required=True, help="Path to the managed-image factory shared library")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def make_managed_image(factory_lib: Path, phantom_path: Path):
    factory = ctypes.CDLL(str(factory_lib), mode=ctypes.RTLD_GLOBAL)
    factory.make_managed_stir_image_handle_from_file.argtypes = [ctypes.c_char_p]
    factory.make_managed_stir_image_handle_from_file.restype = ctypes.c_void_p
    handle = factory.make_managed_stir_image_handle_from_file(str(phantom_path).encode("utf-8"))
    if not handle:
        raise RuntimeError("factory returned a null managed image handle")
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
raw_pystir.cSTIR_getImageDimensions.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
raw_pystir.cSTIR_getImageDimensions.restype = ctypes.c_void_p
raw_pystir.cSTIR_getImageData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
raw_pystir.cSTIR_getImageData.restype = ctypes.c_void_p


def expect_ok(status_handle):
    if raw_pyi.executionStatus(status_handle) != 0:
        msg = raw_pyi.executionError(status_handle)
        text = msg.decode("utf-8", errors="replace") if msg else "unknown engine error"
        raw_pyi.deleteDataHandle(status_handle)
        raise RuntimeError(text)
    raw_pyi.deleteDataHandle(status_handle)


def get_managed_array(handle):
    dims = np.zeros(10, dtype=np.int32)
    expect_ok(raw_pystir.cSTIR_getImageDimensions(handle, dims.ctypes.data))
    shape = tuple(int(v) for v in dims[:3])
    data = np.empty(shape, dtype=np.float32)
    expect_ok(raw_pystir.cSTIR_getImageData(handle, data.ctypes.data))
    return data


def get_managed_flags(handle):
    supports_h = raw_pysirf.cSIRF_supportsArrayView(handle)
    is_managed_h = raw_pystir.cSTIR_parameter(handle, b"ImageData", b"is_cuda_managed")
    cuda_addr_h = raw_pystir.cSTIR_parameter(handle, b"ImageData", b"cuda_address")
    supports_view = bool(raw_pyi.intDataFromHandle(supports_h))
    is_managed = bool(raw_pyi.intDataFromHandle(is_managed_h))
    cuda_addr = int(raw_pyi.size_tDataFromHandle(cuda_addr_h))
    raw_pyi.deleteDataHandle(supports_h)
    raw_pyi.deleteDataHandle(is_managed_h)
    raw_pyi.deleteDataHandle(cuda_addr_h)
    return supports_view, is_managed, cuda_addr


def save_panel(phantom_arr, managed_arr, error_arr, output_dir: Path):
    mid = phantom_arr.shape[0] // 2
    vmax = float(np.max(phantom_arr))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(phantom_arr[mid], cmap="gray", vmin=0.0, vmax=vmax)
    axes[0].set_title("Reference Phantom")
    axes[1].imshow(managed_arr[mid], cmap="gray", vmin=0.0, vmax=vmax)
    axes[1].set_title("Managed-Memory Image")
    axes[2].imshow(error_arr[mid], cmap="magma")
    axes[2].set_title("Absolute Error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "phantom_managed_error_strip.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for name, arr, cmap, kwargs in (
        ("phantom.png", phantom_arr[mid], "gray", {"vmin": 0.0, "vmax": vmax}),
        ("managed_image.png", managed_arr[mid], "gray", {"vmin": 0.0, "vmax": vmax}),
        ("abs_error.png", error_arr[mid], "magma", {}),
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
    phantom_path = Path(args.phantom)
    phantom = STIR.ImageData(str(phantom_path))
    managed_handle = make_managed_image(Path(args.factory), phantom_path)

    phantom_arr = phantom.as_array()
    managed_arr = get_managed_array(managed_handle)
    error_arr = np.abs(phantom_arr - managed_arr)
    supports_view, is_managed, cuda_addr = get_managed_flags(managed_handle)

    rmse = float(np.sqrt(np.mean((phantom_arr - managed_arr) ** 2)))
    mae = float(np.mean(error_arr))
    max_abs_error = float(np.max(error_arr))
    corr = float(np.corrcoef(phantom_arr.ravel(), managed_arr.ravel())[0, 1])

    metrics = {
        "shape": list(phantom_arr.shape),
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "correlation": corr,
        "supports_array_view": supports_view,
        "is_cuda_managed": is_managed,
        "cuda_address": cuda_addr,
        "mid_slice_index": int(phantom_arr.shape[0] // 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    save_panel(phantom_arr, managed_arr, error_arr, output_dir)

    print(json.dumps(metrics, indent=2))
    raw_pyi.deleteDataHandle(managed_handle)


if __name__ == "__main__":
    main()
