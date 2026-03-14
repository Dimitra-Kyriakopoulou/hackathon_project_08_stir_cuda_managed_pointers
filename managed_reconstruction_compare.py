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

import sirf.STIR as STIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acq", required=True, help="Path to the PET acquisition .hs file")
    parser.add_argument("--image-factory", required=True, help="Path to the managed-image factory shared library")
    parser.add_argument("--acq-factory", required=True, help="Path to the managed-acquisition factory shared library")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--subsets", type=int, default=12, help="Number of subsets")
    return parser.parse_args()


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


def wrap_managed_image(factory_lib: Path, source: STIR.ImageData):
    factory = ctypes.CDLL(str(factory_lib), mode=ctypes.RTLD_GLOBAL)
    factory.make_managed_stir_image_handle_from_existing.argtypes = [ctypes.c_void_p]
    factory.make_managed_stir_image_handle_from_existing.restype = ctypes.c_void_p
    handle = factory.make_managed_stir_image_handle_from_existing(ctypes.c_void_p(int(source.handle)))
    if not handle:
        raise RuntimeError("image factory returned a null managed handle")
    image = STIR.ImageData()
    image.handle = handle
    return image


def wrap_managed_acquisition(factory_lib: Path, source: STIR.AcquisitionData):
    factory = ctypes.CDLL(str(factory_lib), mode=ctypes.RTLD_GLOBAL)
    factory.make_managed_stir_acquisition_handle_from_existing.argtypes = [ctypes.c_void_p]
    factory.make_managed_stir_acquisition_handle_from_existing.restype = ctypes.c_void_p
    handle = factory.make_managed_stir_acquisition_handle_from_existing(ctypes.c_void_p(int(source.handle)))
    if not handle:
        raise RuntimeError("acquisition factory returned a null managed handle")
    ad = STIR.AcquisitionData()
    ad.handle = handle
    return ad


def get_flags(handle, kind: bytes):
    supports_h = raw_pysirf.cSIRF_supportsArrayView(handle)
    is_managed_h = raw_pystir.cSTIR_parameter(handle, kind, b"supports_cuda_array_view")
    cuda_addr_h = raw_pystir.cSTIR_parameter(handle, kind, b"cuda_address")
    supports_view = bool(raw_pyi.intDataFromHandle(supports_h))
    is_managed = bool(raw_pyi.intDataFromHandle(is_managed_h))
    cuda_addr = int(raw_pyi.size_tDataFromHandle(cuda_addr_h))
    raw_pyi.deleteDataHandle(supports_h)
    raw_pyi.deleteDataHandle(is_managed_h)
    raw_pyi.deleteDataHandle(cuda_addr_h)
    return supports_view, is_managed, cuda_addr


def build_reconstructor(acq, image, subsets: int):
    acq_model = STIR.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_up(acq, image)
    obj_fun = STIR.make_Poisson_loglikelihood(acq, acq_model=acq_model)
    recon = STIR.OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(subsets)
    recon.set_num_subiterations(1)
    recon.set_input(acq)
    recon.set_up(image)
    return recon


def save_panel(normal_arr, managed_arr, error_arr, output_dir: Path):
    mid = normal_arr.shape[0] // 2
    vmax = float(max(np.max(normal_arr[mid]), np.max(managed_arr[mid])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(normal_arr[mid], cmap="gray", vmin=0.0, vmax=vmax)
    axes[0].set_title("Normal OSMAPOSL")
    axes[1].imshow(managed_arr[mid], cmap="gray", vmin=0.0, vmax=vmax)
    axes[1].set_title("Managed OSMAPOSL")
    axes[2].imshow(error_arr[mid], cmap="magma")
    axes[2].set_title("Absolute Error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_error_strip.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for name, arr, cmap, kwargs in (
        ("normal_reconstruction.png", normal_arr[mid], "gray", {"vmin": 0.0, "vmax": vmax}),
        ("managed_reconstruction.png", managed_arr[mid], "gray", {"vmin": 0.0, "vmax": vmax}),
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

    normal_acq = STIR.AcquisitionData(args.acq)
    managed_acq = wrap_managed_acquisition(Path(args.acq_factory), normal_acq)

    normal_image = normal_acq.create_uniform_image(1.0)
    managed_image = wrap_managed_image(Path(args.image_factory), normal_image)

    in_supports, in_managed, in_addr = get_flags(managed_acq.handle, b"AcquisitionData")
    out_supports_before, out_managed_before, out_addr_before = get_flags(managed_image.handle, b"ImageData")

    normal_recon = build_reconstructor(normal_acq, normal_image, args.subsets)
    managed_recon = build_reconstructor(managed_acq, managed_image, args.subsets)

    normal_recon.update(normal_image)
    managed_recon.update(managed_image)

    out_supports_after, out_managed_after, out_addr_after = get_flags(managed_image.handle, b"ImageData")

    normal_arr = normal_image.as_array()
    managed_arr = managed_image.as_array()
    error_arr = np.abs(normal_arr - managed_arr)

    rmse = float(np.sqrt(np.mean((normal_arr - managed_arr) ** 2)))
    mae = float(np.mean(error_arr))
    max_abs_error = float(np.max(error_arr))
    corr = float(np.corrcoef(normal_arr.ravel(), managed_arr.ravel())[0, 1])

    metrics = {
        "shape": list(normal_arr.shape),
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "correlation": corr,
        "num_subsets": args.subsets,
        "num_subiterations": 1,
        "managed_input": {
            "supports_array_view": in_supports,
            "supports_cuda_array_view": in_managed,
            "cuda_address": in_addr,
        },
        "managed_output_before": {
            "supports_array_view": out_supports_before,
            "supports_cuda_array_view": out_managed_before,
            "cuda_address": out_addr_before,
        },
        "managed_output_after": {
            "supports_array_view": out_supports_after,
            "supports_cuda_array_view": out_managed_after,
            "cuda_address": out_addr_after,
        },
        "output_address_unchanged": out_addr_before == out_addr_after,
        "normal_norm": float(np.linalg.norm(normal_arr.ravel())),
        "managed_norm": float(np.linalg.norm(managed_arr.ravel())),
        "tested_slice": int(normal_arr.shape[0] // 2),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    save_panel(normal_arr, managed_arr, error_arr, output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
