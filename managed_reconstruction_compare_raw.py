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

raw_pyi = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pyiutilities.so"), mode=ctypes.RTLD_GLOBAL)
raw_pystir = ctypes.CDLL(str(SIRF_OVERLAY / "sirf" / "_pystir.so"), mode=ctypes.RTLD_GLOBAL)


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
raw_pyi.intDataHandle.argtypes = [ctypes.c_int]
raw_pyi.intDataHandle.restype = ctypes.c_void_p
raw_pyi.intDataFromHandle.argtypes = [ctypes.c_void_p]
raw_pyi.intDataFromHandle.restype = ctypes.c_int
raw_pyi.size_tDataFromHandle.argtypes = [ctypes.c_void_p]
raw_pyi.size_tDataFromHandle.restype = ctypes.c_size_t

raw_pystir.cSTIR_objectFromFile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
raw_pystir.cSTIR_objectFromFile.restype = ctypes.c_void_p
raw_pystir.cSTIR_newObject.argtypes = [ctypes.c_char_p]
raw_pystir.cSTIR_newObject.restype = ctypes.c_void_p
raw_pystir.cSTIR_setParameter.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
raw_pystir.cSTIR_setParameter.restype = ctypes.c_void_p
raw_pystir.cSTIR_parameter.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
raw_pystir.cSTIR_parameter.restype = ctypes.c_void_p
raw_pystir.cSTIR_setupAcquisitionModel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
raw_pystir.cSTIR_setupAcquisitionModel.restype = ctypes.c_void_p
raw_pystir.cSTIR_setupReconstruction.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
raw_pystir.cSTIR_setupReconstruction.restype = ctypes.c_void_p
raw_pystir.cSTIR_updateReconstruction.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
raw_pystir.cSTIR_updateReconstruction.restype = ctypes.c_void_p
raw_pystir.cSTIR_imageFromAcquisitionData.argtypes = [ctypes.c_void_p]
raw_pystir.cSTIR_imageFromAcquisitionData.restype = ctypes.c_void_p
raw_pystir.cSTIR_fillImage.argtypes = [ctypes.c_void_p, ctypes.c_float]
raw_pystir.cSTIR_fillImage.restype = ctypes.c_void_p
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


def delete_handle(handle):
    if handle:
        raw_pyi.deleteDataHandle(handle)


def check_object_handle(handle, label: str):
    if not handle:
        raise RuntimeError(f"{label}: null handle")
    status = raw_pyi.executionStatus(handle)
    if status != 0:
        msg = raw_pyi.executionError(handle)
        text = msg.decode("utf-8", errors="replace") if msg else "unknown engine error"
        raise RuntimeError(f"{label}: {text}")
    return handle


def make_managed_image(factory_lib: Path, source_handle):
    factory = ctypes.CDLL(str(factory_lib), mode=ctypes.RTLD_GLOBAL)
    factory.make_managed_stir_image_handle_from_existing.argtypes = [ctypes.c_void_p]
    factory.make_managed_stir_image_handle_from_existing.restype = ctypes.c_void_p
    handle = factory.make_managed_stir_image_handle_from_existing(source_handle)
    if not handle:
        raise RuntimeError("image factory returned a null managed handle")
    return handle


def make_managed_acquisition(factory_lib: Path, source_handle):
    factory = ctypes.CDLL(str(factory_lib), mode=ctypes.RTLD_GLOBAL)
    factory.make_managed_stir_acquisition_handle_from_existing.argtypes = [ctypes.c_void_p]
    factory.make_managed_stir_acquisition_handle_from_existing.restype = ctypes.c_void_p
    handle = factory.make_managed_stir_acquisition_handle_from_existing(source_handle)
    if not handle:
        raise RuntimeError("acquisition factory returned a null managed handle")
    return handle


def get_flags(handle, kind: bytes):
    is_managed_h = raw_pystir.cSTIR_parameter(handle, kind, b"supports_cuda_array_view")
    cuda_addr_h = raw_pystir.cSTIR_parameter(handle, kind, b"cuda_address")
    is_managed = bool(raw_pyi.intDataFromHandle(is_managed_h))
    cuda_addr = int(raw_pyi.size_tDataFromHandle(cuda_addr_h))
    raw_pyi.deleteDataHandle(is_managed_h)
    raw_pyi.deleteDataHandle(cuda_addr_h)
    return is_managed, cuda_addr


def get_image_array(handle):
    dims = np.zeros(10, dtype=np.int32)
    expect_ok(raw_pystir.cSTIR_getImageDimensions(handle, dims.ctypes.data))
    shape = tuple(int(v) for v in dims[:3])
    data = np.empty(shape, dtype=np.float32)
    expect_ok(raw_pystir.cSTIR_getImageData(handle, data.ctypes.data))
    return data


def set_int_parameter(obj_handle, obj_name: bytes, name: bytes, value: int):
    value_handle = raw_pyi.intDataHandle(value)
    try:
        expect_ok(raw_pystir.cSTIR_setParameter(obj_handle, obj_name, name, value_handle))
    finally:
        raw_pyi.deleteDataHandle(value_handle)


def set_object_parameter(obj_handle, obj_name: bytes, name: bytes, value_handle):
    expect_ok(raw_pystir.cSTIR_setParameter(obj_handle, obj_name, name, value_handle))


def build_reconstructor(acq_handle, image_handle, subsets: int):
    matrix_handle = check_object_handle(raw_pystir.cSTIR_newObject(b"RayTracingMatrix"), "new RayTracingMatrix")
    set_int_parameter(matrix_handle, b"RayTracingMatrix", b"num_tangential_LORs", 2)

    am_handle = check_object_handle(raw_pystir.cSTIR_newObject(b"AcqModUsingMatrix"), "new AcqModUsingMatrix")
    set_object_parameter(am_handle, b"AcqModUsingMatrix", b"matrix", matrix_handle)
    expect_ok(raw_pystir.cSTIR_setupAcquisitionModel(am_handle, acq_handle, image_handle))

    obj_handle = check_object_handle(
        raw_pystir.cSTIR_newObject(b"PoissonLogLikelihoodWithLinearModelForMeanAndProjData"),
        "new PoissonLogLikelihoodWithLinearModelForMeanAndProjData",
    )
    set_object_parameter(obj_handle, b"PoissonLogLikelihoodWithLinearModelForMeanAndProjData", b"acquisition_data", acq_handle)
    set_object_parameter(obj_handle, b"PoissonLogLikelihoodWithLinearModelForMeanAndProjData", b"acquisition_model", am_handle)

    recon_handle = check_object_handle(
        raw_pystir.cSTIR_objectFromFile(b"OSMAPOSLReconstruction", b""),
        "new OSMAPOSLReconstruction",
    )
    set_int_parameter(recon_handle, b"Reconstruction", b"disable_output", 1)
    set_object_parameter(recon_handle, b"IterativeReconstruction", b"objective_function", obj_handle)
    set_int_parameter(recon_handle, b"IterativeReconstruction", b"num_subsets", subsets)
    set_int_parameter(recon_handle, b"IterativeReconstruction", b"num_subiterations", 1)
    set_object_parameter(recon_handle, b"Reconstruction", b"input_data", acq_handle)
    expect_ok(raw_pystir.cSTIR_setupReconstruction(recon_handle, image_handle))

    return matrix_handle, am_handle, obj_handle, recon_handle


def save_panel(normal_arr, managed_arr, error_arr, output_dir: Path):
    mid = normal_arr.shape[0] // 2
    vmax = float(max(np.max(normal_arr[mid]), np.max(managed_arr[mid])))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(normal_arr[mid], cmap="gray", vmin=0.0, vmax=vmax)
    axes[0].set_title("Normal OSMAPOSL")
    axes[1].imshow(managed_arr[mid], cmap="gray", vmin=0.0, vmax=vmax)
    axes[1].set_title("Managed OSMAPOSL")
    im_err = axes[2].imshow(error_arr[mid], cmap="magma")
    axes[2].set_title("Absolute Error")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)
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

    handles = []
    try:
        normal_acq = raw_pystir.cSTIR_objectFromFile(b"AcquisitionData", str(Path(args.acq)).encode("utf-8"))
        handles.append(normal_acq)
        managed_acq = make_managed_acquisition(Path(args.acq_factory), normal_acq)
        handles.append(managed_acq)

        normal_image = raw_pystir.cSTIR_imageFromAcquisitionData(normal_acq)
        handles.append(normal_image)
        expect_ok(raw_pystir.cSTIR_fillImage(normal_image, ctypes.c_float(1.0)))

        managed_image = make_managed_image(Path(args.image_factory), normal_image)
        handles.append(managed_image)

        in_managed, in_addr = get_flags(managed_acq, b"AcquisitionData")
        out_managed_before, out_addr_before = get_flags(managed_image, b"ImageData")

        normal_matrix, normal_am, normal_obj, normal_recon = build_reconstructor(normal_acq, normal_image, args.subsets)
        handles.extend([normal_matrix, normal_am, normal_obj, normal_recon])
        managed_matrix, managed_am, managed_obj, managed_recon = build_reconstructor(managed_acq, managed_image, args.subsets)
        handles.extend([managed_matrix, managed_am, managed_obj, managed_recon])

        expect_ok(raw_pystir.cSTIR_updateReconstruction(normal_recon, normal_image))
        expect_ok(raw_pystir.cSTIR_updateReconstruction(managed_recon, managed_image))

        out_managed_after, out_addr_after = get_flags(managed_image, b"ImageData")

        normal_arr = get_image_array(normal_image)
        managed_arr = get_image_array(managed_image)
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
                "supports_cuda_array_view": in_managed,
                "cuda_address": in_addr,
            },
            "managed_output_before": {
                "supports_cuda_array_view": out_managed_before,
                "cuda_address": out_addr_before,
            },
            "managed_output_after": {
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
    finally:
        for handle in reversed(handles):
            delete_handle(handle)


if __name__ == "__main__":
    main()
