#ifndef __stir_cuda_managed_memory_H__
#define __stir_cuda_managed_memory_H__
/*
    Copyright (C) 2026
*/

#include "stir/common.h"

#ifdef STIR_WITH_CUDA

#include "stir/Array.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/error.h"
#include "stir/shared_ptr.h"
#include <cuda_runtime.h>
#include <utility>

START_NAMESPACE_STIR

template <typename elemT>
inline shared_ptr<elemT[]>
make_cuda_managed_shared_array(const size_t num_elems)
{
  elemT* raw = nullptr;
  const cudaError_t cuda_error =
      cudaMallocManaged(reinterpret_cast<void**>(&raw), num_elems * sizeof(elemT));
  if (cuda_error != cudaSuccess)
    error(std::string("make_cuda_managed_shared_array failed: ") +
          cudaGetErrorString(cuda_error));
  return shared_ptr<elemT[]>(raw, [](elemT* ptr) {
    if (ptr != nullptr)
      cudaFree(ptr);
  });
}

template <int num_dimensions, typename elemT>
inline Array<num_dimensions, elemT>
make_cuda_managed_array(const IndexRange<num_dimensions>& range)
{
  Array<num_dimensions, elemT> array(
      range, make_cuda_managed_shared_array<elemT>(range.size_all()));
  std::fill(array.begin_all(), array.end_all(), elemT(0));
  return array;
}

template <typename elemT>
inline VoxelsOnCartesianGrid<elemT>
make_cuda_managed_voxels_on_cartesian_grid(
    const IndexRange<3>& range,
    const CartesianCoordinate3D<float>& origin =
        CartesianCoordinate3D<float>(0.F, 0.F, 0.F),
    const BasicCoordinate<3, float>& grid_spacing =
        CartesianCoordinate3D<float>(1.F, 1.F, 1.F))
{
  return VoxelsOnCartesianGrid<elemT>(
      make_cuda_managed_array<3, elemT>(range), origin, grid_spacing);
}

template <typename elemT>
inline shared_ptr<VoxelsOnCartesianGrid<elemT>>
make_cuda_managed_voxels_on_cartesian_grid_sptr(
    const IndexRange<3>& range,
    const CartesianCoordinate3D<float>& origin =
        CartesianCoordinate3D<float>(0.F, 0.F, 0.F),
    const BasicCoordinate<3, float>& grid_spacing =
        CartesianCoordinate3D<float>(1.F, 1.F, 1.F))
{
  auto voxels =
      make_cuda_managed_voxels_on_cartesian_grid<elemT>(range, origin, grid_spacing);
  return shared_ptr<VoxelsOnCartesianGrid<elemT>>(
      new VoxelsOnCartesianGrid<elemT>(std::move(voxels)));
}

END_NAMESPACE_STIR

#endif

#endif
