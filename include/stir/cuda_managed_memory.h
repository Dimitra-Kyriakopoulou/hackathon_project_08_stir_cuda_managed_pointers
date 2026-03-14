#ifndef __stir_cuda_managed_memory_H__
#define __stir_cuda_managed_memory_H__
/*
    Copyright (C) 2026
    Prototype helper for Project 8 hackathon work.
*/

#include "stir/common.h"

#ifdef STIR_WITH_CUDA

#include "stir/Array.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/IndexRange.h"
#include "stir/ProjDataInMemory.h"
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

inline std::size_t
proj_data_num_elems(const ProjDataInfo& proj_data_info)
{
  std::size_t num_3d = 0;
  for (int segment_num = proj_data_info.get_min_segment_num();
       segment_num <= proj_data_info.get_max_segment_num();
       ++segment_num)
    {
      num_3d += static_cast<std::size_t>(proj_data_info.get_num_axial_poss(segment_num))
                * static_cast<std::size_t>(proj_data_info.get_num_views())
                * static_cast<std::size_t>(proj_data_info.get_num_tangential_poss());
    }
  return num_3d * static_cast<std::size_t>(proj_data_info.get_num_tof_poss());
}

inline shared_ptr<ProjDataInMemory>
make_cuda_managed_proj_data_in_memory_sptr(
    const shared_ptr<const ExamInfo>& exam_info_sptr,
    const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
{
  const std::size_t num_elems = proj_data_num_elems(*proj_data_info_sptr);
  Array<1, float> buffer(
      IndexRange<1>(0, static_cast<int>(num_elems) - 1),
      make_cuda_managed_shared_array<float>(num_elems));
  std::fill(buffer.begin_all(), buffer.end_all(), 0.F);
  return shared_ptr<ProjDataInMemory>(
      new ProjDataInMemory(exam_info_sptr, proj_data_info_sptr, std::move(buffer)));
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
