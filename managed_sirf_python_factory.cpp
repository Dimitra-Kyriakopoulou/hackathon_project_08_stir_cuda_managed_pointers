#include "stir/IndexRange3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/cuda_managed_memory.h"
#include "sirf/STIR/stir_data_containers.h"
#include "sirf/common/ImageData.h"
#include "sirf/iUtilities/DataHandle.h"
#include <algorithm>
#include <memory>

namespace {

std::shared_ptr<sirf::ImageData>
make_managed_image_data_from_stir(const sirf::STIRImageData& source)
{
  const auto& source_voxels =
      dynamic_cast<const stir::VoxelsOnCartesianGrid<float>&>(source.data());
  auto managed_voxels_sptr = stir::make_cuda_managed_voxels_on_cartesian_grid_sptr<float>(
      source_voxels.get_index_range(),
      source_voxels.get_origin(),
      source_voxels.get_grid_spacing());
  managed_voxels_sptr->set_exam_info(source.data().get_exam_info());
  std::copy(
      source.data().begin_all(),
      source.data().end_all(),
      managed_voxels_sptr->begin_all());
  sirf::sptrImage3DF image_sptr = managed_voxels_sptr;
  return std::make_shared<sirf::STIRImageData>(image_sptr);
}

} // namespace

extern "C" void*
make_managed_stir_image_handle()
{
  try
    {
      const stir::IndexRange3D range(2, 3, 4);
      auto voxels_sptr =
          stir::make_cuda_managed_voxels_on_cartesian_grid_sptr<float>(range);
      sirf::sptrImage3DF image_sptr = voxels_sptr;
      std::shared_ptr<sirf::ImageData> image_data =
          std::make_shared<sirf::STIRImageData>(image_sptr);
      return newObjectHandle<sirf::ImageData>(image_data);
    }
  CATCH;
}

extern "C" void*
make_managed_stir_image_handle_from_file(const char* filename)
{
  try
    {
      const sirf::STIRImageData source{std::string(filename)};
      return newObjectHandle<sirf::ImageData>(make_managed_image_data_from_stir(source));
    }
  CATCH;
}

extern "C" void*
make_managed_stir_image_handle_from_existing(void* existing_handle)
{
  try
    {
      auto& source = objectFromHandle<sirf::STIRImageData>(existing_handle);
      return newObjectHandle<sirf::ImageData>(make_managed_image_data_from_stir(source));
    }
  CATCH;
}
