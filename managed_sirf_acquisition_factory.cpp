#include "stir/cuda_managed_memory.h"
#include "stir/ExamInfo.h"
#include "stir/IndexRange.h"
#include "sirf/STIR/stir_data_containers.h"
#include "sirf/iUtilities/DataHandle.h"
#include <memory>
#include <vector>

namespace {

std::shared_ptr<sirf::STIRAcquisitionData>
make_managed_acquisition_data(const char* scanner_name)
{
  stir::shared_ptr<stir::ExamInfo> exam_info_sptr(new stir::ExamInfo);
  stir::shared_ptr<stir::ProjDataInfo> proj_data_info_sptr =
      sirf::STIRAcquisitionData::proj_data_info_from_scanner(
          scanner_name, 1, 0, 8);
  const std::size_t num_elems = stir::proj_data_num_elems(*proj_data_info_sptr);
  stir::Array<1, float> buffer(
      stir::IndexRange<1>(0, static_cast<int>(num_elems) - 1),
      stir::make_cuda_managed_shared_array<float>(num_elems));
  for (std::size_t i = 0; i < num_elems; ++i)
    buffer[static_cast<int>(i)] = static_cast<float>(i);

  std::unique_ptr<stir::ProjData> proj_data_uptr(
      new stir::ProjDataInMemory(exam_info_sptr, proj_data_info_sptr, std::move(buffer)));
  return std::shared_ptr<sirf::STIRAcquisitionData>(
      new sirf::STIRAcquisitionDataInMemory(std::move(proj_data_uptr)));
}

std::shared_ptr<sirf::STIRAcquisitionData>
make_managed_acquisition_data_from_source(const sirf::STIRAcquisitionData& source)
{
  auto exam_info_sptr = source.data()->get_exam_info_sptr();
  auto proj_data_info_sptr = source.data()->get_proj_data_info_sptr()->create_shared_clone();
  const std::size_t num_elems = stir::proj_data_num_elems(*proj_data_info_sptr);
  stir::Array<1, float> buffer(
      stir::IndexRange<1>(0, static_cast<int>(num_elems) - 1),
      stir::make_cuda_managed_shared_array<float>(num_elems));

  std::unique_ptr<stir::ProjData> proj_data_uptr(
      new stir::ProjDataInMemory(exam_info_sptr, proj_data_info_sptr, std::move(buffer)));
  auto managed = std::shared_ptr<sirf::STIRAcquisitionData>(
      new sirf::STIRAcquisitionDataInMemory(std::move(proj_data_uptr)));
  managed->fill(source);
  return managed;
}

} // namespace

extern "C" void*
make_managed_stir_acquisition_handle()
{
  try
    {
      return newObjectHandle<sirf::STIRAcquisitionData>(
          make_managed_acquisition_data("Siemens_mMR"));
    }
  CATCH;
}

extern "C" void*
make_managed_stir_acquisition_handle_from_existing(void* existing_handle)
{
  try
    {
      auto& source = objectFromHandle<sirf::STIRAcquisitionData>(existing_handle);
      return newObjectHandle<sirf::STIRAcquisitionData>(
          make_managed_acquisition_data_from_source(source));
    }
  CATCH;
}
