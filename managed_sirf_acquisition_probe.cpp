#include "stir/cuda_managed_memory.h"
#include "stir/ExamInfo.h"
#include "sirf/STIR/stir_data_containers.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>

namespace {

bool pointer_is_cuda_managed(const void* ptr)
{
  if (ptr == nullptr)
    return false;
  cudaPointerAttributes attrs{};
  const cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess)
    {
      cudaGetLastError();
      return false;
    }
#if CUDART_VERSION >= 10000
  return attrs.type == cudaMemoryTypeManaged;
#else
  return attrs.memoryType == cudaMemoryTypeManaged && attrs.isManaged;
#endif
}

template <class Container>
const void* address_of_first(const Container& container)
{
  const auto iter = container.begin_all();
  if (iter == container.end_all())
    return nullptr;
  return &*iter;
}

} // namespace

int main()
{
  stir::shared_ptr<stir::ExamInfo> exam_info_sptr(new stir::ExamInfo);
  stir::shared_ptr<stir::ProjDataInfo> proj_data_info_sptr =
      sirf::STIRAcquisitionData::proj_data_info_from_scanner("Siemens_mMR", 1, 0, 8);
  const std::size_t num_elems = stir::proj_data_num_elems(*proj_data_info_sptr);

  stir::Array<1, float> buffer(
      stir::IndexRange<1>(0, static_cast<int>(num_elems) - 1),
      stir::make_cuda_managed_shared_array<float>(num_elems));
  const void* buffer_ptr = address_of_first(buffer);
  std::cout << "buffer_managed " << pointer_is_cuda_managed(buffer_ptr) << std::endl;

  std::unique_ptr<stir::ProjData> proj_data_uptr(
      new stir::ProjDataInMemory(exam_info_sptr, proj_data_info_sptr, std::move(buffer)));
  auto* raw_proj = dynamic_cast<stir::ProjDataInMemory*>(proj_data_uptr.get());
  if (raw_proj == nullptr)
    {
      std::cerr << "raw proj cast failed\n";
      return 1;
    }
  const void* proj_ptr = address_of_first(*raw_proj);
  std::cout << "projdata_managed " << pointer_is_cuda_managed(proj_ptr) << std::endl;

  sirf::STIRAcquisitionDataInMemory ad(std::move(proj_data_uptr));
  const auto* wrapped_proj = dynamic_cast<const stir::ProjDataInMemory*>(ad.data().get());
  if (wrapped_proj == nullptr)
    {
      std::cerr << "wrapped proj cast failed\n";
      return 1;
    }
  const void* wrapped_ptr = address_of_first(*wrapped_proj);
  std::cout << "wrapped_proj_managed " << pointer_is_cuda_managed(wrapped_ptr) << std::endl;
  std::cout << "wrapped_proj_address " << reinterpret_cast<std::uintptr_t>(wrapped_ptr) << std::endl;

  const auto sirf_addr = ad.address();
  const void* sirf_ptr = reinterpret_cast<const void*>(sirf_addr);
  std::cout << "sirf_address " << sirf_addr << std::endl;
  std::cout << "sirf_address_managed_localcheck " << pointer_is_cuda_managed(sirf_ptr) << std::endl;
  std::cout << "sirf_address_matches_wrapped "
            << (reinterpret_cast<std::uintptr_t>(wrapped_ptr) == sirf_addr) << std::endl;
  std::cout << "sirf_supports_cuda_array_view " << ad.supports_cuda_array_view() << std::endl;
  return 0;
}
