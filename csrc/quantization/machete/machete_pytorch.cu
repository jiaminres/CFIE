#include "machete_mm_launcher.cuh"
#include "machete_prepack_launcher.cuh"
#include "core/scalar_type.hpp"

#include "core/registration.h"

namespace machete {

using namespace vllm;

std::vector<std::string> supported_schedules(
    at::ScalarType a_type, int64_t b_type_id,
    std::optional<at::ScalarType> maybe_group_scales_type,
    std::optional<at::ScalarType> maybe_group_zeros_type,
    std::optional<at::ScalarType> maybe_channel_scales_type,
    std::optional<at::ScalarType> maybe_token_scales_type,
    std::optional<at::ScalarType> maybe_out_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  SupportedSchedulesArgs const args{
      a_type,
      b_type,
      maybe_group_scales_type,
      maybe_group_zeros_type,
      maybe_channel_scales_type,
      maybe_token_scales_type,
      maybe_out_type,
  };
  return supported_schedules_dispatch(args);
}

torch::Tensor mm(torch::Tensor const& A, torch::Tensor const& B,
                 int64_t b_type_id,
                 std::optional<at::ScalarType> const& maybe_out_type,
                 std::optional<torch::Tensor> const& maybe_group_scales,
                 std::optional<torch::Tensor> const& maybe_group_zeros,
                 std::optional<int64_t> maybe_group_size,
                 std::optional<torch::Tensor> const& maybe_channel_scales,
                 std::optional<torch::Tensor> const& maybe_token_scales,
                 std::optional<std::string> maybe_schedule) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  MMArgs const args{
      A,
      B,
      b_type,
      maybe_out_type,
      maybe_group_scales,
      maybe_group_zeros,
      maybe_group_size,
      maybe_channel_scales,
      maybe_token_scales,
      maybe_schedule,
  };
  return mm_dispatch(args);
}

torch::Tensor prepack_B(
    torch::Tensor const& B, at::ScalarType const& a_type, int64_t b_type_id,
    std::optional<at::ScalarType> const& maybe_group_scales_type) {
  ScalarType const b_type = ScalarType::from_id(b_type_id);
  PrepackBArgs const args{
      B,
      a_type,
      b_type,
      maybe_group_scales_type,
  };
  return prepack_B_dispatch(args);
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("machete_prepack_B", &prepack_B);
  m.impl("machete_mm", &mm);
}

// use CatchAll since supported_schedules has no tensor arguments
TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CatchAll, m) {
  m.impl("machete_supported_schedules", &supported_schedules);
}

};  // namespace machete
