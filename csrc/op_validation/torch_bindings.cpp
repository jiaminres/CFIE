#include "core/registration.h"

#include <torch/library.h>

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "gptq_marlin_fp8_dequantize(Tensor qweight, Tensor scales, "
      "SymInt size_k, SymInt size_n, int group_size, bool transpose) -> Tensor");
  m.def(
      "gptq_marlin_fp8_bwd_input(Tensor grad_output_fp8, "
      "Tensor grad_output_scales, Tensor qweight_fwd, Tensor scales_fwd, "
      "SymInt size_k, SymInt size_n, int group_size) -> Tensor");
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
