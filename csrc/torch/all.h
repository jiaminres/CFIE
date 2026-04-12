#pragma once

// Keep the local extension build on the lean Tensor/ATen surface instead of
// pulling the full C++ frontend umbrella, which drags in autograd headers that
// currently miscompile under nvcc+MSVC on Windows.
#include <torch/types.h>
#include <torch/cuda.h>
#include <torch/nn/functional/padding.h>
