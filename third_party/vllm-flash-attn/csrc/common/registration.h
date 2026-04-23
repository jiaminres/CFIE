#pragma once

#include <Python.h>

#define VLLM_FA_CONCAT_IMPL(A, B) A##B
#define VLLM_FA_CONCAT(A, B) VLLM_FA_CONCAT_IMPL(A, B)

#define VLLM_FA_STRINGIFY_IMPL(A) #A
#define VLLM_FA_STRINGIFY(A) VLLM_FA_STRINGIFY_IMPL(A)

// A version of the TORCH_LIBRARY macro that expands the NAME, i.e. so NAME
// could be a macro instead of a literal token.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// REGISTER_EXTENSION allows the shared library to be loaded and initialized
// via python's import statement.
#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC VLLM_FA_CONCAT(PyInit_, NAME)() {                             \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        VLLM_FA_STRINGIFY(NAME), nullptr, 0,   \
                                        nullptr};                              \
    return PyModule_Create(&module);                                           \
  }
