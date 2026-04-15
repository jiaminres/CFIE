#pragma once

#include <Python.h>

// Token concatenation helpers.
#define CFIE_CONCAT_IMPL(A, B) A##B
#define CONCAT(A, B) CFIE_CONCAT_IMPL(A, B)

// Stringification helpers.
#define CFIE_STRINGIFY_IMPL(A) #A
#define STRINGIFY(A) CFIE_STRINGIFY_IMPL(A)

// Expand NAME before forwarding into the PyTorch registration macro.
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// Expand NAME before forwarding into the PyTorch impl registration macro.
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

// Export the Python extension init symbol.
#define REGISTER_EXTENSION(NAME)                                            \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                  \
    static struct PyModuleDef module = {                                    \
        PyModuleDef_HEAD_INIT,                                              \
        STRINGIFY(NAME),                                                    \
        nullptr,                                                            \
        0,                                                                  \
        nullptr,                                                            \
    };                                                                      \
    return PyModule_Create(&module);                                        \
  }
