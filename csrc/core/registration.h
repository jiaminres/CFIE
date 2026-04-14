#pragma once

#include <Python.h>

// ------------------------------- 定义基础的 token 拼接宏 -------------------------------
// 使用预处理器的 ## 操作符，将两个 token 在预处理阶段直接拼接成一个新 token。
#define _CONCAT(A, B) A##B

// 对 _CONCAT 再包一层，确保如果 A 或 B 本身是宏时，也会先展开再拼接。
#define CONCAT(A, B) _CONCAT(A, B)

// ------------------------------- 定义基础的字符串化宏 -------------------------------
// 使用预处理器的 # 操作符，将参数直接转换为字符串字面量。
#define _STRINGIFY(A) #A

// 对 _STRINGIFY 再包一层，确保如果 A 本身是宏时，也会先展开再转成字符串。
#define STRINGIFY(A) _STRINGIFY(A)

// ------------------------------- 定义可展开 NAME 的 TORCH_LIBRARY 包装宏 -------------------------------
// 对 TORCH_LIBRARY 做一层轻量包装，使 NAME 参数即使是宏，也能先展开后再传给 TORCH_LIBRARY。
#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

// ------------------------------- 定义可展开 NAME 的 TORCH_LIBRARY_IMPL 包装宏 -------------------------------
// 对 TORCH_LIBRARY_IMPL 做一层轻量包装，使 NAME 参数即使是宏，也能先展开后再传给 TORCH_LIBRARY_IMPL。
#define TORCH_LIBRARY_IMPL_EXPAND(NAME, DEVICE, MODULE) \
  TORCH_LIBRARY_IMPL(NAME, DEVICE, MODULE)

// ------------------------------- 定义 Python 扩展模块初始化入口注册宏 -------------------------------
// 为当前共享库生成一个符合 Python C 扩展规范的模块初始化函数，
// 使该动态库可以通过 Python 的 import 语句被加载和初始化。
#define REGISTER_EXTENSION(NAME)                                               \
  /* 定义 Python 扩展模块初始化函数，函数名固定为 PyInit_<模块名>。 */            \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    /* 构造一个最小化的 Python 模块定义对象。 */                                \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        /* 将模块名写入模块定义中。 */           \
                                        STRINGIFY(NAME),                       \
                                        /* 当前模块不提供文档字符串。 */         \
                                        nullptr,                               \
                                        /* 当前模块不使用额外的模块状态空间。 */ \
                                        0,                                     \
                                        /* 当前模块不在这里注册 Python 方法表。 */\
                                        nullptr};                              \
    /* 基于上述模块定义对象创建并返回 Python 模块实例。 */                       \
    return PyModule_Create(&module);                                           \
  }
