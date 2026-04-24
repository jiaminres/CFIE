#include <torch/all.h>
#include <torch/cuda.h>
#include <cuda_runtime.h>

torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor) {
  // ------------------------------- 把 CPU Tensor 映射成 CUDA 可访问视图 -------------------------------
  // 只允许从 CPU tensor 建立 UVA 视图。
  TORCH_CHECK(cpu_tensor.device().is_cpu(), "Input tensor must be on CPU");

  // 空 tensor 直接返回同形状的 CUDA 空张量。
  if (cpu_tensor.numel() == 0) {
    return torch::empty(cpu_tensor.sizes(),
                        cpu_tensor.options().device(torch::kCUDA));
  }

  // 已经是 pinned memory 时，可以直接查询对应的 device pointer。
  if (cpu_tensor.is_pinned()) {
    // host_ptr: 指向 pinned CPU backing storage。
    void* host_ptr = const_cast<void*>(cpu_tensor.data_ptr());
    // device_ptr: 指向同一块 host memory 的 CUDA 侧地址。
    void* device_ptr = nullptr;
    // 让 CUDA 为这块 host memory 返回可访问的 device pointer。
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    // 映射失败时直接终止，避免生成无效的 CUDA 视图。
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));

    // 用 device_ptr 包装出 CUDA tensor view，并绑定原 CPU tensor 的生命周期。
    return torch::from_blob(
        device_ptr, cpu_tensor.sizes(), cpu_tensor.strides(),
        [base = cpu_tensor](void*) {},  // 保持原 CPU tensor 存活。
        cpu_tensor.options().device(torch::kCUDA));
  }

  // 非 pinned memory 先转成连续 CPU tensor，便于后续整块复制。
  torch::Tensor contiguous_cpu = cpu_tensor.contiguous();
  // nbytes: 连续副本需要复制的总字节数。
  size_t nbytes = contiguous_cpu.nbytes();

  // host_ptr: 新申请的 mapped pinned memory 起始地址。
  void* host_ptr = nullptr;
  // 先申请一块可映射到 CUDA 的 pinned host memory。
  cudaError_t err = cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    // 申请失败时直接报错，避免继续使用空指针。
    AT_ERROR("cudaHostAlloc failed: ", cudaGetErrorString(err));
  }

  // 把连续 CPU 副本整块拷入新申请的 mapped pinned memory。
  err = cudaMemcpy(host_ptr, contiguous_cpu.data_ptr(), nbytes,
                   cudaMemcpyDefault);
  if (err != cudaSuccess) {
    // 拷贝失败时先释放 host_ptr，再抛出异常。
    cudaFreeHost(host_ptr);
    AT_ERROR("cudaMemcpy failed: ", cudaGetErrorString(err));
  }

  // device_ptr: 映射后对应的 CUDA 侧地址。
  void* device_ptr = nullptr;
  // 让 CUDA 返回这块 mapped host memory 的 device pointer。
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  if (err != cudaSuccess) {
    // 映射失败时同样先释放 host_ptr。
    cudaFreeHost(host_ptr);
    AT_ERROR("cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
  }

  // deleter: 视图销毁时负责回收这块 host memory。
  auto deleter = [host_ptr](void*) { cudaFreeHost(host_ptr); };

  // 用 mapped host memory 包装出最终的 CUDA tensor view。
  return torch::from_blob(device_ptr, contiguous_cpu.sizes(),
                          contiguous_cpu.strides(), deleter,
                          contiguous_cpu.options().device(torch::kCUDA));
}
