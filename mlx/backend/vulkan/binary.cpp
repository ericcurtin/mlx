// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/binary.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/common/binary.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace vulkan {

void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const char* op,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& out = outputs[0];
  
  if (out.size() == 0) {
    return;
  }

  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);

  // Get the kernel for this operation
  auto kernel_info = get_binary_kernel(op, a.dtype(), out.dtype());
  
  // Set up buffers
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  // Calculate dispatch dimensions
  uint32_t num_elements = static_cast<uint32_t>(out.data_size());
  uint32_t workgroup_count = div_ceil(num_elements, WORKGROUP_SIZE_1D);

  // Dispatch the compute shader
  encoder.dispatch(
      kernel_info.pipeline,
      kernel_info.layout,
      workgroup_count);
}

void binary_op_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const char* op) {
  set_binary_output_data(inputs, outputs);
  auto& s = outputs[0].primitive().stream();
  binary_op_gpu_inplace(inputs, outputs, op, s);
}

void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op) {
  std::vector<array> outputs = {out};
  binary_op_gpu(inputs, outputs, op);
}

} // namespace vulkan

#define BINARY_GPU(func)                                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    vulkan::binary_op_gpu(inputs, out, name());                       \
  }

#define BINARY_GPU_MULTI(func)                                         \
  void func::eval_gpu(                                                 \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    vulkan::binary_op_gpu(inputs, outputs, name());                    \
  }

BINARY_GPU(Add)
BINARY_GPU(ArcTan2)
BINARY_GPU(Divide)
BINARY_GPU(Equal)
BINARY_GPU(Greater)
BINARY_GPU(GreaterEqual)
BINARY_GPU(Less)
BINARY_GPU(LessEqual)
BINARY_GPU(LogAddExp)
BINARY_GPU(LogicalAnd)
BINARY_GPU(LogicalOr)
BINARY_GPU(Maximum)
BINARY_GPU(Minimum)
BINARY_GPU(Multiply)
BINARY_GPU(NaNEqual)
BINARY_GPU(NotEqual)
BINARY_GPU(Power)
BINARY_GPU(Remainder)
BINARY_GPU(Subtract)

BINARY_GPU_MULTI(DivMod)
BINARY_GPU_MULTI(BitwiseBinary)

} // namespace mlx::core
