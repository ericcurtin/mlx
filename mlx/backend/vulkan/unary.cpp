// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/unary.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/common/unary.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace vulkan {

void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s) {
  auto& in = inputs[0];
  if (in.size() == 0) {
    return;
  }

  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);

  // Get the kernel for this operation
  auto kernel_info = get_unary_kernel(op, in.dtype(), out.dtype());
  
  // Set up buffers
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Calculate dispatch dimensions
  uint32_t num_elements = static_cast<uint32_t>(in.data_size());
  uint32_t workgroup_count = div_ceil(num_elements, WORKGROUP_SIZE_1D);

  // Dispatch the compute shader
  encoder.dispatch(
      kernel_info.pipeline,
      kernel_info.layout,
      workgroup_count);
}

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op) {
  set_unary_output_data(inputs[0], out);
  auto& s = out.primitive().stream();
  unary_op_gpu_inplace(inputs, out, op, s);
}

} // namespace vulkan

#define UNARY_GPU(func)                                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    vulkan::unary_op_gpu(inputs, out, name());                        \
  }

UNARY_GPU(Abs)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcTan)
UNARY_GPU(ArcTanh)
UNARY_GPU(Ceil)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Floor)
UNARY_GPU(Imag)
UNARY_GPU(Log)
UNARY_GPU(Log2)
UNARY_GPU(Log10)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Negative)
UNARY_GPU(Real)
UNARY_GPU(Round)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Sqrt)
UNARY_GPU(Square)
UNARY_GPU(Tan)
UNARY_GPU(Tanh)

} // namespace mlx::core
