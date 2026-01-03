// Copyright © 2025 Apple Inc.
//
// This file contains implementations for primitives that are not covered by
// the unary.cpp and binary.cpp files, such as reductions, matrix operations,
// data movement, and other specialized operations.

#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/utils.h"
#include "mlx/backend/common/utils.h"
#include "mlx/allocator.h"
#include "mlx/primitives.h"

#include <stdexcept>

namespace mlx::core {

namespace vulkan {

// Helper to copy data between arrays
void copy_gpu(const array& src, array& dst, const Stream& s) {
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  // For contiguous arrays, use simple copy
  if (src.flags().contiguous && dst.flags().contiguous) {
    auto src_buf = static_cast<VulkanBuffer*>(src.buffer().ptr());
    auto dst_buf = static_cast<VulkanBuffer*>(dst.buffer().ptr());
    encoder.copy_buffer(src_buf->buffer, dst_buf->buffer, 
                        static_cast<VkDeviceSize>(src.nbytes()));
  }
}

// Reduction implementation
void reduce_gpu(
    const array& in,
    array& out,
    const char* op,
    const std::vector<int>& axes,
    const Stream& s) {
  
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  auto kernel_info = get_reduce_kernel(op, in.dtype(), out.dtype());
  
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Calculate reduction size
  size_t reduce_size = 1;
  for (int axis : axes) {
    reduce_size *= in.shape()[axis];
  }
  
  uint32_t output_size = static_cast<uint32_t>(out.size());
  uint32_t workgroup_count = output_size;
  
  encoder.dispatch(kernel_info.pipeline, kernel_info.layout, workgroup_count);
}

// Matmul implementation
void matmul_gpu(
    const array& a,
    const array& b,
    array& out,
    bool transpose_a,
    bool transpose_b,
    const Stream& s) {
  
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  auto kernel_info = get_matmul_kernel(a.dtype(), out.dtype(), transpose_a, transpose_b);
  
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  
  // Get dimensions
  int ndim = out.ndim();
  uint32_t M = out.shape()[ndim - 2];
  uint32_t N = out.shape()[ndim - 1];
  uint32_t K = transpose_a ? a.shape()[ndim - 2] : a.shape()[ndim - 1];
  
  // Calculate workgroup counts
  uint32_t grid_x = div_ceil(N, WORKGROUP_SIZE_2D);
  uint32_t grid_y = div_ceil(M, WORKGROUP_SIZE_2D);
  uint32_t batch_size = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batch_size *= out.shape()[i];
  }
  
  encoder.dispatch(kernel_info.pipeline, kernel_info.layout, grid_x, grid_y, batch_size);
}

// Softmax implementation
void softmax_gpu(const array& in, array& out, const Stream& s) {
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  auto kernel_info = get_softmax_kernel(in.dtype());
  
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Last dimension is the softmax dimension
  uint32_t batch_size = static_cast<uint32_t>(in.size() / in.shape().back());
  
  encoder.dispatch(kernel_info.pipeline, kernel_info.layout, batch_size);
}

// Layer norm implementation
void layer_norm_gpu(
    const array& in,
    const array& weight,
    const array& bias,
    array& out,
    float eps,
    const Stream& s) {
  
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  auto kernel_info = get_norm_kernel("layer_norm", in.dtype());
  
  encoder.set_input_array(in);
  encoder.set_input_array(weight);
  encoder.set_input_array(bias);
  encoder.set_output_array(out);
  
  uint32_t batch_size = static_cast<uint32_t>(in.size() / in.shape().back());
  
  encoder.dispatch(kernel_info.pipeline, kernel_info.layout, batch_size);
}

// RMS norm implementation
void rms_norm_gpu(
    const array& in,
    const array& weight,
    array& out,
    float eps,
    const Stream& s) {
  
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  auto kernel_info = get_norm_kernel("rms_norm", in.dtype());
  
  encoder.set_input_array(in);
  encoder.set_input_array(weight);
  encoder.set_output_array(out);
  
  uint32_t batch_size = static_cast<uint32_t>(in.size() / in.shape().back());
  
  encoder.dispatch(kernel_info.pipeline, kernel_info.layout, batch_size);
}

// Attention implementation
void attention_gpu(
    const array& queries,
    const array& keys,
    const array& values,
    const std::optional<array>& mask,
    array& out,
    float scale,
    const Stream& s) {
  
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }
  
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  bool use_mask = mask.has_value();
  auto kernel_info = get_attention_kernel(queries.dtype(), use_mask);
  
  encoder.set_input_array(queries);
  encoder.set_input_array(keys);
  encoder.set_input_array(values);
  if (use_mask) {
    encoder.set_input_array(mask.value());
  }
  encoder.set_output_array(out);
  
  // Dimensions: [batch, num_heads, seq_len, head_dim]
  int ndim = out.ndim();
  uint32_t batch_size = out.shape()[0];
  uint32_t num_heads = out.shape()[1];
  uint32_t seq_len = out.shape()[2];
  
  encoder.dispatch(kernel_info.pipeline, kernel_info.layout, seq_len, num_heads, batch_size);
}

// RoPE implementation
void rope_gpu(
    array& data,
    const array& freqs_cos,
    const array& freqs_sin,
    uint32_t offset,
    const Stream& s) {
  
  if (data.size() == 0) {
    return;
  }
  
  auto& device = Device::instance();
  auto& encoder = device.get_command_encoder(s);
  
  auto kernel_info = get_rope_kernel(data.dtype());
  
  encoder.set_input_array(data);
  encoder.set_input_array(freqs_cos);
  encoder.set_input_array(freqs_sin);
  
  uint32_t total_elements = static_cast<uint32_t>(data.size());
  uint32_t workgroup_count = div_ceil(total_elements / 2, WORKGROUP_SIZE_1D);
  
  encoder.dispatch(kernel_info.pipeline, kernel_info.layout, workgroup_count);
}

} // namespace vulkan

// Note: Unary operations (Abs, Exp, Log, etc.) are implemented in unary.cpp
// Note: Binary operations (Add, Multiply, etc.) are implemented in binary.cpp

// Reduction operations
void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  vulkan::reduce_gpu(inputs[0], out, reduce_type_name(reduce_type_).c_str(), 
                     axes_, stream());
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  // ArgReduce requires special handling - fallback to CPU for now
  eval(inputs, out);
}

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Scan (cumsum, etc.) - fallback to CPU for now
  eval(inputs, out);
}

// Matrix operations
void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  vulkan::matmul_gpu(inputs[0], inputs[1], out, false, false, stream());
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // AddMM = alpha * (A @ B) + beta * C - implement with matmul + add
  // For now, fallback to CPU
  eval(inputs, out);
}

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

// Convolution
void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Convolution requires specialized kernels - fallback to CPU
  eval(inputs, out);
}

// Attention
void ScaledDotProductAttention::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  std::optional<array> mask;
  if (inputs.size() > 3) {
    mask = inputs[3];
  }
  vulkan::attention_gpu(inputs[0], inputs[1], inputs[2], mask, out, scale_, stream());
}

// Normalization
void LayerNorm::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  vulkan::layer_norm_gpu(inputs[0], inputs[1], inputs[2], out, eps_, stream());
}

void RMSNorm::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  vulkan::rms_norm_gpu(inputs[0], inputs[1], out, eps_, stream());
}

void GroupNorm::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Group norm - fallback to CPU
  eval(inputs, out);
}

void RoPE::eval_gpu(const std::vector<array>& inputs, array& out) {
  // RoPE operates in-place on a copy
  auto& in = inputs[0];
  if (in.is_donatable()) {
    out.copy_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
    vulkan::copy_gpu(in, out, stream());
  }
  vulkan::rope_gpu(out, inputs[1], inputs[2], static_cast<uint32_t>(offset_), stream());
}

// Data movement operations - many can delegate to eval() which does the right thing
void Copy::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Reshape::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Transpose::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For simple transposes, we can use GPU. Complex cases fall back to CPU.
  eval(inputs, out);
}

void Broadcast::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void BroadcastTo::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Concatenate::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Concatenate can be done on GPU but for now fallback
  eval(inputs, out);
}

void Pad::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Split::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Slice::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void SliceUpdate::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Squeeze::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void ExpandDims::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Take::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void TakeAlongAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

// Type conversion
void AsType::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Type conversion can be done with copy kernel - for now fallback
  eval(inputs, out);
}

void View::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void AsStrided::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

// Creation operations
void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Arange can be done on GPU with a simple kernel
  // For now, fallback to CPU
  eval(inputs, out);
}

void Full::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Full/fill can use the fill shader
  eval(inputs, out);
}

void NumberOfElements::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

// Random
void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Random number generation - fallback to CPU for now
  eval(inputs, out);
}

// Sorting
void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  // GPU sorting requires bitonic sort or similar - fallback to CPU
  eval(inputs, out);
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

// Linear algebra - these are complex and typically use libraries like LAPACK
void QRF::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void SVD::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Inverse::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Cholesky::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Eig::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Eigh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

// Quantization
void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Quantized matmul requires dequantization + matmul
  eval(inputs, out);
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void AffineQuantize::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

// Ternary
void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Select/where can be done on GPU
  eval(inputs, out);
}

void Where::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

// FFT
void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  // FFT requires specialized implementation - fallback to CPU
  eval(inputs, out);
}

// Softmax
void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  vulkan::softmax_gpu(inputs[0], out, stream());
}

// Misc
void StopGradient::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

void Compiled::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  // Compiled kernels - not supported in Vulkan backend yet
  eval(inputs, outputs);
}

void Depends::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  eval(inputs, outputs);
}

void Load::eval_gpu(const std::vector<array>& inputs, array& out) {
  eval(inputs, out);
}

}  // namespace mlx::core
