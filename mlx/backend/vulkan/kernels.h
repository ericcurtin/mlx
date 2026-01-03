// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/dtype.h"
#include <vulkan/vulkan.h>
#include <string>

namespace mlx::core::vulkan {

// Kernel information structure
struct KernelInfo {
  VkPipeline pipeline;
  VkPipelineLayout layout;
  VkDescriptorSetLayout descriptor_layout;
};

// Get or create unary kernel
KernelInfo get_unary_kernel(
    const char* op,
    Dtype in_dtype,
    Dtype out_dtype);

// Get or create binary kernel
KernelInfo get_binary_kernel(
    const char* op,
    Dtype in_dtype,
    Dtype out_dtype);

// Get or create reduction kernel
KernelInfo get_reduce_kernel(
    const char* op,
    Dtype in_dtype,
    Dtype out_dtype);

// Get or create matmul kernel
KernelInfo get_matmul_kernel(
    Dtype in_dtype,
    Dtype out_dtype,
    bool transpose_a,
    bool transpose_b);

// Get or create softmax kernel
KernelInfo get_softmax_kernel(Dtype dtype);

// Get or create normalization kernel
KernelInfo get_norm_kernel(
    const char* norm_type,
    Dtype dtype);

// Get or create attention kernel
KernelInfo get_attention_kernel(Dtype dtype, bool use_mask);

// Get or create RoPE kernel
KernelInfo get_rope_kernel(Dtype dtype);

// Map operation name to shader name
std::string op_to_shader_name(const char* op);

// Map dtype to shader suffix
std::string dtype_to_shader_suffix(Dtype dtype);

} // namespace mlx::core::vulkan
