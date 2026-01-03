// Copyright © 2025 Apple Inc.

#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace mlx::core::vulkan {

// Load a SPIR-V shader from file
std::vector<uint32_t> load_shader(const std::string& path);

// Create a shader module from SPIR-V code
VkShaderModule create_shader_module(VkDevice device, const std::vector<uint32_t>& code);

// Create a compute pipeline layout
VkPipelineLayout create_pipeline_layout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptor_layouts,
    uint32_t push_constant_size = 0);

// Create a descriptor set layout for storage buffers
VkDescriptorSetLayout create_storage_buffer_layout(
    VkDevice device,
    uint32_t num_buffers,
    VkShaderStageFlags stage = VK_SHADER_STAGE_COMPUTE_BIT);

// Utility to compute workgroup counts
inline uint32_t div_ceil(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

// Default workgroup sizes for common operations
constexpr uint32_t WORKGROUP_SIZE_1D = 256;
constexpr uint32_t WORKGROUP_SIZE_2D = 16;

} // namespace mlx::core::vulkan
