// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/utils.h"

#include <fstream>
#include <stdexcept>

namespace mlx::core::vulkan {

std::vector<uint32_t> load_shader(const std::string& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  
  if (!file.is_open()) {
    throw std::runtime_error("[vulkan::utils] Failed to open shader file: " + path);
  }
  
  size_t file_size = static_cast<size_t>(file.tellg());
  std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));
  
  file.seekg(0);
  file.read(reinterpret_cast<char*>(buffer.data()), file_size);
  file.close();
  
  return buffer;
}

VkShaderModule create_shader_module(VkDevice device, const std::vector<uint32_t>& code) {
  VkShaderModuleCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  create_info.codeSize = code.size() * sizeof(uint32_t);
  create_info.pCode = code.data();
  
  VkShaderModule shader_module;
  if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
    throw std::runtime_error("[vulkan::utils] Failed to create shader module");
  }
  
  return shader_module;
}

VkPipelineLayout create_pipeline_layout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayout>& descriptor_layouts,
    uint32_t push_constant_size) {
  
  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = static_cast<uint32_t>(descriptor_layouts.size());
  layout_info.pSetLayouts = descriptor_layouts.data();
  
  VkPushConstantRange push_constant_range{};
  if (push_constant_size > 0) {
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = push_constant_size;
    
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant_range;
  }
  
  VkPipelineLayout pipeline_layout;
  if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
    throw std::runtime_error("[vulkan::utils] Failed to create pipeline layout");
  }
  
  return pipeline_layout;
}

VkDescriptorSetLayout create_storage_buffer_layout(
    VkDevice device,
    uint32_t num_buffers,
    VkShaderStageFlags stage) {
  
  std::vector<VkDescriptorSetLayoutBinding> bindings(num_buffers);
  
  for (uint32_t i = 0; i < num_buffers; i++) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = stage;
    bindings[i].pImmutableSamplers = nullptr;
  }
  
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = num_buffers;
  layout_info.pBindings = bindings.data();
  
  VkDescriptorSetLayout layout;
  if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
    throw std::runtime_error("[vulkan::utils] Failed to create descriptor set layout");
  }
  
  return layout;
}

} // namespace mlx::core::vulkan
