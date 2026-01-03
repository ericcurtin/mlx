// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/vulkan/allocator.h"
#include "mlx/stream.h"

#include <vulkan/vulkan.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mlx::core::vulkan {

class CommandEncoder;

class Device {
 public:
  static Device& instance();
  static bool available();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  bool is_initialized() const { return initialized_; }
  
  VkInstance vk_instance() const { return instance_; }
  VkPhysicalDevice physical_device() const { return physical_device_; }
  VkDevice device() const { return device_; }
  VkQueue compute_queue() const { return compute_queue_; }
  uint32_t compute_queue_family() const { return compute_queue_family_; }
  
  uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
  
  const std::unordered_map<std::string, std::variant<std::string, size_t>>& info() const {
    return device_info_;
  }
  
  CommandEncoder& get_command_encoder(Stream s);
  
  // Get or create a compute pipeline for a shader
  VkPipeline get_pipeline(const std::string& shader_name, 
                          VkShaderModule shader_module,
                          VkPipelineLayout layout);

 private:
  Device();
  ~Device();
  
  void init();
  void cleanup();
  
  bool initialized_{false};
  VkInstance instance_{VK_NULL_HANDLE};
  VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
  VkDevice device_{VK_NULL_HANDLE};
  VkQueue compute_queue_{VK_NULL_HANDLE};
  uint32_t compute_queue_family_{0};
  
  VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
  VkCommandPool command_pool_{VK_NULL_HANDLE};
  
  std::unordered_map<std::string, std::variant<std::string, size_t>> device_info_;
  std::unordered_map<std::string, VkPipeline> pipeline_cache_;
  std::unordered_map<int, std::unique_ptr<CommandEncoder>> encoders_;
  
  std::mutex mutex_;
};

class CommandEncoder {
 public:
  explicit CommandEncoder(Device& device);
  ~CommandEncoder();
  
  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;
  
  void set_input_array(const array& arr);
  void set_output_array(const array& arr);
  
  void dispatch(VkPipeline pipeline, 
                VkPipelineLayout layout,
                uint32_t group_count_x,
                uint32_t group_count_y = 1,
                uint32_t group_count_z = 1);
  
  void copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);
  
  void add_completed_handler(std::function<void()> task);
  void commit();
  void synchronize();
  
  VkCommandBuffer command_buffer() const { return command_buffer_; }
  
 private:
  Device& device_;
  VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
  VkFence fence_{VK_NULL_HANDLE};
  bool recording_{false};
  
  std::vector<std::function<void()>> completed_handlers_;
};

Device& device(mlx::core::Device d);
CommandEncoder& get_command_encoder(Stream s);

} // namespace mlx::core::vulkan
