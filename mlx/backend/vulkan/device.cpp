// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/device.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace mlx::core::vulkan {

Device& Device::instance() {
  static Device device;
  return device;
}

bool Device::available() {
  return instance().is_initialized();
}

Device::Device() {
  init();
}

Device::~Device() {
  cleanup();
}

void Device::init() {
  // Create Vulkan instance
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "MLX";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "MLX";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;
  
  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  
  if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
    return; // Vulkan not available
  }
  
  // Enumerate physical devices
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  
  if (device_count == 0) {
    cleanup();
    return;
  }
  
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
  
  // Select the first suitable device (prefer discrete GPU)
  for (const auto& dev : devices) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(dev, &props);
    
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      physical_device_ = dev;
      break;
    }
  }
  
  if (physical_device_ == VK_NULL_HANDLE) {
    physical_device_ = devices[0]; // Fall back to first device
  }
  
  // Get device properties for info
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physical_device_, &props);
  
  device_info_["device_name"] = std::string(props.deviceName);
  device_info_["vendor_id"] = static_cast<size_t>(props.vendorID);
  device_info_["device_id"] = static_cast<size_t>(props.deviceID);
  device_info_["api_version"] = static_cast<size_t>(props.apiVersion);
  device_info_["driver_version"] = static_cast<size_t>(props.driverVersion);
  
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_props);
  
  size_t total_memory = 0;
  for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
    if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      total_memory = std::max(total_memory, mem_props.memoryHeaps[i].size);
    }
  }
  device_info_["total_memory"] = total_memory;
  
  // Find compute queue family
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
  
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());
  
  for (uint32_t i = 0; i < queue_family_count; i++) {
    if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      compute_queue_family_ = i;
      break;
    }
  }
  
  // Create logical device
  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_create_info{};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = compute_queue_family_;
  queue_create_info.queueCount = 1;
  queue_create_info.pQueuePriorities = &queue_priority;
  
  VkPhysicalDeviceFeatures device_features{};
  device_features.shaderFloat64 = VK_TRUE;
  device_features.shaderInt64 = VK_TRUE;
  device_features.shaderInt16 = VK_TRUE;
  
  VkDeviceCreateInfo device_create_info{};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.pQueueCreateInfos = &queue_create_info;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pEnabledFeatures = &device_features;
  
  if (vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_) != VK_SUCCESS) {
    cleanup();
    return;
  }
  
  vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
  
  // Create command pool
  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.queueFamilyIndex = compute_queue_family_;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  
  if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
    cleanup();
    return;
  }
  
  // Create descriptor pool
  VkDescriptorPoolSize pool_sizes[] = {
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 100}
  };
  
  VkDescriptorPoolCreateInfo desc_pool_info{};
  desc_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  desc_pool_info.poolSizeCount = 2;
  desc_pool_info.pPoolSizes = pool_sizes;
  desc_pool_info.maxSets = 1000;
  desc_pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  
  if (vkCreateDescriptorPool(device_, &desc_pool_info, nullptr, &descriptor_pool_) != VK_SUCCESS) {
    cleanup();
    return;
  }
  
  initialized_ = true;
}

void Device::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
    
    for (auto& [name, pipeline] : pipeline_cache_) {
      vkDestroyPipeline(device_, pipeline, nullptr);
    }
    pipeline_cache_.clear();
    
    if (descriptor_pool_ != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    }
    
    if (command_pool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    
    vkDestroyDevice(device_, nullptr);
  }
  
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
  }
  
  initialized_ = false;
}

uint32_t Device::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties mem_properties;
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem_properties);
  
  for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  
  throw std::runtime_error("[vulkan::device] Failed to find suitable memory type");
}

VkPipeline Device::get_pipeline(const std::string& shader_name,
                                 VkShaderModule shader_module,
                                 VkPipelineLayout layout) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = pipeline_cache_.find(shader_name);
  if (it != pipeline_cache_.end()) {
    return it->second;
  }
  
  VkComputePipelineCreateInfo pipeline_info{};
  pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipeline_info.stage.module = shader_module;
  pipeline_info.stage.pName = "main";
  pipeline_info.layout = layout;
  
  VkPipeline pipeline;
  if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
    throw std::runtime_error("[vulkan::device] Failed to create compute pipeline");
  }
  
  pipeline_cache_[shader_name] = pipeline;
  return pipeline;
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    auto encoder = std::make_unique<CommandEncoder>(*this);
    auto& ref = *encoder;
    encoders_[s.index] = std::move(encoder);
    return ref;
  }
  return *it->second;
}

// CommandEncoder implementation
CommandEncoder::CommandEncoder(Device& device) : device_(device) {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = device_.command_pool_;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = 1;
  
  if (vkAllocateCommandBuffers(device_.device(), &alloc_info, &command_buffer_) != VK_SUCCESS) {
    throw std::runtime_error("[vulkan::encoder] Failed to allocate command buffer");
  }
  
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  
  if (vkCreateFence(device_.device(), &fence_info, nullptr, &fence_) != VK_SUCCESS) {
    throw std::runtime_error("[vulkan::encoder] Failed to create fence");
  }
}

CommandEncoder::~CommandEncoder() {
  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_.device(), fence_, nullptr);
  }
}

void CommandEncoder::set_input_array(const array& arr) {
  // Track input arrays for synchronization
}

void CommandEncoder::set_output_array(const array& arr) {
  // Track output arrays for synchronization
}

void CommandEncoder::dispatch(VkPipeline pipeline,
                               VkPipelineLayout layout,
                               uint32_t group_count_x,
                               uint32_t group_count_y,
                               uint32_t group_count_z) {
  if (!recording_) {
    vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_.device(), 1, &fence_);
    vkResetCommandBuffer(command_buffer_, 0);
    
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(command_buffer_, &begin_info);
    recording_ = true;
  }
  
  vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdDispatch(command_buffer_, group_count_x, group_count_y, group_count_z);
}

void CommandEncoder::copy_buffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
  if (!recording_) {
    vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_.device(), 1, &fence_);
    vkResetCommandBuffer(command_buffer_, 0);
    
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(command_buffer_, &begin_info);
    recording_ = true;
  }
  
  VkBufferCopy copy_region{};
  copy_region.size = size;
  vkCmdCopyBuffer(command_buffer_, src, dst, 1, &copy_region);
}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  completed_handlers_.push_back(std::move(task));
}

void CommandEncoder::commit() {
  if (recording_) {
    vkEndCommandBuffer(command_buffer_);
    
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer_;
    
    vkQueueSubmit(device_.compute_queue(), 1, &submit_info, fence_);
    
    recording_ = false;
  }
}

void CommandEncoder::synchronize() {
  commit();
  vkWaitForFences(device_.device(), 1, &fence_, VK_TRUE, UINT64_MAX);
  
  // Execute completed handlers
  for (auto& handler : completed_handlers_) {
    handler();
  }
  completed_handlers_.clear();
}

Device& device(mlx::core::Device d) {
  return Device::instance();
}

CommandEncoder& get_command_encoder(Stream s) {
  return Device::instance().get_command_encoder(s);
}

} // namespace mlx::core::vulkan
