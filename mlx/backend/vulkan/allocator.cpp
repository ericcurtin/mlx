// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/allocator.h"
#include "mlx/backend/vulkan/device.h"

#include <algorithm>
#include <stdexcept>

namespace mlx::core::vulkan {

namespace {
size_t get_buffer_size(VulkanBuffer* buf) {
  return buf->size;
}

void free_buffer(VulkanBuffer* buf) {
  auto& device = Device::instance();
  if (device.is_initialized()) {
    vkUnmapMemory(device.device(), buf->memory);
    vkFreeMemory(device.device(), buf->memory, nullptr);
    vkDestroyBuffer(device.device(), buf->buffer, nullptr);
  }
  delete buf;
}
}  // namespace

VulkanAllocator::VulkanAllocator()
    : memory_limit_(0),
      buffer_cache_(4096, get_buffer_size, free_buffer) {
  // Get device memory properties to determine limit
  auto& device = Device::instance();
  if (device.is_initialized()) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(device.physical_device(), &mem_props);
    
    // Find the largest device-local heap
    for (uint32_t i = 0; i < mem_props.memoryHeapCount; i++) {
      if (mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
        memory_limit_ = std::max(memory_limit_, mem_props.memoryHeaps[i].size);
      }
    }
  }
}

VulkanAllocator& allocator() {
  static VulkanAllocator allocator_;
  return allocator_;
}

Buffer VulkanAllocator::malloc(size_t size) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Try to get from cache first
  VulkanBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (buf != nullptr) {
    return Buffer{buf};
  }

  // Allocate new buffer
  auto& device = Device::instance();
  buf = new VulkanBuffer();
  buf->size = size;
  buf->device = 0;
  
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | 
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  
  if (vkCreateBuffer(device.device(), &buffer_info, nullptr, &buf->buffer) != VK_SUCCESS) {
    delete buf;
    throw std::runtime_error("[vulkan::allocator] Failed to create buffer");
  }
  
  VkMemoryRequirements mem_requirements;
  vkGetBufferMemoryRequirements(device.device(), buf->buffer, &mem_requirements);
  
  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex = device.find_memory_type(
      mem_requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  
  if (vkAllocateMemory(device.device(), &alloc_info, nullptr, &buf->memory) != VK_SUCCESS) {
    vkDestroyBuffer(device.device(), buf->buffer, nullptr);
    delete buf;
    throw std::runtime_error("[vulkan::allocator] Failed to allocate memory");
  }
  
  vkBindBufferMemory(device.device(), buf->buffer, buf->memory, 0);
  vkMapMemory(device.device(), buf->memory, 0, size, 0, &buf->mapped_data);
  
  active_memory_ += size;
  peak_memory_ = std::max(peak_memory_, active_memory_);
  
  return Buffer{buf};
}

void VulkanAllocator::free(Buffer buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  
  active_memory_ -= buf->size;
  
  // Add to cache for potential reuse
  buffer_cache_.recycle_to_cache(buf);
}

size_t VulkanAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<VulkanBuffer*>(buffer.ptr());
  return buf->size;
}

size_t VulkanAllocator::get_active_memory() const {
  return active_memory_;
}

size_t VulkanAllocator::get_peak_memory() const {
  return peak_memory_;
}

void VulkanAllocator::reset_peak_memory() {
  peak_memory_ = active_memory_;
}

size_t VulkanAllocator::get_memory_limit() {
  return memory_limit_;
}

size_t VulkanAllocator::set_memory_limit(size_t limit) {
  size_t old = memory_limit_;
  memory_limit_ = limit;
  return old;
}

size_t VulkanAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t VulkanAllocator::set_cache_limit(size_t limit) {
  // BufferCache doesn't have set_cache_limit, we use release_cached_buffers
  return limit;
}

void VulkanAllocator::clear_cache() {
  buffer_cache_.clear();
}

} // namespace mlx::core::vulkan
