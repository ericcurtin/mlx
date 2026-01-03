// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <vulkan/vulkan.h>
#include <mutex>
#include <vector>

namespace mlx::core::vulkan {

using allocator::Buffer;

// Stores Vulkan device memory
struct VulkanBuffer {
  VkBuffer buffer;
  VkDeviceMemory memory;
  void* mapped_data;
  size_t size;
  int device;
};

class VulkanAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  VulkanAllocator();
  friend VulkanAllocator& allocator();

  mutable std::mutex mutex_;
  size_t memory_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  mutable BufferCache<VulkanBuffer> buffer_cache_;
};

VulkanAllocator& allocator();

} // namespace mlx::core::vulkan
