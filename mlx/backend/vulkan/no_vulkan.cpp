// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/vulkan/vulkan.h"

namespace mlx::core::vulkan {

bool is_available() {
  return false;
}

void start_capture(std::string) {}
void stop_capture() {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[vulkan::device_info] Cannot get device info without Vulkan backend");
}

} // namespace mlx::core::vulkan
