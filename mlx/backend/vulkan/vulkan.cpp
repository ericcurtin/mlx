// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/vulkan.h"
#include "mlx/backend/vulkan/device.h"

namespace mlx::core::vulkan {

bool is_available() {
  return Device::available();
}

void start_capture(std::string path) {
  // Vulkan capture would use RenderDoc or similar
  // Not implemented yet
}

void stop_capture() {
  // Not implemented yet
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  return Device::instance().info();
}

} // namespace mlx::core::vulkan
