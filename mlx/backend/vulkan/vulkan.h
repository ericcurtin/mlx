// Copyright © 2025 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

namespace mlx::core::vulkan {

/* Check if the Vulkan backend is available. */
bool is_available();

/* Get device information */
const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info();

/* Start GPU capture for debugging */
void start_capture(std::string path = "");

/* Stop GPU capture */
void stop_capture();

} // namespace mlx::core::vulkan
