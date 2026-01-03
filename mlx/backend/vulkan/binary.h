// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core::vulkan {

// Dispatch binary operation to Vulkan compute shader
void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op);

void binary_op_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const char* op);

void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const char* op,
    const Stream& s);

} // namespace mlx::core::vulkan
