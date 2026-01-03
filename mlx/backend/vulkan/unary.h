// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core::vulkan {

// Dispatch unary operation to Vulkan compute shader
void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op);

void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s);

} // namespace mlx::core::vulkan
