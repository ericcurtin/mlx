// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/device.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core {

void eval_gpu(std::vector<array>& outputs, std::shared_ptr<Primitive> primitive) {
  auto& s = primitive->stream();
  auto& encoder = vulkan::get_command_encoder(s);
  
  // Get inputs for the primitive
  std::vector<array> inputs;
  for (auto& out : outputs) {
    for (auto& in : out.inputs()) {
      encoder.set_input_array(in);
      inputs.push_back(in);
    }
    encoder.set_output_array(out);
  }
  
  // Evaluate the primitive
  primitive->eval_gpu(inputs, outputs);
}

} // namespace mlx::core
