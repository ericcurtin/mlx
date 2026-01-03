// Copyright © 2025 Apple Inc.

#include "mlx/backend/vulkan/kernels.h"
#include "mlx/backend/vulkan/device.h"
#include "mlx/backend/vulkan/utils.h"

#include <mutex>
#include <unordered_map>
#include <stdexcept>

namespace mlx::core::vulkan {

namespace {

// Cache for compiled kernels
std::unordered_map<std::string, KernelInfo> kernel_cache;
std::mutex kernel_cache_mutex;

// Map operation names to shader file names
std::string get_shader_path(const std::string& shader_name) {
  // In production, this would be the installed path
  // For now, return the shader name which will be looked up in the embedded shaders
  return shader_name + ".spv";
}

KernelInfo create_kernel(
    const std::string& shader_name,
    uint32_t num_inputs,
    uint32_t num_outputs,
    uint32_t push_constant_size) {
  
  auto& device = Device::instance();
  if (!device.is_initialized()) {
    throw std::runtime_error("[vulkan::kernels] Device not initialized");
  }

  // Load shader
  auto shader_path = get_shader_path(shader_name);
  auto shader_code = load_shader(shader_path);
  auto shader_module = create_shader_module(device.device(), shader_code);

  // Create descriptor set layout
  uint32_t num_buffers = num_inputs + num_outputs;
  auto descriptor_layout = create_storage_buffer_layout(device.device(), num_buffers);

  // Create pipeline layout
  std::vector<VkDescriptorSetLayout> layouts = {descriptor_layout};
  auto pipeline_layout = create_pipeline_layout(device.device(), layouts, push_constant_size);

  // Create compute pipeline
  auto pipeline = device.get_pipeline(shader_name, shader_module, pipeline_layout);

  // Cleanup shader module (pipeline keeps a copy)
  vkDestroyShaderModule(device.device(), shader_module, nullptr);

  return KernelInfo{pipeline, pipeline_layout, descriptor_layout};
}

} // namespace

std::string op_to_shader_name(const char* op) {
  static const std::unordered_map<std::string, std::string> op_map = {
    // Unary operations
    {"Abs", "abs"},
    {"ArcCos", "acos"},
    {"ArcCosh", "acosh"},
    {"ArcSin", "asin"},
    {"ArcSinh", "asinh"},
    {"ArcTan", "atan"},
    {"ArcTanh", "atanh"},
    {"Ceil", "ceil"},
    {"Cos", "cos"},
    {"Cosh", "cosh"},
    {"Erf", "erf"},
    {"ErfInv", "erfinv"},
    {"Exp", "exp"},
    {"Expm1", "expm1"},
    {"Floor", "floor"},
    {"Log", "log"},
    {"Log2", "log2"},
    {"Log10", "log10"},
    {"Log1p", "log1p"},
    {"LogicalNot", "logical_not"},
    {"Negative", "neg"},
    {"Round", "round"},
    {"Sigmoid", "sigmoid"},
    {"Sign", "sign"},
    {"Sin", "sin"},
    {"Sinh", "sinh"},
    {"Sqrt", "sqrt"},
    {"Square", "square"},
    {"Tan", "tan"},
    {"Tanh", "tanh"},
    
    // Binary operations
    {"Add", "add"},
    {"Subtract", "sub"},
    {"Multiply", "mul"},
    {"Divide", "div"},
    {"Maximum", "max"},
    {"Minimum", "min"},
    {"Power", "pow"},
    {"Equal", "eq"},
    {"NotEqual", "ne"},
    {"Less", "lt"},
    {"LessEqual", "le"},
    {"Greater", "gt"},
    {"GreaterEqual", "ge"},
    {"LogicalAnd", "logical_and"},
    {"LogicalOr", "logical_or"},
    {"LogAddExp", "logaddexp"},
    {"ArcTan2", "atan2"},
    {"Remainder", "remainder"},
    
    // Reduction operations
    {"Sum", "reduce_sum"},
    {"Prod", "reduce_prod"},
    {"Max", "reduce_max"},
    {"Min", "reduce_min"},
    {"Mean", "reduce_mean"},
  };

  auto it = op_map.find(op);
  if (it != op_map.end()) {
    return it->second;
  }
  
  // Default: lowercase the operation name
  std::string name(op);
  for (auto& c : name) {
    c = std::tolower(c);
  }
  return name;
}

std::string dtype_to_shader_suffix(Dtype dtype) {
  switch (dtype) {
    case float32: return "_f32";
    case float16: return "_f16";
    case bfloat16: return "_bf16";
    case int32: return "_i32";
    case int64: return "_i64";
    case uint32: return "_u32";
    case uint64: return "_u64";
    case int16: return "_i16";
    case uint16: return "_u16";
    case int8: return "_i8";
    case uint8: return "_u8";
    case bool_: return "_bool";
    case complex64: return "_c64";
    default: return "_f32";
  }
}

KernelInfo get_unary_kernel(
    const char* op,
    Dtype in_dtype,
    Dtype out_dtype) {
  
  std::string shader_name = op_to_shader_name(op);
  std::string cache_key = "unary_" + shader_name + dtype_to_shader_suffix(in_dtype);

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel with 1 input, 1 output, and push constants for size
  auto info = create_kernel(shader_name, 1, 1, sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

KernelInfo get_binary_kernel(
    const char* op,
    Dtype in_dtype,
    Dtype out_dtype) {
  
  std::string shader_name = op_to_shader_name(op);
  std::string cache_key = "binary_" + shader_name + dtype_to_shader_suffix(in_dtype);

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel with 2 inputs, 1 output, and push constants for size
  auto info = create_kernel(shader_name, 2, 1, sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

KernelInfo get_reduce_kernel(
    const char* op,
    Dtype in_dtype,
    Dtype out_dtype) {
  
  std::string shader_name = op_to_shader_name(op);
  std::string cache_key = "reduce_" + shader_name + dtype_to_shader_suffix(in_dtype);

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel with 1 input, 1 output, and push constants for sizes
  auto info = create_kernel(shader_name, 1, 1, 3 * sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

KernelInfo get_matmul_kernel(
    Dtype in_dtype,
    Dtype out_dtype,
    bool transpose_a,
    bool transpose_b) {
  
  std::string cache_key = "matmul" + dtype_to_shader_suffix(in_dtype);
  if (transpose_a) cache_key += "_ta";
  if (transpose_b) cache_key += "_tb";

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel with 2 inputs, 1 output, and push constants for M, N, K, batch_size
  auto info = create_kernel("matmul", 2, 1, 4 * sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

KernelInfo get_softmax_kernel(Dtype dtype) {
  std::string cache_key = "softmax" + dtype_to_shader_suffix(dtype);

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel with 1 input, 1 output
  auto info = create_kernel("softmax", 1, 1, 2 * sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

KernelInfo get_norm_kernel(const char* norm_type, Dtype dtype) {
  std::string shader_name = std::string(norm_type);
  std::string cache_key = shader_name + dtype_to_shader_suffix(dtype);

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel - layer_norm has 4 buffers (input, weight, bias, output)
  // rms_norm has 3 buffers (input, weight, output)
  uint32_t num_inputs = (strcmp(norm_type, "layer_norm") == 0) ? 3 : 2;
  auto info = create_kernel(shader_name, num_inputs, 1, 3 * sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

KernelInfo get_attention_kernel(Dtype dtype, bool use_mask) {
  std::string cache_key = "attention" + dtype_to_shader_suffix(dtype);
  if (use_mask) cache_key += "_masked";

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel - Q, K, V, output (+ optional mask)
  uint32_t num_inputs = use_mask ? 4 : 3;
  auto info = create_kernel("attention", num_inputs, 1, 6 * sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

KernelInfo get_rope_kernel(Dtype dtype) {
  std::string cache_key = "rope" + dtype_to_shader_suffix(dtype);

  std::lock_guard<std::mutex> lock(kernel_cache_mutex);
  
  auto it = kernel_cache.find(cache_key);
  if (it != kernel_cache.end()) {
    return it->second;
  }

  // Create kernel - input/output, freqs_cos, freqs_sin
  auto info = create_kernel("rope", 3, 0, 5 * sizeof(uint32_t));
  kernel_cache[cache_key] = info;
  return info;
}

} // namespace mlx::core::vulkan
