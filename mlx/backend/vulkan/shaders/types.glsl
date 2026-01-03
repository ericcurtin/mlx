// Common types and macros for MLX Vulkan compute shaders

#ifndef MLX_VULKAN_TYPES
#define MLX_VULKAN_TYPES

// Data type definitions
#define FLOAT_TYPE float
#define FLOAT16_TYPE float16_t
#define BFLOAT16_TYPE uint16_t  // BF16 stored as uint16
#define INT32_TYPE int
#define INT64_TYPE int64_t
#define UINT32_TYPE uint
#define UINT64_TYPE uint64_t
#define BOOL_TYPE uint

// Workgroup size defaults
#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 256
#endif

// Helper macros for indexing
#define IDX_1D(x, stride_x) ((x) * (stride_x))
#define IDX_2D(x, y, stride_x, stride_y) ((x) * (stride_x) + (y) * (stride_y))
#define IDX_3D(x, y, z, stride_x, stride_y, stride_z) \
    ((x) * (stride_x) + (y) * (stride_y) + (z) * (stride_z))
#define IDX_4D(x, y, z, w, stride_x, stride_y, stride_z, stride_w) \
    ((x) * (stride_x) + (y) * (stride_y) + (z) * (stride_z) + (w) * (stride_w))

// BFloat16 conversion functions
float bf16_to_float(uint16_t v) {
    return uintBitsToFloat(uint(v) << 16);
}

uint16_t float_to_bf16(float v) {
    uint bits = floatBitsToUint(v);
    // Round to nearest even
    uint rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return uint16_t((bits + rounding_bias) >> 16);
}

// Activation functions
float gelu_approx(float x) {
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

float silu(float x) {
    return x / (1.0 + exp(-x));
}

float quick_gelu(float x) {
    return x * (1.0 / (1.0 + exp(-1.702 * x)));
}

#endif // MLX_VULKAN_TYPES
