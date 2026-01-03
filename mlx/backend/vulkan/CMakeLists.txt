# Note: This is a placeholder for the Vulkan backend.
# When MLX_BUILD_VULKAN=ON is set, this file will be used to add Vulkan
# specific sources to the mlx target.

target_compile_definitions(mlx PRIVATE MLX_USE_VULKAN)

# TODO: Add Vulkan-specific sources here when the backend is implemented.
# For now, this is a stub that marks the backend as available.
