# MLX Vulkan Backend
# This provides GPU compute capabilities using the Vulkan API

# Find Vulkan
find_package(Vulkan REQUIRED)

target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/vulkan.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_VULKAN)

target_include_directories(mlx PRIVATE ${Vulkan_INCLUDE_DIRS})
target_link_libraries(mlx PRIVATE Vulkan::Vulkan)

# Shader compilation (GLSL -> SPIR-V)
# The shaders are compiled at build time using glslc
if(Vulkan_GLSLC_EXECUTABLE)
  set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
  set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
  
  file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})
  
  # List all shader files
  file(GLOB SHADER_SOURCES "${SHADER_SOURCE_DIR}/*.comp")
  
  set(SHADER_SPIRV_FILES "")
  foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPIRV_FILE ${SHADER_BINARY_DIR}/${SHADER_NAME}.spv)
    
    add_custom_command(
      OUTPUT ${SPIRV_FILE}
      COMMAND ${Vulkan_GLSLC_EXECUTABLE} -o ${SPIRV_FILE} ${SHADER}
      DEPENDS ${SHADER}
      COMMENT "Compiling ${SHADER_NAME} to SPIR-V"
    )
    
    list(APPEND SHADER_SPIRV_FILES ${SPIRV_FILE})
  endforeach()
  
  add_custom_target(mlx_vulkan_shaders DEPENDS ${SHADER_SPIRV_FILES})
  add_dependencies(mlx mlx_vulkan_shaders)
  
  # Install compiled shaders
  install(FILES ${SHADER_SPIRV_FILES}
          DESTINATION ${CMAKE_INSTALL_DATADIR}/mlx/shaders)
endif()

# Set RPATH for Python bindings
if(MLX_BUILD_PYTHON_BINDINGS)
  set_property(
    TARGET mlx
    APPEND
    PROPERTY INSTALL_RPATH "$ORIGIN/../../")
endif()
