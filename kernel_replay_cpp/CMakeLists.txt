cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(kernel_replay_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Find libtorch
# Set this to your libtorch installation path
# Download from: https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/opt/libtorch")
find_package(Torch REQUIRED)

# Set CUDA architectures (adjust for your GPU)
# sm_61 for GTX 1080 Ti, sm_86 for A6000, etc.
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)

# Find nlohmann/json (header-only library)
# If not found system-wide, it will try to use it from include/
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann/json not found system-wide, expecting it in include/")
    # You can download it: 
    # wget https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
    # and place it in include/nlohmann/json.hpp
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Source files
set(SOURCES
    src/main.cpp
    src/kernel_registry.cpp
    src/cuda_kernels.cpp
    src/cublas_kernels.cpp
    src/libtorch_kernels.cpp
    src/full_model_benchmark.cpp
    src/aggregation.cpp
)

# Create executable
add_executable(kernel_benchmark ${SOURCES})

# Link libraries
target_link_libraries(kernel_benchmark
    CUDA::cudart
    CUDA::cublas
    "${TORCH_LIBRARIES}"
)

# Set C++ compile flags for libtorch
set_property(TARGET kernel_benchmark PROPERTY CXX_STANDARD 17)

# Copy CUDA/Torch DLLs on Windows
if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET kernel_benchmark
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:kernel_benchmark>)
endif(MSVC)

# Enable verbose output for debugging
set(CMAKE_VERBOSE_MAKEFILE ON)

# Print configuration
message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
message(STATUS "Torch found: ${TORCH_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
