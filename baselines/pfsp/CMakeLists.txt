cmake_minimum_required(VERSION 3.18)  # You can specify your required CMake version

# Project name and version
project(PFSP LANGUAGES C CUDA)

# Specify the C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Add CUDA-specific flags (if needed)
set(CUDA_ARCHITECTURE "86")  # Change this to your target architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CUDA_ARCHITECTURE}")

# Include directories
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/lib)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Include CUDA directories
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Specify the source files
set(SOURCES
    pfsp_c.c
    lib/c_taillard.c
    lib/c_bound_simple.c
    lib/c_bound_johnson.c
    lib/PFSP_node.c
    lib/Pool.c
)

# Add the executable
add_executable(pfsp_c.out ${SOURCES})

# Add compile options if needed
target_compile_options(pfsp_c.out PRIVATE -Wall -O3 -g)

# Specify the CUDA source files
set(CUDA_SOURCES
    lib/c_bounds_gpu.cu
    lib/evaluate.cu
)

# Add the second executable (C + CUDA)
set(CUDA_C_SOURCES
    pfsp_gpu_cuda.c
    ${CUDA_SOURCES}
    lib/c_taillard.c
    lib/c_bound_simple.c
    lib/c_bound_johnson.c
    lib/PFSP_node.c
    lib/Pool.c
)

add_executable(pfsp_gpu_cuda.out ${CUDA_C_SOURCES})

# Link CUDA runtime library (if needed)
#find_package(CUDAToolkit REQUIRED)
target_link_libraries(pfsp_gpu_cuda.out PRIVATE ${CUDAToolkit_LIBRARIES})

# Add compile options for the second executable if needed
target_compile_options(pfsp_gpu_cuda.out PRIVATE -Wall -O3 -g)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add the third executable (C + CUDA + OpenMP)
set(CUDA_C_OMP_SOURCES
    pfsp_multigpu_cuda.c
    ${CUDA_SOURCES}
    lib/c_taillard.c
    lib/c_bound_simple.c
    lib/c_bound_johnson.c
    lib/PFSP_node.c
    lib/Pool_multi.c
)

add_executable(pfsp_multigpu_cuda.out ${CUDA_C_OMP_SOURCES})

# Link CUDA runtime library (if needed)
target_link_libraries(pfsp_multigpu_cuda.out PRIVATE ${CUDAToolkit_LIBRARIES} OpenMP::OpenMP_C)

# Add compile options for the third executable if needed
target_compile_options(pfsp_multigpu_cuda.out PRIVATE -Wall -O3 -fopenmp -g)
#target_compile_definitions(pfsp_multigpu_cuda.out PRIVATE -fopenmp)
