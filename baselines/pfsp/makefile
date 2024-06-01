SHELL := /bin/bash

# Compilers & common options
C_COMPILER    := gcc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc

C_COMMON_OPTS    := -O3 -Wall -g
CUDA_COMMON_OPTS := -O3 -arch=sm_86 #check how to adapt SM automatically
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) -offload-arch=gfx906

HIP_PATCH_G5K    := DEVICE_LIB_PATH=/opt/rocm-4.5.0/amdgcn/bitcode/

# Source files
C_SOURCES    :=  lib/c_taillard.c lib/c_bound_simple.c lib/c_bound_johnson.c pfsp_c.c pfsp_gpu_cuda.c pfsp_multigpu_cuda.c
CUDA_SOURCES :=  evaluate.cu lib/c_bounds_gpu.cu

# Object files
C_OBJECTS    := $(C_SOURCES:.c=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
# HIP_OBJECTS  := $(CUDA_SOURCES:cuda.cu=hip.o)

# Executable names
EXECUTABLES := pfsp_c.out pfsp_gpu_cuda.out pfsp_multigpu_cuda.out

# Library paths
C_PFSP_LIBPATH := lib/c_bound_simple.c lib/c_bound_johnson.c lib/c_taillard.c
CUDA_PFSP_LIBPATH := lib/c_bounds_gpu.cu

# Build codes
all: $(EXECUTABLES)

# Pattern rule for C source files
%.o: %.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c -fopenmp $< -o $@ -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/

# Pattern rule for CUDA source files
%.o: %.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -c $< -o $@

# Build executable for CPU only
pfsp_c.out: pfsp_c.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@

# Build executable for CUDA
pfsp_gpu_cuda.out: evaluate.o pfsp_gpu_cuda.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o lib/c_bounds_gpu.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@ -lm -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/

#Build executable for CUDA+OpenMP
pfsp_multigpu_cuda.out: evaluate.o pfsp_multigpu_cuda.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o lib/c_bounds_gpu.o
	$(C_COMPILER) $(C_COMMON_OPTS) -fopenmp $^ -o $@ -lm -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/

# Pattern rule for HIP source files
# pfsp_gpu_hip.o: pfsp_gpu_cuda.cu
# 	hipify-perl $< > $<.hip
# 	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $<.hip -o $@

# Pattern rule for hybrid OpenMP+HIP source files
# pfsp_multigpu_hip.o: pfsp_multigpu_cuda.cu
# 	hipify-perl $< > $<.hip
# 	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp $<.hip -o $@


# Utilities
.PHONY: clean

clean:
	rm -f $(C_OBJECTS) $(CUDA_OBJECTS) *.o *.out #$(HIP_OBJECTS) *.hip
