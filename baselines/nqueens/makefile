SHELL := /bin/bash

# Compilers & common options
C_COMPILER    := gcc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc

C_COMMON_OPTS    := -O3
CUDA_COMMON_OPTS := $(C_COMMON_OPTS) -arch=sm_70
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) -offload-arch=gfx906

HIP_PATCH_G5K    := DEVICE_LIB_PATH=/opt/rocm-4.5.0/amdgcn/bitcode/

# Source files
C_SOURCES    := nqueens_c.c
CUDA_SOURCES := nqueens_gpu_cuda.cu nqueens_multigpu_cuda.cu

# Object files
C_OBJECTS    := $(C_SOURCES:.c=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
HIP_OBJECTS  := $(CUDA_SOURCES:cuda.cu=hip.o)

# Build codes
all: $(C_OBJECTS) $(CUDA_OBJECTS) $(HIP_OBJECTS)

# Pattern rule for C source files
nqueens_c.o: nqueens_c.c
	$(C_COMPILER) $(C_COMMON_OPTS) $< -o $@

# Pattern rule for CUDA source files
nqueens_gpu_cuda.o: nqueens_gpu_cuda.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) $< -o $@

# Pattern rule for hybrid OpenMP+CUDA source files
nqueens_multigpu_cuda.o: nqueens_multigpu_cuda.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -Xcompiler -fopenmp $< -o $@

# Pattern rule for HIP source files
nqueens_gpu_hip.o: nqueens_gpu_cuda.cu
	hipify-perl $< > $<.hip
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) $<.hip -o $@

# Pattern rule for hybrid OpenMP+HIP source files
nqueens_multigpu_hip.o: nqueens_multigpu_cuda.cu
	hipify-perl $< > $<.hip
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp $<.hip -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(C_OBJECTS) $(CUDA_OBJECTS) $(HIP_OBJECTS) *.hip
