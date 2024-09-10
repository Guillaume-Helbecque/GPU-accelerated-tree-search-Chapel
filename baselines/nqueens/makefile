SHELL := /bin/bash

SYSTEM ?= g5k

# Compilers & common options
C_COMPILER    := gcc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc

C_COMMON_OPTS    := -O3
CUDA_COMMON_OPTS := $(C_COMMON_OPTS) -arch=sm_70
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) -offload-arch=gfx906

LIBPATH := lib

# Platform-specific flags and libraries
ifeq ($(SYSTEM), g5k)
  # HIP compiler patch
  HIP_COMPILER := DEVICE_LIB_PATH=/opt/rocm-4.5.0/amdgcn/bitcode/ hipcc
else ifeq ($(SYSTEM), lumi)

endif

# Source files
# C_SOURCES    := nqueens_c.c
# CUDA_SOURCES := nqueens_gpu_cuda.cu nqueens_multigpu_cuda.cu

# Object files
# C_OBJECTS    := $(C_SOURCES:.c=.o)
# CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
# HIP_OBJECTS  := $(CUDA_SOURCES:cuda.cu=hip.o)

# Executable names
all: nqueens_c.out nqueens_gpu_cuda.out nqueens_multigpu_cuda.out nqueens_gpu_hip.out nqueens_multigpu_hip.out

# Pattern rule for C library source files
$(LIBPATH)/%.o: $(LIBPATH)/%.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

# Build executable for sequential in C
nqueens_c.out: nqueens_c.c $(LIBPATH)/NQueens_node.o $(LIBPATH)/Pool.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@

# Build executable for single-GPU in C+CUDA
nqueens_gpu_cuda.out: nqueens_gpu_cuda.cu $(LIBPATH)/NQueens_node.o $(LIBPATH)/Pool.o
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) $^ -o $@

# Build executable for multi-GPU in C+OpenMP+CUDA
nqueens_multigpu_cuda.out: nqueens_multigpu_cuda.cu $(LIBPATH)/NQueens_node.o $(LIBPATH)/Pool.o
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -Xcompiler -fopenmp $^ -o $@

# TODO: find an elegant way to avoid intermediate *_hip.o object files

# Build executable for single-GPU in C+HIP
nqueens_gpu_hip.o: nqueens_gpu_cuda.cu
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -c $<.hip -o $@

nqueens_gpu_hip.out: nqueens_gpu_hip.o $(LIBPATH)/NQueens_node.o $(LIBPATH)/Pool.o
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $^ -o $@

# Build executable for multi-GPU in C+OpenMP+HIP
nqueens_multigpu_hip.o: nqueens_multigpu_cuda.cu
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp -c $<.hip -o $@

nqueens_multigpu_hip.out: nqueens_multigpu_hip.o $(LIBPATH)/NQueens_node.o $(LIBPATH)/Pool.o
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp $^ -o $@

# Utilities
.PHONY: clean

clean:
	rm -f *.out *.o *.hip $(LIBPATH)/*.o
