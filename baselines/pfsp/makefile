SHELL := /bin/bash

# Compilers & common options
C_COMPILER    := gcc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc

C_COMMON_OPTS    := -O3
CUDA_COMMON_OPTS := $(C_COMMON_OPTS) -arch=sm_86
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) -offload-arch=gfx906

HIP_PATCH_G5K    := DEVICE_LIB_PATH=/opt/rocm-4.5.0/amdgcn/bitcode/

# Source files
C_SOURCES    := pfsp_c.c
CUDA_SOURCES := pfsp_gpu_cuda.cu lib/c_bound_simple_gpu_cuda.cu #pfsp_multigpu_cuda.cu

# Object files
C_OBJECTS    := $(C_SOURCES:.c=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
# HIP_OBJECTS  := $(CUDA_SOURCES:cuda.cu=hip.o)

# Library paths
C_PFSP_LIBPATH := lib/c_bound_simple.c lib/c_bound_johnson.c lib/c_taillard.c
CUDA_PFSP_LIBPATH := lib/c_bound_simple_gpu_cuda.cu #lib/c_bound_johnson_gpu_cuda.cu


# Build codes
all: $(C_OBJECTS) $(CUDA_OBJECTS) # $(HIP_OBJECTS)

# Pattern rule for C source files
pfsp_c.o: pfsp_c.c
	$(C_COMPILER) $(C_COMMON_OPTS) $< -o $@ $(C_PFSP_LIBPATH)

# Pattern rule for CUDA source files
lib/c_bound_simple_gpu_cuda.o: lib/c_bound_simple_gpu_cuda.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -dc $< -o $@

pfsp_gpu_cuda.o: pfsp_gpu_cuda.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -dc $< -o $@

	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) $(CUDA_OBJECTS) -dlink -o exe.o

	gcc exe.o $(CUDA_OBJECTS) $(C_OBJECTS) -lcudart -o exe

#exe.o: pfsp_gpu_cuda.o lib/c_bound_simple_gpu_cuda.o
#	$(CUDA_COMPILER) -x cu $(CUDA_COMMON_OPTS) -I. -dc $< -o $@

# Pattern rule for hybrid OpenMP+CUDA source files
# pfsp_multigpu_cuda.o: pfsp_multigpu_cuda.cu
# 	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -Xcompiler -fopenmp $< -o $@

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
	rm -f $(C_OBJECTS) $(CUDA_OBJECTS) #$(HIP_OBJECTS) *.hip
