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
C_SOURCES    :=   pfsp_c.c pfsp_gpu_cuda.c pfsp_multigpu_cuda.c
C_PFSP_LIBPATH	:=  lib/c_taillard.c lib/c_bound_simple.c lib/c_bound_johnson.c lib/PFSP_node.c lib/Auxiliary.c
C_PFSP_POOL_LIBPATH  :=  lib/Pool.c lib/Pool_ext.c
CUDA_PFSP_LIBPATH :=  lib/evaluate.cu lib/c_bounds_gpu.cu
HIP_PFSP_LIBPATH  :=  $(CUDA_PFSP_LIBPATH:.cu=.hip.cpp) # HIP source files generated from CUDA sources


# Object files
C_OBJECTS  :=  $(C_SOURCES:.c=.o)
C_PFSP_OBJECTS	:=  $(C_PFSP_LIBPATH:.c=.o)
C_PFSP_POOL_OBJECTS  :=  $(C_PFSP_POOL_LIBPATH:.c=.o)
CUDA_PFSP_OBJECTS := $(CUDA_PFSP_LIBPATH:.cu=.o)
HIP_PFSP_OBJECTS  := $(HIP_PFSP_LIBPATH:.hip.cpp=.o)

# Executable names
EXECUTABLES := pfsp_c.out pfsp_gpu_cuda.out pfsp_multigpu_cuda.out pfsp_gpu_hip.out #pfsp_multigpu_hip.out

# Build codes
all: $(EXECUTABLES)

# Pattern rule for C source files
%.o: %.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c -fopenmp $< -o $@ -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/

# Pattern rule for CUDA source files
%.o: %.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -c $< -o $@

# Generate HIP source files from CUDA source files
%.hip.cpp: %.cu
	hipify-perl $< > $@

# Pattern rule for HIP source files
%.o: %.hip.cpp
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -c $< -o $@

# Build executable for CPU only
pfsp_c.out: pfsp_c.o $(C_PFSP_OBJECTS) lib/Pool.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@

# Build executable for CUDA
pfsp_gpu_cuda.out: pfsp_gpu_cuda.o $(C_PFSP_OBJECTS) $(CUDA_PFSP_OBJECTS) lib/Pool.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@ -lm -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/

#Build executable for CUDA+OpenMP
pfsp_multigpu_cuda.out: pfsp_multigpu_cuda.o $(C_PFSP_OBJECTS) $(CUDA_PFSP_OBJECTS) lib/Pool_ext.o
	$(C_COMPILER) $(C_COMMON_OPTS) -fopenmp $^ -o $@ -lm -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/

# Build executable for HIP
#pfsp_gpu_hip.o: pfsp_gpu_cuda.c
#	hipify-perl $< > $<.hip.cpp
#	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $<.hip.cpp -o $@

pfsp_gpu_hip.out: pfsp_gpu_cuda.o $(C_PFSP_OBJECTS) lib/Pool.o $(HIP_PFSP_OBJECTS)	
	hipify-perl $< > $<.hip.cpp
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $^ -o $@ -lm -L/opt/rocm-4.5.0/hip/lib

# pfsp_gpu_hip.o: pfsp_gpu_cuda.c
# 	hipify-perl $< > $<.hip
# 	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $<.hip -o $@

# pfsp_gpu_hip.out: pfsp_gpu_hip.o $(C_PFSP_OBJECTS) lib/Pool.o $(HIP_PFSP_OBJECTS)
# 	hipify-perl $< > $<.hip
#  	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $<.hip -o $@ #-lm #-L/opt/rocm-4.5.0/hip/lib

# #$(HIP_COMPILER) $(HIP_COMMON_OPTS) $^ -o $@ -lm -L/opt/rocm-4.5.0/hip/lib

# # Build executable for HIP+OpenMP
# pfsp_multigpu_hip.out: pfsp_multigpu_cuda.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o lib/c_bounds_gpu.o lib/evaluate.o lib/PFSP_node.o lib/Pool_ext.o lib/Auxiliary.o
# 	hipify-perl $< > $<.hip	
# 	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp $^ -o $@ -lm -L/opt/rocm-4.5.0/hip/lib

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
	rm -f $(C_OBJECTS) $(C_PFSP_OBJECTS) $(C_PFSP_POOL_OBJECTS) $(CUDA_PFSP_OBJECTS) $(HIP_PFSP_OBJECTS) *.o *.out *hip *.hip.cpp *.dat
