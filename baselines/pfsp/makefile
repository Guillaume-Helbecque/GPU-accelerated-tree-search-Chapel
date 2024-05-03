SHELL := /bin/bash

# Compilers & common options
C_COMPILER    := gcc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc

C_COMMON_OPTS    := -O3 -Wall -g
CUDA_COMMON_OPTS := -O3 -arch=sm_86 #$(C_COMMON_OPTS) -arch=sm_70 #for grid5k
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) -offload-arch=gfx906

HIP_PATCH_G5K    := DEVICE_LIB_PATH=/opt/rocm-4.5.0/amdgcn/bitcode/

# Source files
C_SOURCES    :=  lib/c_taillard.c lib/c_bound_simple.c lib/c_bound_johnson.c pfsp_c.c pfsp_gpu_cuda.c pfsp_multi_gpu_cuda.c
CUDA_SOURCES :=  evaluate.cu lib/c_bounds_gpu.cu # lib/c_bound_johnson_gpu_cuda.cu pfsp_multigpu_cuda.cu

# Object files
C_OBJECTS    := $(C_SOURCES:.c=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
# HIP_OBJECTS  := $(CUDA_SOURCES:cuda.cu=hip.o)

# Library paths
C_PFSP_LIBPATH := lib/c_bound_simple.c lib/c_bound_johnson.c lib/c_taillard.c
CUDA_PFSP_LIBPATH := lib/c_bounds_gpu.cu


# Build codes
all: pfsp_c.out pfsp_cuda.out pfsp_multi_cuda.out #evaluate.o $(C_OBJECTS) # $(CUDA_OBJECTS) # $(HIP_OBJECTS)

# Pattern rule for C source files

c_bound_simple.o : lib/c_bound_simple.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

c_bound_johnson.o : lib/c_bound_johnson.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

c_taillard.o : lib/c_taillard.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

evaluate.o : evaluate.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -c $< -o $@

pfsp_c.o: pfsp_c.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

#pfsp_gpu_cuda.o: pfsp_gpu_cuda.c
#	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@ -I/share/compilers/nvidia/cuda/12.0/include

#pfsp_multi_gpu_cuda.o: pfsp_multi_gpu_cuda.c
#	$(C_COMPILER) $(C_COMMON_OPTS) -c -fopenmp $< -o $@ -I/share/compilers/nvidia/cuda/12.0/include

#Grid5000 compilation
pfsp_gpu_cuda.o: pfsp_gpu_cuda.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@ -I/usr/local/cuda-11.2/targets/x86_64-linux/include/

pfsp_multi_gpu_cuda.o: pfsp_multi_gpu_cuda.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c -fopenmp $< -o $@ -I/usr/local/cuda-11.2/targets/x86_64-linux/include/


# Pattern rule for CUDA source files

c_bounds_gpu.o: lib/c_bounds_gpu.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -c $< -o $@

# Pattern for executable file

pfsp_c.out: pfsp_c.o c_taillard.o c_bound_simple.o c_bound_johnson.o
	$(C_COMPILER) $(C_COMMON_OPTS) pfsp_c.o c_taillard.o c_bound_simple.o c_bound_johnson.o   -o pfsp_c.out

#pfsp_cuda.out: evaluate.o pfsp_gpu_cuda.o c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o
#	$(C_COMPILER) $(C_COMMON_OPTS) evaluate.o pfsp_gpu_cuda.o c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o -o pfsp_cuda.out -lcudart #-L/share/compilers/nvidia/cuda/12.0/lib64 -lcudart

#pfsp_multi_cuda.out: evaluate.o pfsp_multi_gpu_cuda.o c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o
#	$(C_COMPILER) $(C_COMMON_OPTS) -fopenmp evaluate.o pfsp_multi_gpu_cuda.o  c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o -o pfsp_multi_cuda.out -lcudart #-L/share/compilers/nvidia/cuda/12.0/lib64 -lcudart


# Grid5000 compilation
pfsp_cuda.out: evaluate.o pfsp_gpu_cuda.o c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o
	$(C_COMPILER) $(C_COMMON_OPTS) evaluate.o pfsp_gpu_cuda.o c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o -o pfsp_cuda.out -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/

pfsp_multi_cuda.out: evaluate.o pfsp_multi_gpu_cuda.o c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o
	$(C_COMPILER) $(C_COMMON_OPTS) -fopenmp evaluate.o pfsp_multi_gpu_cuda.o  c_taillard.o c_bound_simple.o c_bound_johnson.o c_bounds_gpu.o -o pfsp_multi_cuda.out -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/

# EVERYTHING FROM THE OLD COMPILATION

# pfsp_gpu_cuda.o: pfsp_gpu_cuda.cu
# 	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -dc $< -o $@

# #	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) lib/c_bound_simple.o $< -o exe.o 
# 	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) lib/c_bound_simple_gpu_cuda.o pfsp_gpu_cuda.o -dlink -o exe.o
# # $(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -dc $< -o $@
# # $(CUDA_COMPILER) $(CUDA_COMMON_OPTS) $(CUDA_OBJECTS) -dlink -o exe.o
# # gcc exe.o $(CUDA_OBJECTS) $(C_OBJECTS) -lcudart -o exe

#gcc exe.o $(CUDA_OBJECTS) $(C_OBJECTS) -lcudart -o exe

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
	rm -f $(C_OBJECTS) $(CUDA_OBJECTS) *.o *.out #$(HIP_OBJECTS) *.hip
