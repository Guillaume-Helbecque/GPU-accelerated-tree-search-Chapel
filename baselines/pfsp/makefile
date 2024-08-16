SHELL := /bin/bash

# Compilers & common options
C_COMPILER    := gcc
MPI_COMPILER  := mpicc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc

C_COMMON_OPTS    := -O3 -Wall -g
CUDA_COMMON_OPTS := -O3 -arch=sm_86 # TODO: adapt SM automatically, if possible
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) -offload-arch=gfx906

HIP_PATCH_G5K    := DEVICE_LIB_PATH=/opt/rocm-4.5.0/amdgcn/bitcode/

MPI_INCLUDE_G5K  := -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB_G5K      := -L/usr/lib/x86_64-linux-gnu/openmpi/lib
CUDA_INCLUDE_G5K := -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include
CUDA_LIB_G5K     := -L/usr/local/cuda-11.2/targets/x86_64-linux/lib

LIBPATH := lib

# Source files
C_SOURCES        := pfsp_c.c pfsp_gpu_cuda.c pfsp_multigpu_cuda.c pfsp_dist_multigpu_cuda.c
C_LIB_SOURCES    := $(LIBPATH)/c_taillard.c $(LIBPATH)/c_bound_simple.c $(LIBPATH)/c_bound_johnson.c $(LIBPATH)/PFSP_node.c $(LIBPATH)/Pool.c $(LIBPATH)/Pool_ext.c $(LIBPATH)/Auxiliary.c
CUDA_LIB_SOURCES := $(LIBPATH)/evaluate.cu $(LIBPATH)/c_bounds_gpu.cu

# Object files
C_OBJECTS        := $(C_SOURCES:.c=.o)
C_LIB_OBJECTS    := $(C_LIB_SOURCES:.c=.o)
CUDA_LIB_OBJECTS := $(CUDA_LIB_SOURCES:.cu=.o)
HIP_OBJECTS      := $(HIP_SOURCES:hip.cu=hip.o)

# Executable names
EXECUTABLES := pfsp_c.out pfsp_gpu_cuda.out pfsp_multigpu_cuda.out pfsp_dist_multigpu_cuda.out pfsp_gpu_hip.out pfsp_multigpu_hip.out pfsp_dist_multigpu_hip.out

# Build codes
all: $(EXECUTABLES)

# Pattern rule for C source files
%.o: %.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c -fopenmp $< -o $@ $(CUDA_INCLUDE_G5K)

pfsp_dist_multigpu_cuda.o: pfsp_dist_multigpu_cuda.c
	$(MPI_COMPILER) $(C_COMMON_OPTS) -c -fopenmp $< -o $@ $(CUDA_INCLUDE_G5K)

# Pattern rule for C library source files
$(LIBPATH)/%.o: $(LIBPATH)/%.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

# Pattern rule for CUDA source files
$(LIBPATH)/%.o: $(LIBPATH)/%.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -c $< -o $@

# Build executable for CPU only
pfsp_c.out: pfsp_c.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/Pool.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@

# Build executable for CUDA
pfsp_gpu_cuda.out: pfsp_gpu_cuda.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/evaluate.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/Pool.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@ -lm -lcudart $(CUDA_LIB_G5K)

#Build executable for CUDA+OpenMP
pfsp_multigpu_cuda.out: pfsp_multigpu_cuda.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/evaluate.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/Pool_ext.o $(LIBPATH)/Auxiliary.o
	$(C_COMPILER) $(C_COMMON_OPTS) -fopenmp $^ -o $@ -lm -lcudart $(CUDA_LIB_G5K)

#Build executable CUDA+OpenMP+MPI
pfsp_dist_multigpu_cuda.out: pfsp_dist_multigpu_cuda.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/evaluate.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/Pool_ext.o $(LIBPATH)/Auxiliary.o
	$(MPI_COMPILER) $(C_COMMON_OPTS) -fopenmp $^ -o $@ -lm -lcudart $(CUDA_LIB_G5K)

#############################################################
#############################################################

# Pattern rule for HIP source files
	hipify-perl pfsp_gpu_cuda.c > pfsp_gpu_cuda.c.hip
	hipify-perl $(LIBPATH)/evaluate.cu > $(LIBPATH)/evaluate.cu.hip
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) -c pfsp_gpu_cuda.c.hip -o pfsp_gpu_hip.o
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) -c $(LIBPATH)/evaluate.cu.hip -o $(LIBPATH)/evaluate_hip.o

pfsp_gpu_hip.out: pfsp_gpu_hip.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/Pool.o $(LIBPATH)/evaluate_hip.o
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) $^ -o $@

# Pattern rule for hybrid OpenMP+HIP source files
	hipify-perl pfsp_multigpu_cuda.c > pfsp_multigpu_cuda.c.hip
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp -c pfsp_multigpu_cuda.c.hip -o pfsp_multigpu_hip.o

pfsp_multigpu_hip.out: pfsp_multigpu_hip.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/Pool_ext.o $(LIBPATH)/Auxiliary.o $(LIBPATH)/evaluate_hip.o
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp $^ -o $@

# Pattern rule for hybrid MPI+OpenMP+HIP source files
	hipify-perl pfsp_dist_multigpu_cuda.c > pfsp_dist_multigpu_cuda.c.hip
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) $(MPI_INCLUDE_G5K) $(MPI_LIB_G5K) -fopenmp -c pfsp_dist_multigpu_cuda.c.hip -o pfsp_dist_multigpu_hip.o

pfsp_dist_multigpu_hip.out: pfsp_dist_multigpu_hip.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/Pool_ext.o $(LIBPATH)/Auxiliary.o $(LIBPATH)/evaluate_hip.o
	$(HIP_PATCH_G5K) $(HIP_COMPILER) $(HIP_COMMON_OPTS) $(MPI_INCLUDE_G5K) $(MPI_LIB_G5K) -lmpi_cxx -lmpi -lopen-pal -fopenmp $^ -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(C_OBJECTS) $(CUDA_OBJECTS) *.o *.out *.dat $(HIP_OBJECTS) *.hip
