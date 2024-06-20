hipcc -O3 -Wall -g -c -fopenmp lib/c_taillard.c -o lib/c_taillard.o
hipcc -O3 -Wall -g -c -fopenmp lib/c_bound_simple.c -o lib/c_bound_simple.o
hipcc -O3 -Wall -g -c -fopenmp lib/c_bound_johnson.c -o lib/c_bound_johnson.o 
hipcc -O3 -Wall -g -c -fopenmp lib/PFSP_node.c -o lib/PFSP_node.o 
hipcc -O3 -Wall -g -c -fopenmp lib/Auxiliary.c -o lib/Auxiliary.o 
hipcc -O3 -Wall -g -c -fopenmp lib/Pool.c -o lib/Pool.o

hipify-perl pfsp_gpu_hip.cu > pfsp_gpu_hip.cu.hip
hipcc -O3 -offload-arch=gfx906 -c pfsp_gpu_hip.cu.hip -o pfsp_gpu_hip.o
hipcc -O3 -offload-arch=gfx906 pfsp_gpu_hip.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o lib/PFSP_node.o lib/Auxiliary.o lib/Pool.o -o pfsp_gpu_hip.out -L/opt/rocm-4.5.0/hip/lib

#hipcc -O3 -Wall -g -c -fopenmp pfsp_gpu_hip.c.hip -o pfsp_gpu_hip.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/ -L/opt/rocm-4.5.0/hip/lib
#hipcc -O3 -Wall -g pfsp_gpu_hip.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o lib/PFSP_node.o lib/Auxiliary.o lib/evaluate.o lib/c_bounds_gpu.o lib/Pool.o -o pfsp_gpu_hip.out -lm -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/

#nvcc -O3 -arch=sm_86 -c pfsp_gpu_hip.cu -o pfsp_gpu_hip.o -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include/
#nvcc -O3 pfsp_gpu_hip.o lib/c_taillard.o lib/c_bound_simple.o lib/c_bound_johnson.o lib/PFSP_node.o lib/Auxiliary.o lib/Pool.o -o pfsp_gpu_hip.out -lm -lcudart -L/usr/local/cuda-11.2/targets/x86_64-linux/lib/
