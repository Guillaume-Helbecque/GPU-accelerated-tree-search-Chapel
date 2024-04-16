extern "C" {
#include "evaluate.h"
#include "stdlib.h" 
#include "lib/c_bound_simple_gpu_cuda.cu"
  //#include "lib/c_bound_johnson_gpu_cuda.cu"

  __device__ void swap_cuda(int* a, int* b)
  {
    int tmp = *b;
    *b = *a;
    *a = tmp;
  }

  void printDims(dim3 gridDim, dim3 blockDim) {
    printf("Grid Dimensions : [%d, %d, %d] blocks. \n",
    gridDim.x, gridDim.y, gridDim.z);

    printf("Block Dimensions : [%d, %d, %d] threads.\n",
    blockDim.x, blockDim.y, blockDim.z);
}

  // Evaluate a bulk of parent nodes on GPU using lb1
  __global__ void evaluate_gpu_lb1 (const int jobs, const int size, Node* parents_d, const lb1_bound_data* const lbound1_d, int* bounds)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("thread Id: %d\n", threadId);
    if (threadId < size) {
      const int parentId = threadId / jobs; 
      const int k = threadId % jobs; 
      Node parent = parents_d[parentId]; 
      const uint8_t depth = parent.depth;
  
      // We evaluate all permutations by varying index k from limit1 forward
      if (k >= parent.limit1+1) {
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
	//bounds[threadId] =
	lb1_bound_gpu(lbound1_d, parent.prmu, parent.limit1+1,jobs,&bounds[threadId]);
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
      }
    }
  }

  //Still need to solve lb1_d index
  /*
    NOTE: This lower bound evaluates all the children of a given parent at the same time.
    Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
    to the other lower bounds.
  */
  // Evaluate a bulk of parent nodes on GPU using lb1_d.
  __global__ void evaluate_gpu_lb1_d(const int jobs, const int size, const int* best, Node* parents_d, const lb1_bound_data* const lbound1_d, int* bounds)
  {
    // How does the NOTE translates into CUDA indices for searching only the parent nodes?
    int parentId = blockIdx.x * blockDim.x + threadIdx.x; // How to manage the proper indices?
    // I think that here maybe we do not to run through the threads ? 
    if(parentId < size/jobs){ 
      Node parent = parents_d[parentId];
      //const uint8_t depth = parent.depth; //not needed
      //const int* prmu = parent.prmu;

      // Vector of integers of size MAX_JOBS
      int lb_begin[MAX_JOBS];
    
      lb1_children_bounds_gpu(lbound1_d, parent.prmu, parent.limit1, jobs, lb_begin);

      // Going through the children for each parent node ?
      for(int k = 0; k < jobs; k++) {
	if (k >= parent.limit1+1) {
	  const int job = parent.prmu[k];
	  bounds[parentId*jobs+k] = lb_begin[job];
	}
      }
    }
  }

  // Evaluate a bulk of parent nodes on GPU using lb2.
  __global__ void evaluate_gpu_lb2(const int jobs, const int size, int* best, Node* parents_d, const lb1_bound_data* const lbound1_d, const lb2_bound_data* const lbound2_d, int* bounds)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < size) {
      const int parentId = threadId / jobs; 
      const int k = threadId % jobs; 
      Node parent = parents_d[parentId];
      const uint8_t depth = parent.depth;
  
      // We evaluate all permutations by varying index k from limit1 forward
      if (k >= parent.limit1+1) {
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
	bounds[threadId] = lb2_bound_gpu(lbound1_d, lbound2_d, parent.prmu, parent.limit1+1, jobs, *best);
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
      }
    }
  }


  void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, int* best, const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, Node* parents, int* bounds)
  {
    // 1D grid of 1D blocks
    dim3 gridDim(nbBlocks);      // nbBlocks blocks in x direction, y, z default to 1
    dim3 blockDim(BLOCK_SIZE);     // BLOCK_SIZE threads per block in x direction
    // printDims(gridDim, blockDim);
    switch (lb) {
    case 0: // lb1_d
      evaluate_gpu_lb1_d<<<gridDim, blockDim>>>(jobs, size, best, parents, lbound1, bounds);
      return;
      break;

    case 1: // lb1
      evaluate_gpu_lb1<<<gridDim, blockDim>>>(jobs, size, parents, lbound1, bounds);
      return;
      break;

    case 2: // lb2
      evaluate_gpu_lb2<<<gridDim, blockDim>>>(jobs, size, best, parents, lbound1, lbound2, bounds);
      return;
      break;
    }
  }
}
