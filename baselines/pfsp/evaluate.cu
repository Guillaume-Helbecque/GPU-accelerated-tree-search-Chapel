#ifdef __cplusplus
extern "C" {
#endif
  
#include "evaluate.h"
#include "stdlib.h" 
#include "lib/c_bounds_gpu.cu"
  
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
  __global__ void evaluate_gpu_lb1 (const int jobs, const int size, Node* parents_d, const lb1_bound_data  lbound1_d, int* bounds/*, int *front, int *back, int *remain*/)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < size) {
      const int parentId = threadId / jobs; 
      const int k = threadId % jobs; 
      Node parent =  parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;

      // Approach with parents_d as int** (same for lb2)
      // int prmu[MAX_JOBS];
      // // Variables from parent nodes
      // for(int i = 0; i < MAX_JOBS; i++)
      // 	prmu[i] = parents_d[parentId*(MAX_JOBS+2) + i];
      // const int depth = parents_d[parentId*(MAX_JOBS+2) + 20];
      // const int limit1 = parents_d[parentId*(MAX_JOBS+2) + 21];
  
      // We evaluate all permutations by varying index k from limit1 forward
      if (k >= limit1+1) {
	//swap_cuda(&prmu[depth],&prmu[k]);
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
	lb1_bound_gpu(lbound1_d, parent.prmu, limit1+1, jobs, &bounds[threadId]/*, front, back, remain*/);
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
	//swap_cuda(&prmu[depth],&prmu[k]);
      }
    }
  }

  /*
    NOTE: This lower bound evaluates all the children of a given parent at the same time.
    Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
    to the other lower bounds.
  */
  // Evaluate a bulk of parent nodes on GPU using lb1_d.
  __global__ void evaluate_gpu_lb1_d(const int jobs, const int size, int best, Node* parents_d, const lb1_bound_data lbound1_d, int* bounds)
  {
    int parentId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(parentId < size){ 
      Node parent = parents_d[parentId];
     
      // Vector of integers of size MAX_JOBS
      int lb_begin[MAX_JOBS];
    
      lb1_children_bounds_gpu(lbound1_d, parent.prmu, parent.limit1, jobs, lb_begin);

      for(int k = 0; k < jobs; k++) {
	if (k >= parent.limit1+1) {
	  const int job = parent.prmu[k];
	  bounds[parentId*jobs+k] = lb_begin[job];
	}
      }
    }
  }

  // Evaluate a bulk of parent nodes on GPU using lb2.
  __global__ void evaluate_gpu_lb2(const int jobs, const int size, int best, Node* parents_d, const lb1_bound_data lbound1_d, const lb2_bound_data lbound2_d, int* bounds)
  {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadId < size) {
      const int parentId = threadId / jobs; 
      const int k = threadId % jobs; 
      Node parent =  parents_d[parentId];
      int depth = parent.depth;
      int limit1 = parent.limit1;

      // We evaluate all permutations by varying index k from limit1 forward
      if (k >= limit1+1) {
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
	lb2_bound_gpu(lbound1_d, lbound2_d, parent.prmu, limit1+1, jobs, best, &bounds[threadId]);
	swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
      }
    }
  } 
 

  void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, const int nbBlocks_lb1_d, int* best, const lb1_bound_data lbound1, const lb2_bound_data lbound2, Node* parents, int* bounds)
  {
    // 1D grid of 1D blocks
    //dim3 gridDim(nbBlocks);      // nbBlocks blocks in x direction, y, z default to 1
    dim3 gridDim(nbBlocks_lb1_d);
    dim3 blockDim(BLOCK_SIZE);     // BLOCK_SIZE threads per block in x direction
    switch (lb) {
    case 0: // lb1_d
      //dim3 gridDim_lb1_d(nbBlocks_lb1_d);
      evaluate_gpu_lb1_d<<<nbBlocks_lb1_d, blockDim>>>(jobs, size, *best, parents, lbound1, bounds);
      return;
      break;

    case 1: // lb1
      //dim3 gridDim(nbBlocks);
      evaluate_gpu_lb1<<<nbBlocks, blockDim>>>(jobs, size, parents, lbound1, bounds/*, front, back, remain*/);
      return;
      break;

    case 2: // lb2
      //dim3 gridDim(nbBlocks);
      evaluate_gpu_lb2<<<nbBlocks, blockDim>>>(jobs, size, *best, parents, lbound1, lbound2, bounds);
      return;
      break;
    }
  }
#ifdef __cplusplus
}
#endif
