/*
  Multi-GPU B&B to solve Taillard instances of the PFSP in C+CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "lib/Pool_ext.h"
#include "lib/Auxiliary.h"
#include "lib/PFSP_node.h"

/*****************************************************************************
Bouding functions
*****************************************************************************/
//---------------One-machine bound functions-------------------

__device__ inline void
add_forward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * front)
{
  front[0] += p_times[job];
  for (int j = 1; j < nb_machines; j++) {
    front[j] = MAX(front[j - 1], front[j]) + p_times[j * nb_jobs + job];
  }
}

__device__ inline void
add_backward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * back)
{
  int j = nb_machines - 1;

  back[j] += p_times[j * nb_jobs + job];
  for (int j = nb_machines - 2; j >= 0; j--) {
    back[j] = MAX(back[j], back[j + 1]) + p_times[j * nb_jobs + job];
  }
}

__device__ void
schedule_front_gpu(const lb1_bound_data lb1_data, const int * const permutation, const int limit1, int * front)
{
  const int N = lb1_data.nb_jobs;
  const int M = lb1_data.nb_machines;
  const int *const p_times = lb1_data.p_times;

  if (limit1 == -1) {
    for (int i = 0; i < M; i++)
      front[i] = lb1_data.min_heads[i];
    return;
  }
  for (int i = 0; i < M; i++){
    front[i] = 0;
  }
  for (int i = 0; i < limit1 + 1; i++) {
    add_forward_gpu(permutation[i], p_times, N, M, front);
  }
}

__device__ void
schedule_back_gpu(const lb1_bound_data lb1_data, const int * const permutation, const int limit2, int * back)
{
  const int N = lb1_data.nb_jobs;
  const int M = lb1_data.nb_machines;
  const int *const p_times = lb1_data.p_times;

  if (limit2 == N) {
    for (int i = 0; i < M; i++)
      back[i] = lb1_data.min_tails[i];
    return;
  }

  for (int i = 0; i < M; i++) {
    back[i] = 0;
  }
  for (int k = N - 1; k >= limit2; k--) {
    add_backward_gpu(permutation[k], p_times, N, M, back);
  }
}

__device__ void
sum_unscheduled_gpu(const lb1_bound_data lb1_data, const int * const permutation, const int limit1, const int limit2, int * remain)
{
  const int nb_jobs = lb1_data.nb_jobs;
  const int nb_machines = lb1_data.nb_machines;
  const int * const p_times = lb1_data.p_times;

  //memset(remain, 0, nb_machines*sizeof(int));
  
  for (int j = 0; j < nb_machines; j++) {
    remain[j] = 0;
  }
  
  for (int k = limit1 + 1; k < limit2; k++) {
    const int job = permutation[k];
    for (int j = 0; j < nb_machines; j++) {
      remain[j] += p_times[j * nb_jobs + job];
    }
  }
}

__device__ int
machine_bound_from_parts_gpu(const int * const front, const int * const back, const int * const remain, const int nb_machines)
{
  int tmp0 = front[0] + remain[0];
  int lb = tmp0 + back[0]; // LB on machine 0
  int tmp1;

  for (int i = 1; i < nb_machines; i++) {
    tmp1 = MAX(tmp0, front[i] + remain[i]);
    lb = MAX(lb, tmp1 + back[i]);
    tmp0 = tmp1;
  }

  return lb;
}

// adds job to partial schedule in front and computes lower bound on optimal cost
// NB1: schedule is no longer needed at this point
// NB2: front, remain and back need to be set before calling this
// NB3: also compute total idle time added to partial schedule (can be used a criterion for job ordering)
// nOps : m*(3 add+2 max)  ---> O(m)
__device__ int
add_front_and_bound_gpu(const lb1_bound_data lb1_data, const int job, const int * const front, const int * const back, const int * const remain/*, int *delta_idle*/)
{
  int nb_jobs = lb1_data.nb_jobs;
  int nb_machines = lb1_data.nb_machines;
  int* p_times = lb1_data.p_times;

  int lb   = front[0] + remain[0] + back[0];
  int tmp0 = front[0] + p_times[job];
  int tmp1;

  int idle = 0;
  for (int i = 1; i < nb_machines; i++) {
    idle += MAX(0, tmp0 - front[i]);

    tmp1 = MAX(tmp0, front[i]);
    lb   = MAX(lb, tmp1 + remain[i] + back[i]);
    tmp0 = tmp1 + p_times[i * nb_jobs + job];
  }

  //can pass NULL
  // if (delta_idle) {
  //   delta_idle[job] = idle;
  // }

  return lb;
}

// ... same for back
__device__ int
add_back_and_bound_gpu(const lb1_bound_data lb1_data, const int job, const int * const front, const int * const back, const int * const remain, int *delta_idle)
{
  int nb_jobs = lb1_data.nb_jobs;
  int nb_machines = lb1_data.nb_machines;
  int* p_times = lb1_data.p_times;

  int last_machine = nb_machines - 1;

  int lb   = front[last_machine] + remain[last_machine] + back[last_machine];
  int tmp0 = back[last_machine] + p_times[last_machine*nb_jobs + job];
  int tmp1;

  int idle = 0;
  for (int i = last_machine-1; i >= 0; i--) {
    idle += MAX(0, tmp0 - back[i]);

    tmp1 = MAX(tmp0, back[i]);
    lb = MAX(lb, tmp1 + remain[i] + front[i]);
    tmp0 = tmp1 + p_times[i*nb_jobs + job];
  }

  //can pass NULL
  if (delta_idle) {
    delta_idle[job] = idle;
  }

  return lb;
}

__device__ void
lb1_bound_gpu(const lb1_bound_data lb1_data, const int * const permutation, const int limit1, const int limit2, int *bounds)
{
  int nb_machines = lb1_data.nb_machines;
  int front[MAX_MACHINES];
  int back[MAX_MACHINES];
  int remain[MAX_MACHINES];
  
  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);

  sum_unscheduled_gpu(lb1_data, permutation, limit1, limit2, remain);

  // Same as in function eval_solution_gpu
  *bounds = machine_bound_from_parts_gpu(front, back, remain, nb_machines);

  return;
}

__device__ void lb1_children_bounds_gpu(const lb1_bound_data lb1_data, const int *const permutation, const int limit1, const int limit2, int *const lb_begin/*, int *const lb_end, int *const prio_begin, int *const prio_end, const int direction*/)
{
  int N = lb1_data.nb_jobs;
  //int M = lb1_data.nb_machines;

  int front[MAX_MACHINES];
  int back[MAX_MACHINES];
  int remain[MAX_MACHINES];

  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);
  sum_unscheduled_gpu(lb1_data, permutation, limit1, limit2, remain);

  // switch (direction)  {
  //   case -1: //begin
  //   {
  //memset(lb_begin, 0, N*sizeof(int));
  // if (prio_begin) memset(prio_begin, 0, N*sizeof(int));
  for(int i = 0; i < N; i++)
    lb_begin[i]=0;

  
  for (int i = limit1+1; i < limit2; i++) {
    int job = permutation[i];
    lb_begin[job] = add_front_and_bound_gpu(lb1_data, job, front, back, remain/*, prio_begin*/);
  }
  //     break;
  //   }
  //   case 0: //begin-end
  //   {
  //     memset(lb_begin, 0, N*sizeof(int));
  //     memset(lb_end, 0, N*sizeof(int));
  //     if (prio_begin) memset(prio_begin, 0, N*sizeof(int));
  //     if (prio_end) memset(prio_end, 0, N*sizeof(int));
  //
  //     for (int i = limit1+1; i < limit2; i++) {
  //       int job = permutation[i];
  //       lb_begin[job] = add_front_and_bound(data, job, front, back, remain, prio_begin);
  //       lb_end[job] = add_back_and_bound(data, job, front, back, remain, prio_end);
  //     }
  //     break;
  //   }
  //   case 1: //end
  //   {
  //     memset(lb_end, 0, N*sizeof(int));
  //     if (prio_end) memset(prio_end, 0, N*sizeof(int));
  //
  //     for (int i = limit1+1; i < limit2; i++) {
  //       int job = permutation[i];
  //       lb_end[job] = add_back_and_bound(data, job, front, back, remain, prio_end);
  //     }
  //     break;
  //   }
  // }
}

//------------------Two-machine bound functions(johnson)---------------------------

__device__ int lb_makespan_gpu(int* lb1_p_times, const lb2_bound_data lb2_data, const int* const flag,int* front, int* back, const int minCmax)
{
  int lb = 0;
  int nb_jobs = lb2_data.nb_jobs;

  //for all machine-pairs : O(m^2) m*(m-1)/2
  for (int l = 0; l < lb2_data.nb_machine_pairs; l++) {
    int i = lb2_data.machine_pair_order[l];

    int ma0 = lb2_data.machine_pairs_1[i];
    int ma1 = lb2_data.machine_pairs_2[i];

    int tmp0 = front[ma0];
    int tmp1 = front[ma1];

    for (int j = 0; j < nb_jobs; j++) {
      int job = lb2_data.johnson_schedules[i*nb_jobs + j];
      // j-loop is on unscheduled jobs... (==0 if jobCour is unscheduled)
      if (flag[job] == 0) {
	int ptm0 = lb1_p_times[ma0*nb_jobs + job];
	int ptm1 = lb1_p_times[ma1*nb_jobs + job];
	int lag = lb2_data.lags[i*nb_jobs + job];
	// add job on ma0 and ma1
	tmp0 += ptm0;
	tmp1 = MAX(tmp1,tmp0 + lag);
	tmp1 += ptm1;
      }
    }
    
    tmp1 = MAX(tmp1 + back[ma1], tmp0 + back[ma0]);

    lb = MAX(lb, tmp1);
    
    if (lb > minCmax) {
      break;
    }
  }

  return lb;
}

__device__ void lb2_bound_gpu(const lb1_bound_data lb1_data, const lb2_bound_data lb2_data, const int* const permutation, const int limit1, const int limit2,const int best_cmax,int *bounds)
{
  const int N = lb1_data.nb_jobs;

  int front[MAX_MACHINES];
  int back[MAX_MACHINES];

  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);

  int flags[MAX_JOBS];
   
  // Set flags
  for (int i = 0; i < N; i++)
    flags[i] = 0;
  for (int j = 0; j <= limit1; j++)
    flags[permutation[j]] = 1;
  for (int j = limit2; j < N; j++)
    flags[permutation[j]] = 1;

  *bounds = lb_makespan_gpu(lb1_data.p_times, lb2_data, flags, front, back, best_cmax);
  return;
}



/******************************************************************************
Evaluate functions
******************************************************************************/

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
__global__ void evaluate_gpu_lb1 (const int jobs, const int size, Node* parents_d, const lb1_bound_data  lbound1_d, int* bounds)
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
      lb1_bound_gpu(lbound1_d, parent.prmu, limit1+1, jobs, &bounds[threadId]);
      swap_cuda(&parent.prmu[depth],&parent.prmu[k]);
    }
  }
}

/*
  NOTE: This lower bound evaluates all the children of a given parent at the same time.
  Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
  to the other lower bounds.
*/
// Evaluate a bulk of parent nodes on GPU using lb1_d.
__global__ void evaluate_gpu_lb1_d(const int jobs, const int size, Node* parents_d, const lb1_bound_data lbound1_d, int* bounds)
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
  // 1D grid of 1D nbBlocks(_lb1_d) blocks with block size BLOCK_SIZE
  switch (lb) {
  case 0: // lb1_d
    evaluate_gpu_lb1_d<<<nbBlocks_lb1_d, BLOCK_SIZE>>>(jobs, size, parents, lbound1, bounds);
    return;
    break;

  case 1: // lb1
    evaluate_gpu_lb1<<<nbBlocks, BLOCK_SIZE>>>(jobs, size, parents, lbound1, bounds);
    return;
    break;

  case 2: // lb2
    evaluate_gpu_lb2<<<nbBlocks, BLOCK_SIZE>>>(jobs, size, *best, parents, lbound1, lbound2, bounds);
    return;
    break;
  }
}

/*******************************************************************************
Implementation of the parallel Multi-GPU C+CUDA+OpenMP PFSP search.
*******************************************************************************/

void parse_parameters(int argc, char* argv[], int* inst, int* lb, int* ub, int* m, int *M, int *D)
{
  *m = 25;
  *M = 50000;
  *inst = 14;
  *lb = 1;
  *ub = 1;
  *D = 1;
  /*
    NOTE: Only forward branching is considered because other strategies increase a
    lot the implementation complexity and do not add much contribution.
  */
  
  // Define long options
  static struct option long_options[] = {
    {"inst", required_argument, NULL, 'i'},
    {"lb", required_argument, NULL, 'l'},
    {"ub", required_argument, NULL, 'u'},
    {"m", required_argument, NULL, 'm'},
    {"M", required_argument, NULL, 'M'},
    {"D", required_argument, NULL, 'D'},
    {NULL, 0, NULL, 0} // Terminate options array
  };

  int opt, value;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M:D:", long_options, &option_index)) != -1) {
    value = atoi(optarg);

    switch (opt) {
    case 'i':
      if (value < 1 || value > 120) {
	fprintf(stderr, "Error: unsupported Taillard's instance\n");
	exit(EXIT_FAILURE);
      }
      *inst = value;
      break;

    case 'l':
      if (value < 0 || value > 2) {
	fprintf(stderr, "Error: unsupported lower bound function\n");
	exit(EXIT_FAILURE);
      }
      *lb = value;
      break;

    case 'u':
      if (value != 0 && value != 1) {
	fprintf(stderr, "Error: unsupported upper bound initialization\n");
	exit(EXIT_FAILURE);
      }
      *ub = value;
      break;

    case 'm':
      if (value < 1) {
	fprintf(stderr, "Error: unsupported minimal pool for GPU initialization\n");
	exit(EXIT_FAILURE);
      }
      *m = value;
      break;

    case 'M':
      if (value < *m) {
	fprintf(stderr, "Error: unsupported maximal pool for GPU initialization\n");
	exit(EXIT_FAILURE);
      }
      *M = value;
      break;

    case 'D':
      if (value < 0 || value > 16) {
	fprintf(stderr, "Error: unsupported number of GPU's\n");
	exit(EXIT_FAILURE);
      }
      *D = value;
      break;

    default:
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value> --D <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

 
void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb, const int D)
{
  printf("\n=================================================\n");
  printf("Multi-GPU C+CUDA+OpenMP %d GPU's\n\n", D);
  printf("Resolution of PFSP Taillard's instance: ta%d (m = %d, n = %d)\n", inst, machines, jobs);
  if (ub == 0) printf("Initial upper bound: inf\n");
  else /* if (ub == 1) */ printf("Initial upper bound: opt\n");
  if (lb == 0) printf("Lower bound function: lb1_d\n");
  else if (lb == 1) printf("Lower bound function: lb1\n");
  else /* (lb == 2) */ printf("Lower bound function: lb2\n");
  printf("Branching rule: fwd\n");
  printf("=================================================\n");
}

void print_results(const int optimum, const unsigned long long int exploredTree,
		   const unsigned long long int exploredSol, const double timer)
{
  printf("\n=================================================\n");
  printf("Size of the explored tree: %llu\n", exploredTree);
  printf("Number of explored solutions: %llu\n", exploredSol);
  /* TODO: Add 'is_better' */
  printf("Optimal makespan: %d\n", optimum);
  printf("Elapsed time: %.4f [s]\n", timer);
  printf("=================================================\n");
}

void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int D, const int optimum,
			const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer)
{
  FILE *file;
  file = fopen("stats_pfsp_multigpu_cuda_dyn.dat","a");
  fprintf(file,"ta%d lb%d %dGPU %.4f %llu %llu %d\n",inst,lb,D,timer,exploredTree,exploredSol,optimum);
  fclose(file);
  return;
}

inline void swap(int* a, int* b)
{
  int tmp = *b;
  *b = *a;
  *a = tmp;
}

// Evaluate and generate children nodes on CPU.
void decompose_lb1(const int jobs, const lb1_bound_data* const lbound1, const Node parent,
		   int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool_ext* pool)
{
  for (int i = parent.limit1+1; i < jobs; i++) {
    Node child;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;

    int lowerbound = lb1_bound(lbound1, child.prmu, child.limit1, jobs);

    if (child.depth == jobs) { // if child leaf
      *num_sol += 1;

      if (lowerbound < *best) { // if child feasible
        *best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < *best) { // if child feasible
        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

void decompose_lb1_d(const int jobs, const lb1_bound_data* const lbound1, const Node parent,
		     int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool_ext* pool)
{
  int* lb_begin = (int*)malloc(jobs * sizeof(int));

  lb1_children_bounds(lbound1, parent.prmu, parent.limit1, jobs, lb_begin);

  for (int i = parent.limit1+1; i < jobs; i++) {
    const int job = parent.prmu[i];
    const int lb = lb_begin[job];

    if (parent.depth + 1 == jobs) { // if child leaf
      *num_sol += 1;

      if (lb < *best) { // if child feasible
        *best = lb;
      }
    } else { // if not leaf
      if (lb < *best) { // if child feasible
        Node child;
        memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1 + 1;
        swap(&child.prmu[child.limit1], &child.prmu[i]);

        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }

  free(lb_begin);
}

void decompose_lb2(const int jobs, const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2,
		   const Node parent, int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol,
		   SinglePool_ext* pool)
{
  for (int i = parent.limit1+1; i < jobs; i++) {
    Node child;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap(&child.prmu[parent.depth], &child.prmu[i]);
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;

    int lowerbound = lb2_bound(lbound1, lbound2, child.prmu, child.limit1, jobs, *best);

    if (child.depth == jobs) { // if child leaf
      *num_sol += 1;

      if (lowerbound < *best) { // if child feasible
        *best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < *best) { // if child feasible
        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

void decompose(const int jobs, const int lb, int* best, const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, const Node parent, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool_ext* pool)
{
  switch (lb) {
  case 0: // lb1_d
    decompose_lb1_d(jobs, lbound1, parent, best, tree_loc, num_sol, pool);
    break;

  case 1: // lb1
    decompose_lb1(jobs, lbound1, parent, best, tree_loc, num_sol, pool);
    break;

  case 2: // lb2
    decompose_lb2(jobs, lbound1, lbound2, parent, best, tree_loc, num_sol, pool);
    break;
  }
}

// Generate children nodes (evaluated on GPU) on CPU
void generate_children(Node* parents, const int size, const int jobs, int* bounds, unsigned long long int* exploredTree, unsigned long long int* exploredSol, int* best, SinglePool_ext* pool)
{
  for (int i = 0; i < size; i++) {
    Node parent = parents[i];
    const uint8_t depth = parent.depth;

    for (int j = parent.limit1+1; j < jobs; j++) {
      const int lowerbound = bounds[j + i * jobs];

      // If child leaf
      if(depth + 1 == jobs){
	*exploredSol += 1;

	// If child feasible
	if(lowerbound < *best) *best = lowerbound;

      } else { // If not leaf
	if(lowerbound < *best) {
	  Node child;
	  memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
	  swap(&child.prmu[depth], &child.prmu[j]);
	  child.depth = depth + 1;
	  child.limit1 = parent.limit1 + 1;
	  
	  pushBack(pool, child);
	  *exploredTree += 1;
	}
      }
    }
  }
}

// Single-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, const int D, int* best,
		 unsigned long long int* exploredTree, unsigned long long int* exploredSol,
		 double* elapsedTime)
{
  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);
   
  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool_ext pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  // Boolean variables for dynamic workload balance
  _Atomic bool allTasksIdleFlag = false;
  _Atomic bool eachTaskState[D]; // one task per GPU
  for(int i = 0; i < D; i++)//{
    atomic_store(&eachTaskState[i],false);
 
  // Timer
  double startTime, endTime;
  startTime = omp_get_wtime();
  
  // Bounding data
  lb1_bound_data* lbound1;
  lbound1 = new_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1->p_times, inst);
  fill_min_heads_tails(lbound1);
    
  lb2_bound_data* lbound2;
  lbound2 = new_johnson_bd_data(lbound1);
  fill_machine_pairs(lbound2/*, LB2_FULL*/);
  fill_lags(lbound1->p_times, lbound2);
  fill_johnson_schedules(lbound1->p_times, lbound2);

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  
  while(pool.size < D*m) {
    // CPU side
    int hasWork = 0;
    Node parent = popFront(&pool, &hasWork);
    if (!hasWork) break;
    
    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }
  endTime = omp_get_wtime();
  double t1 = endTime - startTime;

  printf("\nInitial search on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t1);

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */

  startTime = omp_get_wtime();
  unsigned long long int eachExploredTree[D], eachExploredSol[D];
  int eachBest[D];
  
  const int poolSize = pool.size;
  const int c = poolSize / D;
  const int l = poolSize - (D-1)*c;
  const int f = pool.front;
   
  pool.front = 0;
  pool.size = 0;

  SinglePool_ext multiPool[D];
  for(int i = 0; i < D; i++)
    initSinglePool(&multiPool[i]);

  //int best_l = *best;

#pragma omp parallel for num_threads(D) shared(eachExploredTree, eachExploredSol, eachBest, eachTaskState, pool, multiPool, lbound1, lbound2) //reduction(min:best_l)
  for (int gpuID = 0; gpuID < D; gpuID++) {
    cudaSetDevice(gpuID);

    printf("Hello from thread %d \n", omp_get_thread_num());
    
    int nSteal = 0, nSSteal = 0;
    
    unsigned long long int tree = 0, sol = 0;
    SinglePool_ext* pool_loc;
    pool_loc = &multiPool[gpuID]; 
    int best_l = *best;
    bool taskState = false;
    bool expected = false;

    // each task gets its chunk
    for (int i = 0; i < c; i++) {
      pool_loc->elements[i] = pool.elements[gpuID+f+i*D];
    }
    pool_loc->size += c;
    if (gpuID == D-1) {
      for (int i = c; i < l; i++) {
        pool_loc->elements[i] = pool.elements[(D*c)+f+i-c];
      }
      pool_loc->size += l-c;
    }
    
    // TODO: add function 'copyBoundsDevice' to perform the deep copy of bounding data
    // Vectors for deep copy of lbound1 to device
    lb1_bound_data lbound1_d;
    int* p_times_d;
    int* min_heads_d;
    int* min_tails_d;

    // Allocating and copying memory necessary for deep copy of lbound1
    cudaMalloc((void**)&p_times_d, jobs * machines * sizeof(int));
    cudaMalloc((void**)&min_heads_d, machines * sizeof(int));
    cudaMalloc((void**)&min_tails_d, machines * sizeof(int));
    cudaMemcpy(p_times_d, lbound1->p_times, jobs * machines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_heads_d, lbound1->min_heads, machines * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(min_tails_d, lbound1->min_tails, machines * sizeof(int), cudaMemcpyHostToDevice);

    // Deep copy of lbound1
    lbound1_d.p_times = p_times_d;
    lbound1_d.min_heads = min_heads_d;
    lbound1_d.min_tails = min_tails_d;
    lbound1_d.nb_jobs = lbound1->nb_jobs;
    lbound1_d.nb_machines = lbound1->nb_machines;

    // Vectors for deep copy of lbound2 to device
    lb2_bound_data lbound2_d;
    int *johnson_schedule_d;
    int *lags_d;
    int *machine_pairs_1_d;
    int *machine_pairs_2_d;
    int *machine_pair_order_d;

    // Allocating and copying memory necessary for deep copy of lbound2
    int nb_mac_pairs = lbound2->nb_machine_pairs;
    cudaMalloc((void**)&johnson_schedule_d, nb_mac_pairs * jobs * sizeof(int));
    cudaMalloc((void**)&lags_d, nb_mac_pairs * jobs * sizeof(int));
    cudaMalloc((void**)&machine_pairs_1_d, nb_mac_pairs * sizeof(int));
    cudaMalloc((void**)&machine_pairs_2_d, nb_mac_pairs * sizeof(int));
    cudaMalloc((void**)&machine_pair_order_d, nb_mac_pairs * sizeof(int));
    cudaMemcpy(johnson_schedule_d, lbound2->johnson_schedules, nb_mac_pairs * jobs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(lags_d, lbound2->lags, nb_mac_pairs * jobs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(machine_pairs_1_d, lbound2->machine_pairs_1, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(machine_pairs_2_d, lbound2->machine_pairs_2, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(machine_pair_order_d, lbound2->machine_pair_order, nb_mac_pairs * sizeof(int), cudaMemcpyHostToDevice);

    // Deep copy of lbound2
    lbound2_d.johnson_schedules = johnson_schedule_d;
    lbound2_d.lags = lags_d;
    lbound2_d.machine_pairs_1 = machine_pairs_1_d;
    lbound2_d.machine_pairs_2 = machine_pairs_2_d;
    lbound2_d.machine_pair_order = machine_pair_order_d;
    lbound2_d.nb_machine_pairs = lbound2->nb_machine_pairs;
    lbound2_d.nb_jobs = lbound2->nb_jobs;
    lbound2_d.nb_machines = lbound2->nb_machines;
    
    // Allocating parents vector on CPU and GPU
    Node* parents = (Node*)malloc(M * sizeof(Node));
    Node* parents_d;
    cudaMalloc((void**)&parents_d, M * sizeof(Node));
    
    // Allocating bounds vector on CPU and GPU
    int* bounds = (int*)malloc((jobs*M) * sizeof(int));
    int *bounds_d;
    cudaMalloc((void**)&bounds_d, (jobs*M) * sizeof(int));

    while (1) {
      // Dynamic workload balance
      /*
	Each task gets its parenst nodes from the pool
      */

      int poolSize = popBackBulk(pool_loc, m, M, parents);
      
      if (poolSize > 0) {
        if (taskState == true) {
          taskState = false;
          atomic_store(&eachTaskState[gpuID],false);
        }
      
	/*
	  TODO: Optimize 'numBounds' based on the fact that the maximum number of
	  generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
	  something like that.
	*/
	const int numBounds = jobs * poolSize;   
	const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);
	const int nbBlocks_lb1_d = ceil((double)nbBlocks/jobs); 

	cudaMemcpy(parents_d, parents, poolSize *sizeof(Node), cudaMemcpyHostToDevice);

	// numBounds is the 'size' of the problem
 	evaluate_gpu(jobs, lb, numBounds, nbBlocks, nbBlocks_lb1_d, &best_l, lbound1_d, lbound2_d, parents_d, bounds_d); 
      
        cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);

	/*
	  each task generates and inserts its children nodes to the pool.
	*/
	generate_children(parents, poolSize, jobs, bounds, &tree, &sol, &best_l, pool_loc);
      }
      else {
        // work stealing
	int tries = 0;
        bool steal = false;
	int victims[D];
	permute(victims,D);

	//int breakWS0 = 0;
	
        while (tries < D && steal == false) { //WS0 loop
          int victimID = victims[tries];
	  
          if (victimID != gpuID) { // if not me
            SinglePool_ext* victim;
	    victim = &multiPool[victimID];
            nSteal ++;
            int nn = 0;
	   
            while (nn < 10) { //WS1 loop
	      expected = false;
	      if (atomic_compare_exchange_strong(&(victim->lock), &expected, true)){ // get the lock
		int size = victim->size;
				
		if (size >= 2*m) {
		  Node* p = popBackBulkFree(victim, m, M, &size); // NO atomic_store inside
		  
		  if (size == 0) { // there is no more work
		    atomic_store(&(victim->lock), false); // reset lock
		    printf("\nDEADCODE\n");
		    //fprintf(stderr, "DEADCODE in work stealing\n");
		    //exit(EXIT_FAILURE);
		  }
		  
		  /* for i in 0..#(size/2) {
		     pool_loc.pushBack(p[i]);
		     } */
		  
		  pushBackBulk(pool_loc, p, size); // atomic_store inside
		  
		  steal = true;
		  nSSteal++;
		  atomic_store(&(victim->lock), false); // reset lock
		  //breakWS0 = 1;
		  goto WS0; // Break out of WS0 loop
		}

		//if(breakWS0 == 1) break; //break of WS0 loop
		
		atomic_store(&(victim->lock), false);// reset lock
		break; // Break out of WS1 loop
	      }
	      
	      nn ++;
	    }
	  }
	
	  tries ++;
	}
      WS0:

	if (steal == false) {
	  // termination
	  if (taskState == false) {
	    taskState = true;
	    atomic_store(&eachTaskState[gpuID],true);
	  }
	  if (allIdle(eachTaskState, D, &allTasksIdleFlag)) {
	    //printf("Termination of the second step");
	    // writeln("task ", gpuID, " exits normally");
	    break;
	  }
	  continue;
	} else {
	  continue;
	}
      }
    }

    printf("\nFor thread[%d] nb of stealing tries is %d and real nb of stealing is %d\n", omp_get_thread_num(), nSteal, nSSteal);
    
    // This comment has to go after the problem is solved
    //double time_partial = omp_get_wtime();
    //printf("\nTime for GPU[%d] = %f, nb of nodes = %lld, nb of sols = %lld\n", gpuID, time_partial - startTime, tree, sol);
    
    // OpenMP environment freeing variables
    cudaFree(parents_d);
    cudaFree(bounds_d);
    cudaFree(p_times_d);
    cudaFree(min_heads_d);
    cudaFree(min_tails_d);
    cudaFree(johnson_schedule_d);
    cudaFree(lags_d);
    cudaFree(machine_pairs_1_d);
    cudaFree(machine_pairs_2_d);
    cudaFree(machine_pair_order_d);
    free(parents);
    free(bounds);

#pragma omp critical
    {
      const int poolLocSize = pool_loc->size;
      for (int i = 0; i < poolLocSize; i++) {
	int hasWork = 0;
	pushBack(&pool, popBack(pool_loc, &hasWork));
	if (!hasWork) break;
      }
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
    eachBest[gpuID] = best_l;

    deleteSinglePool_ext(pool_loc);

  } // End of parallel region
  
  for (int i = 0; i < D; i++) {
    *exploredTree += eachExploredTree[i];
    *exploredSol += eachExploredSol[i];
  }
  *best = findMin(eachBest,D);
  
  endTime = omp_get_wtime();
  double t2 = endTime - startTime;

  printf("\nSearch on GPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);
  for(int gpuID = 0; gpuID < D; gpuID++)
    printf("Workload for GPU[%d]: %f\n", gpuID,(double)100*eachExploredTree[gpuID]/((double)*exploredTree));
    
  /*
    Step 3: We complete the depth-first search on CPU.
  */

  startTime = omp_get_wtime();
  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }

  // Freeing memory for structs common to all steps 
  deleteSinglePool_ext(&pool);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);
  
  endTime = omp_get_wtime();
  double t3 = endTime - startTime;
  *elapsedTime = t1 + t2 + t3;
  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t3);

  printf("\nExploration terminated.\n");
}

int main(int argc, char* argv[])
{
  srand(time(NULL));
  
  int inst, lb, ub, m, M, D;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M, &D);
    
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb, D);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;
  
  pfsp_search(inst, lb, m, M, D, &optimum, &exploredTree, &exploredSol, &elapsedTime);
  
  print_results(optimum, exploredTree, exploredSol, elapsedTime);
  
  print_results_file(inst, machines, jobs, lb, D, optimum, exploredTree, exploredSol, elapsedTime);
  
  return 0;
}
