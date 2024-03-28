/*
  Sequential B&B to solve Taillard instances of the PFSP in C.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <getopt.h>
#include <time.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"

#define BLOCK_SIZE 512

//#define MIN(a, b) ((a) < (b) ? (a) : (b))

/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

#define MAX_JOBS 20

typedef struct
{
  uint8_t depth;
  int limit1;
  int prmu[MAX_JOBS];
} Node;

void initRoot(Node* root, const int jobs)
{
  root->depth = 0;
  root->limit1 = -1;
  for (int i = 0; i < jobs; i++) {
    root->prmu[i] = i;
  }
}

/*******************************************************************************
Implementation of a dynamic-sized single pool data structure.
Its initial capacity is 1024, and we reallocate a new container with double
the capacity when it is full. Since we perform only DFS, it only supports
'pushBack' and 'popBack' operations.
*******************************************************************************/

#define CAPACITY 1024

typedef struct
{
  Node* elements;
  int capacity;
  int size;
} SinglePool;

void initSinglePool(SinglePool* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->size = 0;
}

void pushBack(SinglePool* pool, Node node)
{
  if (pool->size >= pool->capacity) {
    pool->capacity *= 2;
    pool->elements = (Node*)realloc(pool->elements, pool->capacity * sizeof(Node));
  }

  pool->elements[pool->size++] = node;
}

Node popBack(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    return pool->elements[--pool->size];
  }

  return (Node){0};
}

void deleteSinglePool(SinglePool* pool)
{
  free(pool->elements);
}

/*******************************************************************************
Implementation of the parallel CUDA GPU PFSP search.
*******************************************************************************/

void parse_parameters(int argc, char* argv[], int* inst, int* lb, int* ub, int* m, int *M)
{
  *m = 25;
  *M = 50000;
  *inst = 14;
  *lb = 1;
  *ub = 1;
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
    {NULL, 0, NULL, 0} // Terminate options array
  };

  int opt, value;
  int option_index = 0;

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M", long_options, &option_index)) != -1) {
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
      if (value < 25 || value > 100) {
	fprintf(stderr, "Error: unsupported minimal pool for GPU initialization\n");
	exit(EXIT_FAILURE);
      }
      *m = value;
      break;

    case 'M':
      if (value < 45000 || value > 50000) {
	fprintf(stderr, "Error: unsupported maximal pool for GPU initialization\n");
	exit(EXIT_FAILURE);
      }
      *M = value;
      break;

    default:
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb)
{
  printf("\n=================================================\n");
  printf("Parallel GPU CUDA\n\n");
  printf("Resolution of PFSP Taillard's instance: ta%d (m = %d, n = %d) using parallel GPU CUDA\n", inst, machines, jobs);
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

inline void swap(int* a, int* b)
{
  int tmp = *b;
  *b = *a;
  *a = tmp;
}

// Evaluate and generate children nodes on CPU.
void decompose_lb1(const int jobs, const lb1_bound_data* const lbound1, const Node parent,
		   int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
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
		     int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
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
		   SinglePool* pool)
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

void decompose(const int jobs, const int lb, int* best,
	       const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, const Node parent,
	       unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
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

// Here I need to creat three functions evaluate_gpu that depend on the bounds and one that will be the responsible to decide which is going to be chosen

// Evaluate a bulk of parent nodes on GPU using lb1
// Here I am receiving parents_d as Node* and not const Node*, because of call of function swap
__global__ int* evaluate_gpu_lb1 (const int jobs, const int size, Node* parents_d, const lb1_bound_data* const lbound1_d, int* bounds)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < size) {
    const int parentId = threadId / jobs; 
    const int k = threadId % jobs; 
    Node parent = parents_d[parentId]; 
    const uint8_t depth = parent.depth;
    //int* prmu = parent.prmu; // I am not sure, but since parent.prmu is a table of int, a pointer should work 


    // We evaluate all permutations by varying index k from limit1 forward
    if (k >= parent.limit1+1) {
      swap(&parent.prmu[depth],&parent.prmu[k]);
      bounds[threadId] = lb1_bound(lbound1_d, parent.prmu, parent.limit1+1,jobs);
      swap(&parent.prmu[depth],&parent.prmu[k]);
    }
  }
}


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
    
    lb1_children_bounds(lbound1_d, parent.prmu, parent.limit1, jobs, lb_begin);

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
    //const int* prmu = parent.prmu; // I am not sure, but since parent.prmu is a table of int, a pointer should work 


    // We evaluate all permutations by varying index k from limit1 forward
    if (k >= parent.limit1+1) {
      swap(&parent.prmu[depth],&parent.prmu[k]);
      bounds[threadId] = lb2_bound(lbound1_d, lbound2_d, parent.prmu, parent.limit1+1, jobs, *best);
      swap(&parent.prmu[depth],&parent.prmu[k]);
    }
  }
}

// Maybe the parameters are appropriate now
int* evaluate_gpu(const int jobs, const int lb, const int size, int* best,
		  const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, Node* parent)
{
    
  switch (lb) {
  case 0: // lb1_d
    int bounds_lb1_d[size]; // Here check the size of bounds vector
    evaluate_gpu_lb1_d(jobs, size, best, parent, lbound1, bounds_lb1_d);
    return bounds_lb1_d;
    break;

  case 1: // lb1
    int bounds_lb1[size];
    evaluate_gpu_lb1(jobs, size, parent, lbound1, bounds_lb1);
    return bounds_lb1;
    break;

  case 2: // lb2
    int bounds_lb2[size];
    evaluate_gpu_lb2(jobs, size, best, parent, lbound1, lbound2, bounds_lb2);
    return bounds_lb2;
    break;
  }
}

// Generate children nodes (evaluated on GPU) on CPU
void generate_children(Node* parents, const int size, const int jobs, int* bounds,
		       unsigned long long int* exploredTree, unsigned long long int* exploredSol, int* best, SinglePool* pool)
{
  for (int i = 0; i < size; i++) {
    Node parent = parents[i];
    const uint8_t depth = parent.depth;

    for (int j = parent.limit1+1; j < jobs; j++) {
      const int lowerbound = bounds[j + i * jobs];

      // If child leaf
      if(depth + 1 == jobs){
	exploredSol += 1;

	// If child feasible
	if(lowerbound < best) &best = lowerbound;

      } else { // If not leaf
	if(lowerbound < best) {
	  Node child;
	  memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
	  swap(&child.prmu[parent.depth], &child.prmu[j]);
	  child.depth = parent.depth + 1;
	  child.limit1 = parent.limit1 + 1;
	  
	  pool.pushBack(child);
	  exploredTree += 1;
	}
      }
    }
  }
}

// Single-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, int* best,
		 unsigned long long int* exploredTree, unsigned long long int* exploredSol,
		 double* elapsedTime)
{
  // Initializing problem
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  // Starting pool
  Node root;
  initRoot(&root, jobs);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  // Timer
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  lb1_bound_data* lbound1;
  lbound1 = new_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1->p_times, inst);
  fill_min_heads_tails(lbound1);
    
  lb2_bound_data* lbound2;
  lbound2 = new_johnson_bd_data(lbound1);
  fill_machine_pairs(lbound2/*, LB2_FULL*/);
  fill_lags(lbound1->p_times, lbound2);
  fill_johnson_schedules(lbound1->p_times, lbound2);

  // I might need to add some lines here to check on the lbound1_d and lbound2_d (on the devices)
    
  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);

    int poolSize = MIN(pool.size,M);

    // If 'poolSize' is sufficiently large, we offload the pool on GPU.
    // When declaring mallocs inside we will have problems freeing memory
    if (poolSize >= m) {
      Node* parents = (Node*)malloc(poolSize * sizeof(Node));
      for(int i= 0; i < poolSize; i++) {
	int hasWork = 0;
	parents[i] = popBack(&pool,&hasWork);
	if (!hasWork) break;
      }
	
      /*
	TODO: Optimize 'numBounds' based on the fact that the maximum number of
	generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
	something like that.
      */
      const int  numBounds = jobs * poolSize;
      int* bounds;//[numBounds];

      Node* parents_d;
      cudaMalloc(&parents_d, M * sizeof(Node));
      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);

      const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);

      // count += 1;
      // Here should it be lbound1_d, lbound2_d or lbound1, lbound2?
      bounds = evaluate_gpu<<<nbBlocks, BLOCK_SIZE>>>(jobs, lb, poolSize, best, lbound1, lbound2, parents_d);

      // Here we do not have labels, we have the bound data
      // cudaMemcpy(labels, labels_d, numLabels * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	  
      /*
	Each task generates and inserts its children nodes to the pool.
      */
      generate_children(parents, poolSize,jobs, bounds, exploredTree, exploredSol, best, &pool);
    }
  }

  // Attention to these free data
  // labels are represented by the bounds now and they have their own freeing functions (see below)
  cudaFree(parents_d);
  free(parents);
        
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  *elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nExploration terminated.\n");

  // Freeing memory
  deleteSinglePool(&pool);
  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);
}

int main(int argc, char* argv[])
{
  int inst, lb, ub, m, M;
  parse_parameters(argc, argv, &inst, &lb, &ub, &m, &M);

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs, ub, lb);

  int optimum = (ub == 1) ? taillard_get_best_ub(inst) : INT_MAX;
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  pfsp_search(inst, lb, m, M, &optimum, &exploredTree, &exploredSol, &elapsedTime);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  return 0;
}
