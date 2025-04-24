/*
  Single-GPU B&B to solve Taillard instances of the PFSP in C+CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <getopt.h>
#include <time.h>
#include <math.h>
#include <cuda.h>

#include "../commons/util.h"
#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"
#include "lib/evaluate.h"
#include "lib/Pool.h"

/******************************************************************************
CUDA error checking
******************************************************************************/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
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

  while ((opt = getopt_long(argc, argv, "i:l:u:m:M:", long_options, &option_index)) != -1) {
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

    default:
      fprintf(stderr, "Usage: %s --inst <value> --lb <value> --ub <value> --m <value> --M <value>\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int inst, const int machines, const int jobs, const int ub, const int lb)
{
  printf("\n=================================================\n");
  printf("Single-GPU C+CUDA\n\n");
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

void print_results_file(const int inst, const int machines, const int jobs, const int lb, const int optimum,
  const unsigned long long int exploredTree, const unsigned long long int exploredSol, const double timer)
{
  FILE *file;
  file = fopen("stats_pfsp_gpu_cuda.dat","a");
  fprintf(file,"ta%d lb%d S-GPU %.4f %llu %llu %d\n",inst,lb,timer,exploredTree,exploredSol,optimum);
  fclose(file);
  return;
}

// Evaluate and generate children nodes on CPU.
void decompose_lb1(const int jobs, const lb1_bound_data* const lbound1, const Node parent,
  int* best, unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
{
  for (int i = parent.limit1+1; i < jobs; i++) {
    Node child;
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap_int(&child.prmu[parent.depth], &child.prmu[i]);

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
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1 + 1;
        memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
        swap_int(&child.prmu[child.limit1], &child.prmu[i]);

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
    child.depth = parent.depth + 1;
    child.limit1 = parent.limit1 + 1;
    memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
    swap_int(&child.prmu[parent.depth], &child.prmu[i]);

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

void decompose(const int jobs, const int lb, int* best, const lb1_bound_data* const lbound1,
  const lb2_bound_data* const lbound2, const Node parent, unsigned long long int* tree_loc,
  unsigned long long int* num_sol, SinglePool* pool)
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
void generate_children(Node* parents, const int size, const int jobs, int* bounds,
  unsigned long long int* exploredTree, unsigned long long int* exploredSol, int* best, SinglePool* pool)
{
  for (int i = 0; i < size; i++) {
    Node parent = parents[i];
    const uint8_t depth = parent.depth;

    for (int j = parent.limit1+1; j < jobs; j++) {
      const int lowerbound = bounds[j + i * jobs];

      // If child leaf
      if (depth + 1 == jobs) {
        *exploredSol += 1;

        // If child feasible
        if (lowerbound < *best) *best = lowerbound;

      } else { // If not leaf
        if (lowerbound < *best) {
          Node child;
          child.depth = depth + 1;
          child.limit1 = parent.limit1 + 1;
          memcpy(child.prmu, parent.prmu, jobs * sizeof(int));
          swap_int(&child.prmu[depth], &child.prmu[j]);

          pushBack(pool, child);
          *exploredTree += 1;
        }
      }
    }
  }
}

// Single-GPU PFSP search
void pfsp_search(const int inst, const int lb, const int m, const int M, int* best,
  unsigned long long int* exploredTree, unsigned long long int* exploredSol, double* elapsedTime)
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

  struct timespec start, end;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
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

  while (pool.size < m) {
    int hasWork = 0;
    Node parent = popFront(&pool, &hasWork);
    if (!hasWork) break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t1 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nInitial search on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t1);

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  // TODO: add function 'copyBoundsDevice' to perform the deep copy of bounding data
  // Vectors for deep copy of lbound1 to device
  lb1_bound_data lbound1_d;
  int* p_times_d;
  int* min_heads_d;
  int* min_tails_d;

  // Allocating and copying memory necessary for deep copy of lbound1
  cudaMalloc((void**)&p_times_d, jobs*machines*sizeof(int));
  cudaMalloc((void**)&min_heads_d, machines*sizeof(int));
  cudaMalloc((void**)&min_tails_d, machines*sizeof(int));
  cudaMemcpy(p_times_d, lbound1->p_times, (jobs*machines)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(min_heads_d, lbound1->min_heads, machines*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(min_tails_d, lbound1->min_tails, machines*sizeof(int), cudaMemcpyHostToDevice);

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
  cudaMalloc((void**)&johnson_schedule_d, (nb_mac_pairs*jobs) * sizeof(int));
  cudaMalloc((void**)&lags_d, (nb_mac_pairs*jobs) * sizeof(int));
  cudaMalloc((void**)&machine_pairs_1_d, nb_mac_pairs * sizeof(int));
  cudaMalloc((void**)&machine_pairs_2_d, nb_mac_pairs * sizeof(int));
  cudaMalloc((void**)&machine_pair_order_d, nb_mac_pairs * sizeof(int));
  cudaMemcpy(johnson_schedule_d, lbound2->johnson_schedules, (nb_mac_pairs*jobs) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(lags_d, lbound2->lags, (nb_mac_pairs*jobs) * sizeof(int), cudaMemcpyHostToDevice);
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

  Node* parents = (Node*)malloc(M * sizeof(Node));
  int* bounds = (int*)malloc((jobs*M) * sizeof(int));

  Node* parents_d;
  int *bounds_d;
  cudaMalloc((void**)&parents_d, M * sizeof(Node));
  cudaMalloc((void**)&bounds_d, (jobs*M) * sizeof(int));

  while (1) {
    int poolSize = popBackBulk(&pool, m, M, parents);

    if (poolSize > 0) {
      /*
        TODO: Optimize 'numBounds' based on the fact that the maximum number of
        generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
        something like that.
      */
      const int numBounds = jobs * poolSize;
      const int nbBlocks = ceil((double)numBounds / BLOCK_SIZE);

      cudaMemcpy(parents_d, parents, poolSize *sizeof(Node), cudaMemcpyHostToDevice);
      evaluate_gpu(jobs, lb, numBounds, nbBlocks, best, lbound1_d, lbound2_d, parents_d, bounds_d);
      cudaMemcpy(bounds, bounds_d, numBounds * sizeof(int), cudaMemcpyDeviceToHost);

      /*
        each task generates and inserts its children nodes to the pool.
      */
      generate_children(parents, poolSize, jobs, bounds, exploredTree, exploredSol, best, &pool);
    }
    else {
      break;
    }
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t2 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nSearch on GPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t2);

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t3 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  *elapsedTime = t1 + t2 + t3;

  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t3);

  printf("\nExploration terminated.\n");

  cudaFree(p_times_d);
  cudaFree(min_heads_d);
  cudaFree(min_tails_d);
  cudaFree(johnson_schedule_d);
  cudaFree(lags_d);
  cudaFree(machine_pairs_1_d);
  cudaFree(machine_pairs_2_d);
  cudaFree(machine_pair_order_d);
  cudaFree(parents_d);
  cudaFree(bounds_d);

  free_bound_data(lbound1);
  free_johnson_bd_data(lbound2);
  free(parents);
  free(bounds);

  deleteSinglePool(&pool);
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

  print_results_file(inst, machines, jobs, lb, optimum, exploredTree, exploredSol, elapsedTime);

  return 0;
}
