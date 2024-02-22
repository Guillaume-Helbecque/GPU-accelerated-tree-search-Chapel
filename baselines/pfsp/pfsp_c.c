/*
  Sequential B&B to solve Taillard instances of the PFSP in C.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "lib/c_bound_simple.h"
#include "lib/c_bound_johnson.h"
#include "lib/c_taillard.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

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
Implementation of the sequential PFSP search.
*******************************************************************************/

// void parse_parameters(int argc, char* argv[], int* N, int* G, int* m, int* M)
// {
//   *N = 14;
//   *G = 1;
//   *m = 25;
//   *M = 50000;
//
//   int opt, value;
//
//   while ((opt = getopt(argc, argv, "N:g:m:M:")) != -1) {
//     value = atoi(optarg);
//
//     if (value <= 0) {
//       printf("All parameters must be positive integers.\n");
//       exit(EXIT_FAILURE);
//     }
//
//     switch (opt) {
//       case 'N':
//         *N = value;
//         break;
//       case 'g':
//         *G = value;
//         break;
//       case 'm':
//         *m = value;
//         break;
//       case 'M':
//         *M = value;
//         break;
//       default:
//         fprintf(stderr, "Usage: %s -N value -g value -m value -M value\n", argv[0]);
//         exit(EXIT_FAILURE);
//     }
//   }
// }

void print_settings(const int inst, const int machines, const int jobs)
{
  printf("\n=================================================\n");
  printf("Resolution of PFSP Taillard's instance: %d (m = %d, n = %d) using C+CUDA\n", inst, machines, jobs);
  printf("Initial upper bound: opt\n");
  printf("Lower bound function: lb1\n");
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
  printf("Optimal makespan: %d", optimum);
  printf("Elapsed time: %.4f [s]\n", timer);
  printf("=================================================\n");
}

inline void swap(int* a, int* b)
{
  uint8_t tmp = *b;
  *b = *a;
  *a = tmp;
}

// Evaluate and generate children nodes on CPU.
void decompose_lb1(const int jobs, const bound_data* const lbound1, const Node parent,
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

void decompose(const int jobs, const int lb, int* best,
  const bound_data* const lbound1, const johnson_bd_data* const lbound2, const Node parent,
  unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
{
  // switch (lb) {
  //   case "lb1" : {
      decompose_lb1(jobs, lbound1, parent, best, tree_loc, num_sol, pool);
  //     break;
  //   }
  //   case "lb1_d" : {
  //     decompose_lb1_d(parent, tree_loc, num_sol, best, pool);
  //     break;
  //   }
  //   case "lb2" : {
  //     decompose_lb2(parent, tree_loc, num_sol, best, pool);
  //     break;
  //   }
  //   default :
  //     halt("DEADCODE");
  //   }
  // }
}

// Sequential N-Queens search.
void pfsp_search(const int inst, const int lb, const int br, const int ub,
  int* best, unsigned long long int* exploredTree, unsigned long long int* exploredSol,
  double* elapsedTime)
{
  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  bound_data* lbound1;
  lbound1 = new_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1->p_times, inst);
  fill_min_heads_tails(lbound1);

  johnson_bd_data* lbound2;
  lbound2 = new_johnson_bd_data(lbound1);
  fill_machine_pairs(lbound2/*, LB2_FULL*/);
  fill_lags(lbound1, lbound2);
  fill_johnson_schedules(lbound1, lbound2);

  Node root;
  initRoot(&root, jobs);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(jobs, lb, best, lbound1, lbound2, parent, exploredTree, exploredSol, &pool);
  }
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  *elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nExploration terminated.");

  deleteSinglePool(&pool);
}

int main(int argc, char* argv[])
{
  // char* inst, lb, br, ub;
  // parse_parameters(argc, argv, &lb, &br, &ub);
  int inst = 14;
  int lb = 1;
  int br = 1;
  int ub = 1;

  int jobs = taillard_get_nb_jobs(inst);
  int machines = taillard_get_nb_machines(inst);

  print_settings(inst, machines, jobs);

  int optimum = 1377; // opt for ta14
  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  pfsp_search(inst, lb, br, ub, &optimum, &exploredTree, &exploredSol, &elapsedTime);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  return 0;
}
