/*
  Single-GPU backtracking to solve instances of the N-Queens problem in C+CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "cuda_runtime.h"

#include "lib/NQueens_node.h"
#include "lib/Pool.h"

#define BLOCK_SIZE 512

// #define MIN(a, b) ((a) < (b) ? (a) : (b))

/*******************************************************************************
Implementation of the single-GPU N-Queens search.
*******************************************************************************/

void parse_parameters(int argc, char* argv[], int* N, int* G, int* m, int* M)
{
  *N = 14;
  *G = 1;
  *m = 25;
  *M = 50000;

  int opt, value;

  while ((opt = getopt(argc, argv, "N:g:m:M:")) != -1) {
    value = atoi(optarg);

    switch (opt) {
      case 'N':
        if (value < 1) {
          fprintf(stderr, "Error: N must be a positive integer.\n");
          exit(EXIT_FAILURE);
        }
        *N = value;
        break;

      case 'g':
        if (value < 1) {
          fprintf(stderr, "Error: g must be a positive integer.\n");
          exit(EXIT_FAILURE);
        }
        *G = value;
        break;

      case 'm':
        if (value < 1) {
          fprintf(stderr, "Error: m must be a positive integer.\n");
          exit(EXIT_FAILURE);
        }
        *m = value;
        break;

      case 'M':
        if (value < *m) {
          fprintf(stderr, "Error: M must be a positive integer, greater or equal to m.\n");
          exit(EXIT_FAILURE);
        }
        *M = value;
        break;

      default:
        fprintf(stderr, "Usage: %s -N value -g value -m value -M value\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int N, const int G)
{
  printf("\n=================================================\n");
  printf("Single-GPU C+CUDA\n\n");
  printf("Resolution of the %d-Queens instance\n", N);
  printf("  with %d safety check(s) per evaluation\n", G);
  printf("=================================================\n");
}

void print_results(const unsigned long long int exploredTree,
  const unsigned long long int exploredSol, const double timer)
{
  printf("\n=================================================\n");
  printf("Size of the explored tree: %llu\n", exploredTree);
  printf("Number of explored solutions: %llu\n", exploredSol);
  printf("Elapsed time: %.4f [s]\n", timer);
  printf("=================================================\n");
}

void swap(uint8_t* a, uint8_t* b)
{
  uint8_t tmp = *b;
  *b = *a;
  *a = tmp;
}

// Check queen's safety.
uint8_t isSafe(const int G, const uint8_t* board, const uint8_t queen_num, const uint8_t row_pos)
{
  uint8_t isSafe = 1;

  for (int i = 0; i < queen_num; i++) {
    const uint8_t other_row_pos = board[i];

    for (int g = 0; g < G; g++) {
      if (other_row_pos == row_pos - (queen_num - i) ||
          other_row_pos == row_pos + (queen_num - i)) {
        isSafe = 0;
      }
    }
  }

  return isSafe;
}

// Evaluate and generate children nodes on CPU.
void decompose(const int N, const int G, const Node parent,
  unsigned long long int* tree_loc, unsigned long long int* num_sol, SinglePool* pool)
{
  const uint8_t depth = parent.depth;

  if (depth == N) {
    *num_sol += 1;
  }
  else {
    for (int j = depth; j < N; j++) {
      if (isSafe(G, parent.board, depth, parent.board[j])) {
        Node child;
        child.depth = depth + 1;
        memcpy(child.board, parent.board, N * sizeof(uint8_t));
        swap(&child.board[depth], &child.board[j]);
        pushBack(pool, child);
        *tree_loc += 1;
      }
    }
  }
}

// Evaluate a bulk of parent nodes on GPU.
__global__ void evaluate_gpu(const int N, const int G, const Node* parents_d, uint8_t* labels_d, const int size)
{
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadId < size) {
    const int parentId = threadId / N;
    const int k = threadId % N;
    const Node parent = parents_d[parentId];
    const uint8_t depth = parent.depth;
    const uint8_t queen_num = parent.board[k];

    uint8_t isSafe;

    // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
    if (k >= depth) {
      isSafe = 1;
      // const int G_notScheduled = G * (k >= depth);
      for (int i = 0; i < depth; i++) {
        const uint8_t pbi = parent.board[i];

        for (int g = 0; g < G/*G_notScheduled*/; g++) {
          isSafe *= (pbi != queen_num - (depth - i) &&
                     pbi != queen_num + (depth - i));
        }
      }
      labels_d[threadId] = isSafe;
    }
  }
}

// Generate children nodes (evaluated on GPU) on CPU.
void generate_children(const int N, const Node* parents, const int size, const uint8_t* labels,
  unsigned long long int* exploredTree, unsigned long long int* exploredSol, SinglePool* pool)
{
  for (int i = 0; i < size; i++) {
    const Node parent = parents[i];
    const uint8_t depth = parent.depth;

    if (depth == N) {
      *exploredSol += 1;
    }
    else {
      for (int j = depth; j < N; j++) {
        if (labels[j + i * N] == 1) {
          Node child;
          child.depth = depth + 1;
          memcpy(child.board, parent.board, N * sizeof(uint8_t));
          swap(&child.board[depth], &child.board[j]);
          pushBack(pool, child);
          *exploredTree += 1;
        }
      }
    }
  }
}

// Single-GPU N-Queens search.
void nqueens_search(const int N, const int G, const int m, const int M,
  unsigned long long int* exploredTree, unsigned long long int* exploredSol,
  double* elapsedTime)
{
  Node root;
  initRoot(&root, N);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  // Timers
  struct timespec start, end;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  while (pool.size < m) {
    int hasWork = 0;
    Node parent = popFront(&pool, &hasWork);
    if (!hasWork) break;

    decompose(N, G, parent, exploredTree, exploredSol, &pool);
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

  Node* parents = (Node*)malloc(M * sizeof(Node));
  uint8_t* labels = (uint8_t*)malloc(M*N * sizeof(uint8_t));

  Node* parents_d;
  uint8_t* labels_d;
  cudaMalloc(&parents_d, M * sizeof(Node));
  cudaMalloc(&labels_d, M*N * sizeof(uint8_t));

  while (1) {
    int poolSize = popBackBulk(&pool, m, M, parents);

    if (poolSize > 0) {
      // poolSize = MIN(poolSize, M);
      //
      // for (int i = 0; i < poolSize; i++) {
      //   int hasWork = 0;
      //   parents[i] = popBack(&pool, &hasWork);
      //   if (!hasWork) break;
      // }

      const int numLabels = N * poolSize;
      const int nbBlocks = ceil((double)numLabels / BLOCK_SIZE);

      cudaMemcpy(parents_d, parents, poolSize * sizeof(Node), cudaMemcpyHostToDevice);
      evaluate_gpu<<<nbBlocks, BLOCK_SIZE>>>(N, G, parents_d, labels_d, numLabels);
      cudaMemcpy(labels, labels_d, numLabels * sizeof(uint8_t), cudaMemcpyDeviceToHost);

      generate_children(N, parents, poolSize, labels, exploredTree, exploredSol, &pool);
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

    decompose(N, G, parent, exploredTree, exploredSol, &pool);
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  double t3 = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  *elapsedTime = t1 + t2 + t3;

  printf("\nSearch on CPU completed\n");
  printf("Size of the explored tree: %llu\n", *exploredTree);
  printf("Number of explored solutions: %llu\n", *exploredSol);
  printf("Elapsed time: %f [s]\n", t3);

  printf("\nExploration terminated.\n");

  cudaFree(parents_d);
  cudaFree(labels_d);

  free(parents);
  free(labels);

  deleteSinglePool(&pool);
}

int main(int argc, char* argv[])
{
  int N, G, m, M;
  parse_parameters(argc, argv, &N, &G, &m, &M);
  print_settings(N, G);

  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  nqueens_search(N, G, m, M, &exploredTree, &exploredSol, &elapsedTime);

  print_results(exploredTree, exploredSol, elapsedTime);

  return 0;
}
