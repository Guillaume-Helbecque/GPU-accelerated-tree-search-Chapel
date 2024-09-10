/*
  Sequential backtracking to solve instances of the N-Queens problem in C.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include "lib/NQueens_node.h"
#include "lib/Pool.h"

/*******************************************************************************
Implementation of the sequential N-Queens search.
*******************************************************************************/

void parse_parameters(int argc, char* argv[], int* N, int* G)
{
  *N = 14;
  *G = 1;

  int opt, value;

  while ((opt = getopt(argc, argv, "N:g:")) != -1) {
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

      default:
        fprintf(stderr, "Usage: %s -N value -g value\n", argv[0]);
        exit(EXIT_FAILURE);
    }
  }
}

void print_settings(const int N, const int G)
{
  printf("\n=================================================\n");
  printf("Sequential C\n\n");
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

inline void swap(uint8_t* a, uint8_t* b)
{
  uint8_t tmp = *b;
  *b = *a;
  *a = tmp;
}

// Check queen's safety.
uint8_t isSafe(const int G, const uint8_t* board, const uint8_t queen_num, const uint8_t row_pos)
{
  uint8_t isSafe = 1;

  for (int g = 0; g < G; g++) {
    for (int i = 0; i < queen_num; i++) {
      const uint8_t other_row_pos = board[i];

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
  for (int j = depth; j < N; j++) {
    if (isSafe(G, parent.board, depth, parent.board[j])) {
      Node child;
      memcpy(child.board, parent.board, N * sizeof(uint8_t));
      swap(&child.board[depth], &child.board[j]);
      child.depth = depth + 1;
      pushBack(pool, child);
      *tree_loc += 1;
    }
  }
}

// Sequential N-Queens search.
void nqueens_search(const int N, const int G,  unsigned long long int* exploredTree,
  unsigned long long int* exploredSol, double* elapsedTime)
{
  Node root;
  initRoot(&root, N);

  SinglePool pool;
  initSinglePool(&pool);

  pushBack(&pool, root);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

  while (1) {
    int hasWork = 0;
    Node parent = popBack(&pool, &hasWork);
    if (!hasWork) break;

    decompose(N, G, parent, exploredTree, exploredSol, &pool);
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  *elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nExploration terminated.");

  deleteSinglePool(&pool);
}

int main(int argc, char* argv[])
{
  int N, G;
  parse_parameters(argc, argv, &N, &G);
  print_settings(N, G);

  unsigned long long int exploredTree = 0;
  unsigned long long int exploredSol = 0;

  double elapsedTime;

  nqueens_search(N, G, &exploredTree, &exploredSol, &elapsedTime);

  print_results(exploredTree, exploredSol, elapsedTime);

  return 0;
}
