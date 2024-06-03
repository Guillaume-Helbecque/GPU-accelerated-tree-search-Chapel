#ifndef PFSP_NODE_H
#define PFSP_NODE_H

#include <stdlib.h>
#include <stdint.h>

#define BLOCK_SIZE 512

#define MAX_JOBS 40

#define MAX_MACHINES 40

typedef struct
{
  uint8_t depth;
  int limit1;
  int prmu[MAX_JOBS];
} Node;

void initRoot (Node *root, const int jobs);

#endif // PFSP_NODE_H
