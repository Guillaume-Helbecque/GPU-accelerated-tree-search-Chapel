#ifndef NQUEENS_NODE_H
#define NQUEENS_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdint.h>

#define BLOCK_SIZE 512

#define MAX_QUEENS 20

typedef struct
{
  uint8_t depth;
  uint8_t board[MAX_QUEENS];
} Node;

void initRoot(Node* root, const int N);

#ifdef __cplusplus
}
#endif

#endif // NQUEENS_NODE_H
