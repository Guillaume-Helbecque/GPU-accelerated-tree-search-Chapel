#ifndef NQUEENS_NODE_H
#define NQUEENS_NODE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

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
