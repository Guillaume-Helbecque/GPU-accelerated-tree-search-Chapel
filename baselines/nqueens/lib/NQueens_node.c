/*******************************************************************************
Implementation of N-Queens Nodes.
*******************************************************************************/

#include "NQueens_node.h"

void initRoot(Node* root, const int N)
{
  root->depth = 0;
  for (uint8_t i = 0; i < N; i++) {
    root->board[i] = i;
  }
}
