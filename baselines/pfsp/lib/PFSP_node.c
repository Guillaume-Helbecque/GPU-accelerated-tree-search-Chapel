/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

#include "PFSP_node.h"

void initRoot(Node* root, const int jobs)
{
  root->depth = 0;
  root->limit1 = -1;
  for (int i = 0; i < jobs; i++) {
    root->prmu[i] = i;
  }
}
