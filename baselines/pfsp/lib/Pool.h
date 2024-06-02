#ifndef POOL_H
#define POOL_H

#include "PFSP_node.h"

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

void initSinglePool(SinglePool* pool);

void pushBack(SinglePool* pool, Node node);

Node popBack(SinglePool* pool, int* hasWork);

void deleteSinglePool(SinglePool* pool);

#endif // POOL_H
