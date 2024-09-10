#ifndef POOL_H
#define POOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "NQueens_node.h"

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
  int front;
  int size;
} SinglePool;

void initSinglePool(SinglePool* pool);

void pushBack(SinglePool* pool, Node node);

Node popBack(SinglePool* pool, int* hasWork);

Node popFront(SinglePool* pool, int* hasWork);

void deleteSinglePool(SinglePool* pool);

#ifdef __cplusplus
}
#endif

#endif // POOL_H
