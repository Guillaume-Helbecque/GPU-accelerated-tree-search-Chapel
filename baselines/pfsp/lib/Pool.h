#ifndef POOL_H
#define POOL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "PFSP_node.h"

/*******************************************************************************
Dynamic-sized single-pool data structure. Its initial capacity is 1024, and we
reallocate a new container with double the capacity when it is full. The pool
supports operations from both ends, allowing breadth-first and depth-first search
strategies.
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

int popBackBulk(SinglePool* pool, const int m, const int M, Node* parents);

Node popFront(SinglePool* pool, int* hasWork);

void deleteSinglePool(SinglePool* pool);

#ifdef __cplusplus
}
#endif

#endif // POOL_H
