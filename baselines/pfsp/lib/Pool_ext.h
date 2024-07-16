#ifndef POOL_EXT_H
#define POOL_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "PFSP_node.h"
#include "c_bound_simple.h"
#include <stdbool.h>
#include <stdatomic.h>

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
  _Atomic bool lock;
} SinglePool_ext;

void initSinglePool(SinglePool_ext* pool);

void pushBack(SinglePool_ext* pool, Node node);

void pushBackBulk(SinglePool_ext* pool, Node* nodes, int size);

Node popBack(SinglePool_ext* pool, int* hasWork);

Node popBackFree(SinglePool_ext* pool, int* hasWork);

int popBackBulk(SinglePool_ext* pool, const int m, const int M, Node* parents);

Node* popBackBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize);

Node popFront(SinglePool_ext* pool, int* hasWork);

Node* popFrontBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize);

void deleteSinglePool_ext(SinglePool_ext* pool);

#ifdef __cplusplus
}
#endif

#endif // POOL_EXT_H
