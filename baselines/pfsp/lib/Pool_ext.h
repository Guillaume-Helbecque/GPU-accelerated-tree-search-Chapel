#ifndef POOL_EXT_H
#define POOL_EXT_H

#define MIN(a,b) (((a)<(b))?(a):(b))

#ifdef __cplusplus
extern "C" {
#endif

#include "PFSP_node.h"
#include <stdbool.h>
#include <stdatomic.h>

/*******************************************************************************
Extension of the "Pool" data structure ensuring parallel-safety and supporting
bulk operations.
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

Node popFrontFree(SinglePool_ext* pool, int* hasWork);

Node* popFrontBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize, double perc);

//Node* popHalfFrontHalfBackBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize);

void deleteSinglePool_ext(SinglePool_ext* pool);

#ifdef __cplusplus
}
#endif

#endif // POOL_EXT_H
