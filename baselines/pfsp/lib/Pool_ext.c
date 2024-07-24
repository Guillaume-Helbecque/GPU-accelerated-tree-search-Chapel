#include "Pool_ext.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void initSinglePool(SinglePool_ext* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->front = 0;
  pool->size = 0;
  atomic_store(&(pool->lock), false);
}

// Parallel-safe insertion to the end of the deque.
void pushBack(SinglePool_ext* pool, Node node) {
  bool expected;
  while (true) {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true)) {
      if (pool->front + pool->size >= pool->capacity) {
        pool->capacity *= 2;
        pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
      }

      pool->elements[pool->front + pool->size] = node;
      pool->size += 1;
      atomic_store(&(pool->lock), false);
      return;
    }
  }
}

// Parallel-safe bulk insertion to the end of the deque.
void pushBackBulk(SinglePool_ext* pool, Node* nodes, int size) {
  bool expected;
  while (true) {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true)) {
      if (pool->front + pool->size + size >= pool->capacity) {
        pool->capacity *= pow(2, ceil(log2((double)(pool->front + pool->size + size) / pool->capacity)));
        pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
      }

      for (int i = 0; i < size; i++)
        pool->elements[pool->front + pool->size + i] = nodes[i];
      pool->size += size;
      atomic_store(&(pool->lock), false);
      return;
    }
  }
}

// Parallel-safe removal from the end of the deque.
Node popBack(SinglePool_ext* pool, int* hasWork) {
  bool expected;
  while (true) {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, false)) {
      if (pool->size > 0) {
        *hasWork = 1;
        pool->size -= 1;
        Node elt = pool->elements[pool->front + pool->size];
        atomic_store(&pool->lock, false);
        return elt;
      } else {
        atomic_store(&(pool->lock), false);
        break;
      }
    }
  }

  return (Node){0};
}

// Removal from the end of the deque. Parallel-safety is not guaranteed.
Node popBackFree(SinglePool_ext* pool, int* hasWork) {
  if (pool->size > 0) {
    *hasWork = 1;
    pool->size -= 1;
    return pool->elements[pool->front + pool->size];
  }

  return (Node){0};
}

// Parallel-safe bulk removal from the end of the deque.
int popBackBulk(SinglePool_ext* pool, const int m, const int M, Node* parents) {
  bool expected;
  while (true) {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true)) {
      if (pool->size < m) {
        atomic_store(&(pool->lock), false);
        break;
      }
      else {
        int poolSize = MIN(pool->size, M);
        pool->size -= poolSize;
        for(int i = 0; i < poolSize; i++)
          parents[i] = pool->elements[pool->front + pool->size + i];
        atomic_store(&(pool->lock), false);
        return poolSize;
      }
    }
  }

  return 0;
}

// Bulk removal from the end of the deque. Parallel-safety is not guaranteed.
Node* popBackBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize) {
  if (pool->size >= 2*m) {
    *poolSize = pool->size/2;
    pool->size -= *poolSize;
    Node* parents = (Node*)malloc(*poolSize * sizeof(Node));
    for (int i = 0; i < *poolSize; i++)
      parents[i] = pool->elements[pool->front + pool->size + i];
    return parents;
  }

  *poolSize = 0;
  return NULL;
}

// Removal from the front of the deque. Parallel-safety is not guaranteed.
Node popFront(SinglePool_ext* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    pool->size--;
    return pool->elements[pool->front++];
  }

  return (Node){0};
}

// Bulk removal from the front of the deque. Parallel-safety is not guaranteed.
Node* popFrontBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize, double perc) {
  if (pool->size >= 2*m) {
    *poolSize = pool->size*perc;
    pool->size -= *poolSize;
    Node* parents = (Node*)malloc(*poolSize * sizeof(Node));
    for (int i = 0; i < *poolSize; i++)
      parents[i] = pool->elements[pool->front + i];
    pool->front += *poolSize;
    return parents;
  }

  *poolSize = 0;
  return NULL;
}

// TODO : In order to implement this function I would have to introduce a new variable
// inside struct Pool_ext (e.g. back) to keep track of the good indexes and pool size
/*Node* popHalfFrontHalfBackBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize){
  if(pool->size >= 2*m) {
    *poolSize = pool->size/2;
    int index = *poolSize/2;
    pool->size -= (*poolSize-index);
    Node* parents = (Node*)malloc(*poolSize * sizeof(Node));
    // Steal a quarter of the work from the front
    for(int i = 0; i < index; i++)
      parents[i] = pool->elements[pool->front + i];
    pool->front += index;
    //Steal a quarter of the work from the back
    for(int i = index; i < *poolSize; i++)
      parents[i] = pool->elements[pool->front + pool->size+i];
    return parents;
  }else{
    *poolSize = 0;
    printf("\nDEADCODE\n");
    return NULL;
  }
  Node* parents = NULL;
  *poolSize = 0;
  return parents;
  }*/

// Free the memory.
void deleteSinglePool_ext(SinglePool_ext* pool) {
  free(pool->elements);
}
