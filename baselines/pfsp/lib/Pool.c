#include "Pool.h"
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void initSinglePool(SinglePool* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->front = 0;
  pool->size = 0;
}

// Insertion to the end of the deque.
void pushBack(SinglePool* pool, Node node)
{
  if (pool->front + pool->size >= pool->capacity) {
    pool->capacity *= 2;
    pool->elements = (Node*)realloc(pool->elements, pool->capacity * sizeof(Node));
  }

  pool->elements[pool->front + pool->size] = node;
  pool->size += 1;
}

// Removal from the end of the deque.
Node popBack(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    pool->size--;
    return pool->elements[pool->front + pool->size];
  }

  return (Node){0};
}

// Bulk removal from the end of the deque.
int popBackBulk(SinglePool* pool, const int m, const int M, Node* parents) {
  if (pool->size >= m) {
    const int poolSize = MIN(pool->size, M);
    pool->size -= poolSize;
    for (int i = 0; i < poolSize; i++) {
      parents[i] = pool->elements[pool->front + pool->size + i];
    }
    return poolSize;
  }

  return 0;
}

// Removal from the front of the deque.
Node popFront(SinglePool* pool, int* hasWork)
{
  if (pool->size > 0) {
    *hasWork = 1;
    pool->size--;
    return pool->elements[pool->front++];
  }

  return (Node){0};
}

// Free the memory.
void deleteSinglePool(SinglePool* pool)
{
  free(pool->elements);
}
