#include "Pool.h"
#include <stdlib.h>

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
