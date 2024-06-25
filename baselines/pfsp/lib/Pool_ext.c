#include "Pool_ext.h"
#include <stdio.h>
#include <stdlib.h>

//#define MIN(a, b) ((a) < (b) ? (a) : (b))

void initSinglePool(SinglePool_ext* pool)
{
  pool->elements = (Node*)malloc(CAPACITY * sizeof(Node));
  pool->capacity = CAPACITY;
  pool->front = 0;
  pool->size = 0;
  atomic_store(&(pool->lock),false);
}

void pushBack(SinglePool_ext* pool, Node node) {
  bool expected = false;
  while (true) {
    expected = false;
    // We just leave this while loop after the next if has a true value as argument
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true)) {
      if (pool->front + pool->size >= pool->capacity) {
	pool->capacity *= 2;
	pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
      }

      // Copy node to the end of elements array
      pool->elements[pool->front + pool->size] = node;
      pool->size += 1;
      atomic_store(&(pool->lock),false);
      return;
    } 

    // Yield execution (use appropriate synchronization in actual implementation)
  }
}

void pushBackBulk(SinglePool_ext* pool, Node* nodes, int size) {
  bool expected = false;
  while (true) {
    expected = false;
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true)) {
      if (pool->front + pool->size + size >= pool->capacity) {
	pool->capacity += (size+pool->front+1);
	pool->elements = realloc(pool->elements, pool->capacity * sizeof(Node));
	printf("\nRealloc: PushBackBulk\n");
      }
      // Copy of elements from nodes to the end of elements array of pool
      for(int i = 0; i < size; i++){
	pool->elements[pool->front + pool->size+i] = nodes[i];
      }
      pool->size += size;
      atomic_store(&(pool->lock),false);
      return;
    } 
    // Yield execution (use appropriate synchronization in actual implementation)
  }
}

Node popBack(SinglePool_ext* pool, int* hasWork) {
  bool expected = false;
  while (true) {
    expected = false;   
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, false)) {
      if (pool->size > 0) {
	*hasWork = 1;
	pool->size -= 1;
	// Copy last element to elt
	Node elt;
	elt = pool->elements[pool->front + pool->size];
	atomic_store(&pool->lock,false);
	return elt;
      } else {
	atomic_store(&(pool->lock),false);
	break;
      }
    }
    
    // Yield execution (use appropriate synchronization in actual implementation)
  }
  return (Node){0};
}

Node popBackFree(SinglePool_ext* pool, int* hasWork) {
  if (pool->size > 0){
    *hasWork = 1;
    pool->size -= 1;
    return pool->elements[pool->front + pool->size];
  }

  return (Node){0};
}

int popBackBulk(SinglePool_ext* pool, const int m, const int M, Node* parents){
  bool expected = false;
  while(true) {
    expected = false;    
    if (atomic_compare_exchange_strong(&(pool->lock), &expected, true)) {
      if (pool->size < m) {
	atomic_store(&(pool->lock),false);
	break;
      }
      else{
	int poolSize = MIN(pool->size,M);
	pool->size -= poolSize;
	for(int i = 0; i < poolSize; i++)
	  parents[i] = pool->elements[pool->front + pool->size+i];
	atomic_store(&(pool->lock),false);
	return poolSize;
      }
    }
    // Yield execution (use appropriate synchronization in actual implementation)
  }
  return 0;
}

Node* popBackBulkFree(SinglePool_ext* pool, const int m, const int M, int* poolSize){
  if(pool->size >= 2*m) {
    *poolSize = pool->size/2;
    pool->size -= *poolSize;
    Node* parents = (Node*)malloc(*poolSize * sizeof(Node));
    for(int i = 0; i < *poolSize; i++)
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
}

Node popFront(SinglePool_ext* pool, int* hasWork) {
  if(pool->size > 0) {
    *hasWork = 1;
    Node node;
    node = pool->elements[pool->front];
    pool->front += 1;
    pool->size -= 1;
    return node;
  }

  return (Node){0};
}

void deleteSinglePool_ext(SinglePool_ext* pool) {
  free(pool->elements);
  pool->elements = NULL;
  pool->capacity = 0;
  pool->front = 0;
  pool->size = 0;
  atomic_store(&pool->lock,false);
}
