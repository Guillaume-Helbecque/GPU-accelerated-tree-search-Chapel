#ifndef UTIL_H
#define UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#define BUSY false
#define IDLE true

inline void swap_uint8(uint8_t* a, uint8_t* b)
{
  uint8_t tmp = *b;
  *b = *a;
  *a = tmp;
}

inline void swap_int(int* a, int* b)
{
  int tmp = *b;
  *b = *a;
  *a = tmp;
}

bool allIdle(_Atomic bool arr[], int size, _Atomic bool *flag);

void permute(int* arr, int n);

int findMin(int arr[], int size);

#ifdef __cplusplus
}
#endif

#endif // UTIL_H
