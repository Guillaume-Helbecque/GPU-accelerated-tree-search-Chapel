#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>

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

#endif // UTIL_H
