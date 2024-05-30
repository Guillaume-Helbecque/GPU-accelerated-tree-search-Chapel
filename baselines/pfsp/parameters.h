#include <stdlib.h>
#include <stdint.h>

#define BLOCK_SIZE 512

#define MAX_JOBS 20

#define MAX_MACHINES 20

typedef struct
{
  uint8_t depth;
  int limit1;
  int prmu[MAX_JOBS];
} Node;


