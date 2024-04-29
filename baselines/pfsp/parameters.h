#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <getopt.h>
#include <time.h>

#define BLOCK_SIZE 512

#define MAX_JOBS 20

#define MAX_MACHINES 20

#define MAX_SIZE (MAX_JOBS + 2)

typedef struct
{
  uint8_t depth;
  int limit1;
  int prmu[MAX_JOBS];
} Node;


