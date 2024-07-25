#ifndef AUXILIARY_H
#define AUXILIARY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdatomic.h>
#include <stdlib.h>

/******************************************************************************
Auxiliary functions
******************************************************************************/

bool allIdle(_Atomic bool arr[], int size, _Atomic bool *flag);

void permute(int* arr, int n);

int findMin(int arr[], int size);

#ifdef __cplusplus
}
#endif

#endif // AUXILIARY_H
