#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <stdbool.h>
#include <stdatomic.h>
#include <stdlib.h>

/******************************************************************************
Auxiliary functions
******************************************************************************/

bool _allIdle(_Atomic bool arr[], int size);

bool allIdle(_Atomic bool arr[], int size, _Atomic bool *flag);

void permute(int* arr, int n);

int findMin(int arr[], int size);

#endif // AUXILIARY_H
