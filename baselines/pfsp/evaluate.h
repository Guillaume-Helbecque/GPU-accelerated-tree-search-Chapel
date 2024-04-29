#ifndef EVALUATE_H
#define EVALUATE_H

#include <cuda.h>
#include <stdlib.h>
#include "parameters.h"
#include "lib/c_bound_simple.h" // For structs definitions
#include "lib/c_bound_johnson.h" // For structs definitions

void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, int* best, const lb1_bound_data lbound1, const lb2_bound_data lbound2, Node* parents, int* bounds/*, int* front, int *back, int* remain*/);

#endif // EVALUATE_H
