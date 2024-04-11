#ifndef EVALUATE_H
#define EVALUATE_H

#include <cuda.h>
#include <stdlib.h>
#include "parameters.h"
#include "lib/c_bound_simple.h" // For structs definitions
#include "lib/c_bound_johnson.h" // For structs definitions

void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks, const int numBounds, int* best, const lb1_bound_data* const lbound1, const lb2_bound_data* const lbound2, Node* parent, int* bounds);

#endif // EVALUATE_H
