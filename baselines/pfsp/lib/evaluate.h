#ifndef EVALUATE_H
#define EVALUATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "PFSP_node.h" // For Nodes definition
#include "c_bound_simple.h" // For structs definitions
#include "c_bound_johnson.h" // For structs definitions

void evaluate_gpu(const int jobs, const int lb, const int size, const int nbBlocks,
	int* best, const lb1_bound_data lbound1, const lb2_bound_data lbound2, Node* parents, int* bounds);

#ifdef __cplusplus
}
#endif

#endif // EVALUATE_H
