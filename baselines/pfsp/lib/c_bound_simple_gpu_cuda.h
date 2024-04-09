#include <stdlib.h>
#include "c_bound_simple.h"

#ifndef C_BOUND_SIMPLE_GPU_CUDA_H_
#define C_BOUND_SIMPLE_GPU_CUDA_H_

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// #ifdef __cplusplus
// extern "C" {
// #endif

//regroup (constant) bound data
/*struct lb1_bound_data
{
  int *p_times;
  int *min_heads;    // for each machine k, minimum time between t=0 and start of any job
  int *min_tails;    // for each machine k, minimum time between release of any job and end of processing on the last machine
  int nb_jobs;
  int nb_machines;
};

typedef struct lb1_bound_data lb1_bound_data;
*/

//----------------------prepare bound data----------------------
/*
__device__ lb1_bound_data* new_bound_data(int _jobs, int _machines);
__device__ void free_bound_data(lb1_bound_data* lb1_data);
*/
//__device__ void fill_min_heads_tails_gpu(lb1_bound_data* lb1_data);

//----------------------intermediate computations----------------------
__device__ void add_forward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * front);

__device__  void add_backward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * back);

__device__ void schedule_front_gpu(const lb1_bound_data* const lb1_data, const int* const permutation,const int limit1, int* front);

__device__ void schedule_back_gpu(const lb1_bound_data* const lb1_data, const int* const permutation,const int limit2, int* back);

__device__ void sum_unscheduled_gpu(const lb1_bound_data* const lb1_data, const int* const permutation, const int limit1, const int limit2, int* remain);

__device__ int machine_bound_from_parts_gpu(const int* const front, const int* const back, const int* const remain,const int nb_machines);

__device__ int add_front_and_bound_gpu(const lb1_bound_data* const lb1_data, const int job, const int * const front, const int * const back, const int * const remain/*, int* delta_idle*/);

__device__ int add_back_and_bound_gpu(const lb1_bound_data* const lb1_data, const int job, const int * const front, const int * const back, const int * const remain, int* delta_idle);

//------------------evaluate (partial) schedules------------------
__device__ int eval_solution_gpu(const lb1_bound_data* const lb1_data, const int* const permutation);

__device__ int lb1_bound_gpu(const lb1_bound_data* const lb1_data, const int * const permutation, const int limit1, const int limit2);

__device__ void lb1_children_bounds_gpu(const lb1_bound_data* const lb1_data, const int* const permutation, const int limit1, const int limit2, int* const lb_begin/*, int* const lb_end, int* const prio_begin, int* const prio_end, const int direction*/);

// #ifdef __cplusplus
// }
// #endif

#endif
