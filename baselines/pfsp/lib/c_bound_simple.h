#include <stdlib.h>

#ifndef C_BOUND_SIMPLE_H_
#define C_BOUND_SIMPLE_H_

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifdef __cplusplus
extern "C" {
#endif

//regroup (constant) bound data
struct lb1_bound_data
{
  int *p_times;
  int *min_heads;    // for each machine k, minimum time between t=0 and start of any job
  int *min_tails;    // for each machine k, minimum time between release of any job and end of processing on the last machine
  int nb_jobs;
  int nb_machines;
};

typedef struct lb1_bound_data lb1_bound_data;

//----------------------prepare bound data----------------------
lb1_bound_data* new_bound_data(int _jobs, int _machines);
void free_bound_data(lb1_bound_data* lb1_data);

void fill_min_heads_tails(lb1_bound_data* lb1_data);

//----------------------intermediate computations----------------------
void add_forward(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * front);

void add_backward(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * back);

void schedule_front(const lb1_bound_data* const lb1_data, const int* const permutation,const int limit1, int* front);

void schedule_back(const lb1_bound_data* const lb1_data, const int* const permutation,const int limit2, int* back);

void sum_unscheduled(const lb1_bound_data* const lb1_data, const int* const permutation, const int limit1, const int limit2, int* remain);

int machine_bound_from_parts(const int* const front, const int* const back, const int* const remain,const int nb_machines);

int add_front_and_bound(const lb1_bound_data* const lb1_data, const int job, const int * const front, const int * const back, const int * const remain/*, int* delta_idle*/);

int add_back_and_bound(const lb1_bound_data* const lb1_data, const int job, const int * const front, const int * const back, const int * const remain, int* delta_idle);

//------------------evaluate (partial) schedules------------------
int eval_solution(const lb1_bound_data* const lb1_data, const int* const permutation);

int lb1_bound(const lb1_bound_data* const lb1_data, const int * const permutation, const int limit1, const int limit2);

void lb1_children_bounds(const lb1_bound_data* const lb1_data, const int* const permutation, const int limit1, const int limit2, int* const lb_begin/*, int* const lb_end, int* const prio_begin, int* const prio_end, const int direction*/);

#ifdef __cplusplus
}
#endif

#endif
