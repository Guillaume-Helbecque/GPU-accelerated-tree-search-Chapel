#include <limits.h>
#include <string.h>
#include "c_bound_simple.h"
#include "c_bound_simple_gpu_cuda.h"

__device__ inline void
add_forward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * front)
{
  front[0] += p_times[job];
  for (int j = 1; j < nb_machines; j++) {
    front[j] = MAX(front[j - 1], front[j]) + p_times[j * nb_jobs + job];
  }
}

__device__ inline void
add_backward_gpu(const int job, const int * const p_times, const int nb_jobs, const int nb_machines, int * back)
{
  int j = nb_machines - 1;

  back[j] += p_times[j * nb_jobs + job];
  for (int j = nb_machines - 2; j >= 0; j--) {
    back[j] = MAX(back[j], back[j + 1]) + p_times[j * nb_jobs + job];
  }
}

__device__ void
schedule_front_gpu(const lb1_bound_data* const lb1_data, const int * const permutation, const int limit1, int * front)
{
  const int N = lb1_data->nb_jobs;
  const int M = lb1_data->nb_machines;
  const int *const p_times = lb1_data->p_times;

  if (limit1 == -1) {
    for (int i = 0; i < M; i++)
      front[i] = lb1_data->min_heads[i];
    return;
  }
  for (int i = 0; i < M; i++){
    front[i] = 0;
  }
  for (int i = 0; i < limit1 + 1; i++) {
    add_forward_gpu(permutation[i], p_times, N, M, front);
  }
}

__device__ void
schedule_back_gpu(const lb1_bound_data* const lb1_data, const int * const permutation, const int limit2, int * back)
{
  const int N = lb1_data->nb_jobs;
  const int M = lb1_data->nb_machines;
  const int *const p_times = lb1_data->p_times;

  if (limit2 == N) {
    for (int i = 0; i < M; i++)
      back[i] = lb1_data->min_tails[i];
    return;
  }

  for (int i = 0; i < M; i++) {
    back[i] = 0;
  }
  for (int k = N - 1; k >= limit2; k--) {
    add_backward_gpu(permutation[k], p_times, N, M, back);
  }
}

__device__ void
sum_unscheduled_gpu(const lb1_bound_data* const lb1_data, const int * const permutation, const int limit1, const int limit2, int * remain)
{
  const int nb_jobs = lb1_data->nb_jobs;
  const int nb_machines = lb1_data->nb_machines;
  const int * const p_times = lb1_data->p_times;

  for (int j = 0; j < nb_machines; j++) {
    remain[j] = 0;
  }
  for (int k = limit1 + 1; k < limit2; k++) {
    const int job = permutation[k];
    for (int j = 0; j < nb_machines; j++) {
      remain[j] += p_times[j * nb_jobs + job];
    }
  }
}

__device__ int
machine_bound_from_parts_gpu(const int * const front, const int * const back, const int * const remain,
			     const int nb_machines)
{
  int tmp0 = front[0] + remain[0];
  int lb = tmp0 + back[0]; // LB on machine 0
  int tmp1;

  for (int i = 1; i < nb_machines; i++) {
    tmp1 = MAX(tmp0, front[i] + remain[i]);
    lb = MAX(lb, tmp1 + back[i]);
    tmp0 = tmp1;
  }

  return lb;
}

__device__ int
add_front_and_bound_gpu(const lb1_bound_data* const lb1_data, const int job, const int * const front, const int * const back, const int * const remain/*, int *delta_idle*/)
{
  int nb_jobs = lb1_data->nb_jobs;
  int nb_machines = lb1_data->nb_machines;
  int* p_times = lb1_data->p_times;

  int lb   = front[0] + remain[0] + back[0];
  int tmp0 = front[0] + p_times[job];
  int tmp1;

  int idle = 0;
  for (int i = 1; i < nb_machines; i++) {
    idle += MAX(0, tmp0 - front[i]);

    tmp1 = MAX(tmp0, front[i]);
    lb   = MAX(lb, tmp1 + remain[i] + back[i]);
    tmp0 = tmp1 + p_times[i * nb_jobs + job];
  }

  //can pass NULL
  // if (delta_idle) {
  //   delta_idle[job] = idle;
  // }

  return lb;
}

// ... same for back
__device__ int
add_back_and_bound_gpu(const lb1_bound_data* const lb1_data, const int job, const int * const front, const int * const back, const int * const remain, int *delta_idle)
{
  int nb_jobs = lb1_data->nb_jobs;
  int nb_machines = lb1_data->nb_machines;
  int* p_times = lb1_data->p_times;

  int last_machine = nb_machines - 1;

  int lb   = front[last_machine] + remain[last_machine] + back[last_machine];
  int tmp0 = back[last_machine] + p_times[last_machine*nb_jobs + job];
  int tmp1;

  int idle = 0;
  for (int i = last_machine-1; i >= 0; i--) {
    idle += MAX(0, tmp0 - back[i]);

    tmp1 = MAX(tmp0, back[i]);
    lb = MAX(lb, tmp1 + remain[i] + front[i]);
    tmp0 = tmp1 + p_times[i*nb_jobs + job];
  }

  //can pass NULL
  if (delta_idle) {
    delta_idle[job] = idle;
  }

  return lb;
}

//----------------------evaluate (partial) schedules---------------------
__device__ int eval_solution_gpu(const lb1_bound_data* lb1_data, const int* const permutation)
{
  const int N = lb1_data->nb_jobs;
  const int M = lb1_data->nb_machines;

  int *tmp = (int*)malloc(N * sizeof(int)); // Dynamically allocate memory for tmp

  // Check if memory allocation succeeded
  if(tmp == NULL) {
    // Handle memory allocation failure
    return -1; // Return an error code indicating failure
  }
  
  for(int i = 0; i < N; i++) {
    tmp[i] = 0;
  }
  for (int i = 0; i < N; i++) {
    add_forward_gpu(permutation[i], lb1_data->p_times, N, M, tmp);
  }
  return tmp[M-1];
}

__device__ int
lb1_bound_gpu(const lb1_bound_data* const lb1_data, const int * const permutation, const int limit1, const int limit2)
{
  int nb_machines = lb1_data->nb_machines;

  int *front = (int*)malloc(nb_machines * sizeof(int)); // Dynamically allocate memory for front
  int *back = (int*)malloc(nb_machines * sizeof(int)); // Dynamically allocate memory for back
  int *remain = (int*)malloc(nb_machines * sizeof(int)); // Dynamically allocate memory for remain

  // Check if memory allocation succeeded
  if(front == NULL || back == NULL || remain == NULL) {
    // Handle memory allocation failure
    return -1; // Return an error code indicating failure
  }

  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);

  sum_unscheduled_gpu(lb1_data, permutation, limit1, limit2, remain);

  return machine_bound_from_parts_gpu(front, back, remain, nb_machines);
}

__device__ void lb1_children_bounds_gpu(const lb1_bound_data *const lb1_data, const int *const permutation, const int limit1, const int limit2, int *const lb_begin/*, int *const lb_end, int *const prio_begin, int *const prio_end, const int direction*/)
{
  int N = lb1_data->nb_jobs;
  int M = lb1_data->nb_machines;

  int *front = (int*)malloc(M * sizeof(int)); // Dynamically allocate memory for front
  int *back = (int*)malloc(M * sizeof(int)); // Dynamically allocate memory for back
  int *remain = (int*)malloc(M * sizeof(int)); // Dynamically allocate memory for remain

  // Check if memory allocation succeeded
  if(front == NULL || back == NULL || remain == NULL) {
    // Handle memory allocation failure
    return; // Return an error code indicating failure
  }

  schedule_front_gpu(lb1_data, permutation, limit1, front);
  schedule_back_gpu(lb1_data, permutation, limit2, back);
  sum_unscheduled_gpu(lb1_data, permutation, limit1, limit2, remain);

  // switch (direction)  {
  //   case -1: //begin
  //   {
  memset(lb_begin, 0, N*sizeof(int));
  // if (prio_begin) memset(prio_begin, 0, N*sizeof(int));

  for (int i = limit1+1; i < limit2; i++) {
    int job = permutation[i];
    lb_begin[job] = add_front_and_bound_gpu(lb1_data, job, front, back, remain/*, prio_begin*/);
  }
  //     break;
  //   }
  //   case 0: //begin-end
  //   {
  //     memset(lb_begin, 0, N*sizeof(int));
  //     memset(lb_end, 0, N*sizeof(int));
  //     if (prio_begin) memset(prio_begin, 0, N*sizeof(int));
  //     if (prio_end) memset(prio_end, 0, N*sizeof(int));
  //
  //     for (int i = limit1+1; i < limit2; i++) {
  //       int job = permutation[i];
  //       lb_begin[job] = add_front_and_bound(data, job, front, back, remain, prio_begin);
  //       lb_end[job] = add_back_and_bound(data, job, front, back, remain, prio_end);
  //     }
  //     break;
  //   }
  //   case 1: //end
  //   {
  //     memset(lb_end, 0, N*sizeof(int));
  //     if (prio_end) memset(prio_end, 0, N*sizeof(int));
  //
  //     for (int i = limit1+1; i < limit2; i++) {
  //       int job = permutation[i];
  //       lb_end[job] = add_back_and_bound(data, job, front, back, remain, prio_end);
  //     }
  //     break;
  //   }
  // }
}

// adds job to partial schedule in front and computes lower bound on optimal cost
// NB1: schedule is no longer needed at this point
// NB2: front, remain and back need to be set before calling this
// NB3: also compute total idle time added to partial schedule (can be used a criterion for job ordering)
// nOps : m*(3 add+2 max)  ---> O(m)


