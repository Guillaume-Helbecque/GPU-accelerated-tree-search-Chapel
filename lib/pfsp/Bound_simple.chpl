module Bound_simple
{
  record lb1_bound_data
  {
    var ptd: domain(1);
    var p_times: [ptd] int;
    var md: domain(1);
    var min_heads: [md] int;    // for each machine k, minimum time between t=0 and start of any job
    var min_tails: [md] int;    // for each machine k, minimum time between release of any job and end of processing on the last machine
    var nb_jobs: int;
    var nb_machines: int;

    proc init(jobs: int, machines: int)
    {
      this.ptd = {0..#(jobs*machines)};
      this.md = {0..#machines};
      this.nb_jobs = jobs;
      this.nb_machines = machines;
    }
  }

  /* NOTE: This wrapper allows one to store persistent data on the GPU memory. */
  class WrapperClassLB1 {
    forwarding var lb1_bound: lb1_bound_data;

    proc init(jobs: int, machines: int)
    {
      this.lb1_bound = new lb1_bound_data(jobs, machines);
    }
  }

  type WrapperLB1 = owned WrapperClassLB1?;

  inline proc add_forward(const job: int, const p_times: [] int, const nb_jobs: int, const nb_machines: int, ref front): void
  {
    front[0] += p_times[job];
    for j in 1..(nb_machines-1) {
      front[j] = max(front[j - 1], front[j]) + p_times[j * nb_jobs + job];
    }
  }

  inline proc add_backward(const job: int, const p_times: [] int, const nb_jobs: int, const nb_machines: int, ref back): void
  {
    var j = nb_machines - 1;

    back[j] += p_times[j * nb_jobs + job];
    for j in 0..(nb_machines - 2) by -1 {
      back[j] = max(back[j], back[j + 1]) + p_times[j * nb_jobs + job];
    }
  }

  proc schedule_front(const data: lb1_bound_data, const permutation, const limit1: int, ref front): void
  {
    const N = data.nb_jobs;
    const M = data.nb_machines;
    const ref p_times = data.p_times;

    if (limit1 == -1) {
      for i in 0..#M do
        front[i] = data.min_heads[i];
      return;
    }

    for i in 0..limit1 {
      add_forward(permutation[i], p_times, N, M, front);
    }
  }

  proc schedule_back(const data: lb1_bound_data, const permutation, const limit2: int, ref back): void
  {
    const N = data.nb_jobs;
    const M = data.nb_machines;
    const ref p_times = data.p_times;

    if (limit2 == N) {
      for i in 0..#M do
        back[i] = data.min_tails[i];
      return;
    }

    for k in limit2..(N-1) by -1 {
      add_backward(permutation[k], p_times, N, M, back);
    }
  }

  proc eval_solution(const data: lb1_bound_data, const permutation): int
  {
    const N = data.nb_jobs;
    const M = data.nb_machines;
    var tmp: [0..#N] int; // initialized to 0

    for i in 0..#N {
      add_forward(permutation[i], data.p_times, N, M, tmp);
    }
    return tmp[M-1];
  }

  proc sum_unscheduled(const data: lb1_bound_data, const permutation, const limit1: int, const limit2: int, ref remain): void
  {
    const N = data.nb_jobs;
    const M = data.nb_machines;
    const ref p_times = data.p_times;

    for k in (limit1+1)..(limit2-1) {
      const job = permutation[k];
      for j in 0..#M {
        remain[j] += p_times[j * N + job];
      }
    }
  }

  proc machine_bound_from_parts(const front, const back, const remain, const nb_machines: int): int
  {
    var tmp0 = front[0] + remain[0];
    var lb = tmp0 + back[0]; // LB on machine 0
    var tmp1: int;

    for i in 1..(nb_machines-1) {
      tmp1 = max(tmp0, front[i] + remain[i]);
      lb = max(lb, tmp1 + back[i]);
      tmp0 = tmp1;
    }

    return lb;
  }

  param NUM_MACHINES = 10;
  param NUM_JOBS = 10;

  proc lb1_bound(const data: lb1_bound_data, const permutation, const limit1: int, const limit2: int): int
  {
    /* const M = data.nb_machines; */

    var front: NUM_MACHINES*int; //[0..#M] int;
    var back: NUM_MACHINES*int; //[0..#M] int;
    var remain: NUM_MACHINES*int; //[0..#M] int;

    schedule_front(data, permutation, limit1, front);
    schedule_back(data, permutation, limit2, back);
    sum_unscheduled(data, permutation, limit1, limit2, remain);

    return machine_bound_from_parts(front, back, remain, NUM_MACHINES);
  }

  proc lb1_children_bounds(const data: lb1_bound_data, const permutation, const limit1: int, const limit2: int, ref lb_begin/*, ref lb_end, prio_begin, prio_end, const direction: int*/): void
  {
    /* const N = data.nb_jobs;
    const M = data.nb_machines; */

    var front: NUM_MACHINES*int; //[0..#M] int;
    var back: NUM_MACHINES*int; //[0..#M] int;
    var remain: NUM_MACHINES*int; //[0..#M] int;

    schedule_front(data, permutation, limit1, front);
    schedule_back(data, permutation, limit2, back);
    sum_unscheduled(data, permutation, limit1, limit2, remain);

    /* select (direction)  {
      when -1 //begin
      { */
        for i in 0..#NUM_JOBS do
          lb_begin[i] = 0;
        /* if (prio_begin != nil) then prio_begin = 0; */

        for i in (limit1+1)..(limit2-1) {
          var job = permutation[i];
          lb_begin[job] = add_front_and_bound(data, job, front, back, remain/*, prio_begin*/);
        }
      /* }
      when 0 //begin-end
      {
        for i in 0..#NUM_JOBS do
          lb_begin[i] = 0;
        for i in 0..#NUM_JOBS do
          lb_end[i] = 0;
        /* if (prio_begin != nil) then prio_begin = 0;
        if (prio_end != nil) then prio_end = 0; */

        for i in (limit1+1)..(limit2-1) {
          var job = permutation[i];
          lb_begin[job] = add_front_and_bound(data, job, front, back, remain/*, prio_begin*/);
          lb_end[job] = add_back_and_bound(data, job, front, back, remain/*, prio_end*/);
        }
      }
      when 1 //end
      {
        for i in 0..#NUM_JOBS do
          lb_end[i] = 0;
        /* if (prio_end != nil) then prio_end = 0; */

        for i in (limit1+1)..(limit2-1) {
          var job = permutation[i];
          lb_end[job] = add_back_and_bound(data, job, front, back, remain/*, prio_end*/);
        }
      }
    } */
  }

  // adds job to partial schedule in front and computes lower bound on optimal cost
  // NB1: schedule is no longer needed at this point
  // NB2: front, remain and back need to be set before calling this
  // NB3: also compute total idle time added to partial schedule (can be used a criterion for job ordering)
  // nOps : m*(3 add+2 max)  ---> O(m)
  proc add_front_and_bound(const data: lb1_bound_data, const job: int, const front, const back, const remain/*, delta_idle*/): int
  {
    const N = data.nb_jobs;
    const M = data.nb_machines;
    const ref p_times = data.p_times;

    var lb   = front[0] + remain[0] + back[0];
    var tmp0 = front[0] + p_times[job];
    var tmp1: int;

    var idle = 0;
    for i in 1..(M-1) {
      idle += max(0, tmp0 - front[i]);

      tmp1 = max(tmp0, front[i]);
      lb   = max(lb, tmp1 + remain[i] + back[i]);
      tmp0 = tmp1 + p_times[i * N + job];
    }

    //can pass NULL
    /* if (delta_idle != nil) {
      delta_idle[job] = idle;
    } */

    return lb;
  }

  // ... same for back
  proc add_back_and_bound(const data: lb1_bound_data, const job: int, const front, const back, const remain/*, delta_idle*/): int
  {
    const N = data.nb_jobs;
    const M = data.nb_machines;
    const ref p_times = data.p_times;

    const last_machine = M - 1;

    var lb   = front[last_machine] + remain[last_machine] + back[last_machine];
    var tmp0 = back[last_machine] + p_times[last_machine*N + job];
    var tmp1: int;

    var idle = 0;
    for i in 0..#last_machine by -1 {
      idle += max(0, tmp0 - back[i]);

      tmp1 = max(tmp0, back[i]);
      lb = max(lb, tmp1 + remain[i] + front[i]);
      tmp0 = tmp1 + p_times[i*N + job];
    }

    //can pass NULL
    /* if (delta_idle != nil) {
      delta_idle[job] = idle;
    } */

    return lb;
  }

  proc fill_min_heads_tails(ref data: lb1_bound_data): void
  {
    const N = data.nb_jobs;
    const M = data.nb_machines;
    const ref p_times = data.p_times;

    var tmp0, tmp1: int;

    // 1/ min start times on each machine
    data.min_heads = max(int);
    data.min_heads[0] = 0; // per definition =0 on first machine

    for i in 0..#N {
      tmp0 = p_times[i];

      for k in 1..(M-1) {
        tmp1 = tmp0 + p_times[k * N + i];
        data.min_heads[k] = min(max(int), tmp0);
        tmp0 = tmp1;
      }
    }

    // 2/ min run-out times on each machine
    data.min_tails = max(int);
    data.min_tails[M - 1] = 0; // per definition =0 on last machine

    for i in 0..#N {
      tmp0 = p_times[(M - 1) * N + i];

      for k in 0..(M-2) by -1 {
        tmp1 = tmp0 + p_times[k * N + i];
        data.min_tails[k] = min(data.min_tails[k], tmp0);
        tmp0 = tmp1;
      }
    }
  }
}
