/*
  Multi-GPU B&B to solve Taillard instances of the PFSP in Chapel.
*/

use Time;
use GpuDiagnostics;

config const BLOCK_SIZE = 512;

use Pool_ext;

use Bound_johnson;
use Bound_simple;
use Taillard;

/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

config param MAX_JOBS = 20;

record Node {
  var depth: int;
  var limit1: int; // left limit
  var prmu: MAX_JOBS*int; //c_array(c_int, JobsMax);

  // default-initializer
  proc init()
  {}

  // root-initializer
  proc init(jobs)
  {
    this.limit1 = -1;
    init this;
    for i in 0..#jobs do this.prmu[i] = i;
  }

  /*
    NOTE: This copy-initializer makes the Node type "non-trivial" for `noinit`.
    Perform manual copy in the code instead.
  */
  // copy-initializer
  /* proc init(other: Node)
  {
    this.depth  = other.depth;
    this.limit1 = other.limit1;
    this.limit2 = other.limit2;
    this.prmu   = other.prmu;
  } */
}

/*******************************************************************************
Implementation of the multi-GPU PFSP search.
*******************************************************************************/

config const m = 25;
config const M = 50000;
config const D = 1;

config const inst: string = "ta14"; // instance
config const lb: string   = "lb1";  // lower bound function
config const br: string   = "fwd";  // branching rules
config const ub: string   = "opt";  // initial upper bound

const id = inst[2..]:int;
const jobs = taillard_get_nb_jobs(id);
const machines = taillard_get_nb_machines(id);

const initUB = if (ub == "opt") then taillard_get_best_ub(id)
               else max(int);

proc check_parameters()
{
  if ((m <= 0) || (M <= 0) || (D <= 0)) {
    halt("Error: m, M, and D must be positive integers.\n");
  }

  const allowedUpperBounds = ["opt", "inf"];
  const allowedLowerBounds = ["lb1", "lb1_d", "lb2"];
  /*
    NOTE: Backward branching is discarded because it increases a lot the implementation
    complexity and does not add much contribution.
  */
  const allowedBranchingRules = ["fwd"];

  if (inst[0..1] != "ta" || id < 1 || id > 120) then
    halt("Error: instance not recognized");

  if (allowedLowerBounds.find(lb) == -1) then
    halt("Error: unsupported lower bound function");

  if (allowedBranchingRules.find(br) == -1) then
    halt("Error: unsupported branching rule");

  if (allowedUpperBounds.find(ub) == -1) then
    halt("Error: unsupported upper bound initialization");
}

proc print_settings(): void
{
  writeln("\n=================================================");
  writeln("Resolution of PFSP Taillard's instance: ", inst, " (m = ", machines, ", n = ", jobs, ")");
  writeln("Initial upper bound: ", ub);
  writeln("Lower bound function: ", lb);
  writeln("Branching rule: ", br);
  writeln("=================================================");
}

proc print_results(const optimum: int, const exploredTree: uint, const exploredSol: uint,
  const timer: real)
{
  writeln("\n=================================================");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  const is_better = if (optimum < initUB) then " (improved)"
                                          else " (not improved)";
  writeln("Optimal makespan: ", optimum, is_better);
  writeln("Elapsed time: ", timer, " [s]");
  writeln("=================================================\n");
}

// Evaluate and generate children nodes on CPU.
proc decompose_lb1(const lb1_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_ext(Node))
{
  for i in parent.limit1+1..(jobs-1) {
    var child = new Node();
    child.depth = parent.depth;
    child.limit1 = parent.limit1 + 1;
    child.prmu = parent.prmu;
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth += 1;

    var lowerbound = lb1_bound(lb1_data, child.prmu, child.limit1, jobs);

    if (child.depth == jobs) { // if child leaf
      num_sol += 1;

      if (lowerbound < best) { // if child feasible
        best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < best) { // if child feasible
        pool.pushBack(child);
        tree_loc += 1;
      }
    }
  }
}

proc decompose_lb1_d(const lb1_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_ext(Node))
{
  var lb_begin: [0..#jobs] int = noinit; // = allocate(c_int, this.jobs);

  lb1_children_bounds(lb1_data, parent.prmu, parent.limit1, jobs, lb_begin);

  for i in parent.limit1+1..(jobs-1) {
    const job = parent.prmu[i];
    const lb = lb_begin[job];

    if (parent.depth + 1 == jobs) { // if child leaf
      num_sol += 1;

      if (lb < best) { // if child feasible
        best = lb;
      }
    } else { // if not leaf
      if (lb < best) { // if child feasible
        var child = new Node();
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1;
        child.prmu = parent.prmu;
        child.limit1 += 1;
        child.prmu[child.limit1] <=> child.prmu[i];

        pool.pushBack(child);
        tree_loc += 1;
      }
    }
  }

  /* deallocate(lb_begin); */
}

proc decompose_lb2(const lb1_data, const lb2_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_ext(Node))
{
  for i in parent.limit1+1..(jobs-1) {
    var child = new Node();
    child.depth = parent.depth;
    child.limit1 = parent.limit1 + 1;
    child.prmu = parent.prmu;
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth += 1;

    var lowerbound = lb2_bound(lb1_data, lb2_data, child.prmu, child.limit1, jobs, best);

    if (child.depth == jobs) { // if child leaf
      num_sol += 1;

      if (lowerbound < best) { // if child feasible
        best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < best) { // if child feasible
        pool.pushBack(child);
        tree_loc += 1;
      }
    }
  }
}

// Evaluate and generate children nodes on CPU.
proc decompose(const lb1_data, const lb2_data, const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool_ext(Node))
{
  select lb {
    when "lb1" {
      decompose_lb1(lb1_data!.lb1_bound, parent, tree_loc, num_sol, best, pool);
    }
    when "lb1_d" {
      decompose_lb1_d(lb1_data!.lb1_bound, parent, tree_loc, num_sol, best, pool);
    }
    when "lb2" {
      decompose_lb2(lb1_data!.lb1_bound, lb2_data!.lb2_bound, parent, tree_loc, num_sol, best, pool);
    }
    otherwise {
      halt("DEADCODE");
    }
  }
}

// Evaluate a bulk of parent nodes on GPU using lb1.
proc evaluate_gpu_lb1(const parents_d: [] Node, const size, const lbound1_d)
{
  var bounds: [0..#size] int = 0;

  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / jobs;
    const k = threadId % jobs;
    var parent = parents_d[parentId];
    const depth = parent.depth;
    var prmu = parent.prmu;

    if (k >= parent.limit1+1) {
      prmu[depth] <=> prmu[k];
      bounds[threadId] = lb1_bound(lbound1_d, prmu, parent.limit1+1, jobs);
      prmu[depth] <=> prmu[k];
    }
  }

  return bounds;
}

/*
  NOTE: This lower bound evaluates all the children of a given parent at the same time.
  Therefore, the GPU loop is on the parent nodes and not on the children ones, in contrast
  to the other lower bounds.
*/
// Evaluate a bulk of parent nodes on GPU using lb1_d.
proc evaluate_gpu_lb1_d(const parents_d: [] Node, const size, const best, const lbound1_d)
{
  var bounds: [0..#size] int = noinit;

  @assertOnGpu
  foreach parentId in 0..#(size/jobs) {
    var parent = parents_d[parentId];
    const depth = parent.depth;
    var prmu = parent.prmu;

    var lb_begin: MAX_JOBS*int; //[0..#size] int = noinit;

    lb1_children_bounds(lbound1_d!.lb1_bound, parent.prmu, parent.limit1, jobs, lb_begin);

    for k in 0..#jobs {
      if (k >= parent.limit1+1) {
        const job = parent.prmu[k];
        bounds[parentId*jobs+k] = lb_begin[job];
      }
    }
  }

  return bounds;
}

// Evaluate a bulk of parent nodes on GPU using lb2.
proc evaluate_gpu_lb2(const parents_d: [] Node, const size, const best, const lbound1_d, const lbound2_d)
{
  var bounds: [0..#size] int = noinit;

  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / jobs;
    const k = threadId % jobs;
    var parent = parents_d[parentId];
    const depth = parent.depth;
    var prmu = parent.prmu;

    if (k >= parent.limit1+1) {
      prmu[depth] <=> prmu[k];
      bounds[threadId] = lb2_bound(lbound1_d, lbound2_d, prmu, parent.limit1+1, jobs, best);
      prmu[depth] <=> prmu[k];
    }
  }

  return bounds;
}

// Evaluate a bulk of parent nodes on GPU.
proc evaluate_gpu(const parents_d: [] Node, const size, const best, const lbound1_d, const lbound2_d)
{
  select lb {
    when "lb1" {
      return evaluate_gpu_lb1(parents_d, size, lbound1_d!.lb1_bound);
    }
    when "lb1_d" {
      return evaluate_gpu_lb1_d(parents_d, size, best, lbound1_d!.lb1_bound);
    }
    when "lb2" {
      return evaluate_gpu_lb2(parents_d, size, best, lbound1_d!.lb1_bound, lbound2_d!.lb2_bound);
    }
    otherwise {
      halt("DEADCODE");
    }
  }
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_children(const ref parents: [] Node, const size: int, const ref bounds: [] int,
  ref exploredTree: uint, ref exploredSol: uint, ref best: int, ref pool: SinglePool_ext(Node))
{
  for i in 0..#size {
    const parent = parents[i];
    const depth = parent.depth;

    for j in parent.limit1+1..(jobs-1) {
      const lowerbound = bounds[j + i * jobs];

      if (depth + 1 == jobs) { // if child leaf
        exploredSol += 1;

        if (lowerbound < best) { // if child feasible
          best = lowerbound;
        }
      } else { // if not leaf
        if (lowerbound < best) { // if child feasible
          var child = new Node();
          child.prmu = parent.prmu;
          child.prmu[parent.depth] <=> child.prmu[j];
          child.depth = parent.depth + 1;
          child.limit1 = parent.limit1 + 1;

          pool.pushBack(child);
          exploredTree += 1;
        }
      }
    }
  }
}

// Multi-GPU PFSP search.
proc pfsp_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var best: int = initUB;

  var root = new Node(jobs);

  var pool = new SinglePool_ext(Node);
  pool.pushBack(root);

  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();

  var lbound1_p = new WrapperLB1(jobs, machines); //lb1_bound_data(jobs, machines);
  taillard_get_processing_times(lbound1_p!.lb1_bound.p_times, id);
  fill_min_heads_tails(lbound1_p!.lb1_bound);

  const lbound2_p = new WrapperLB2(lbound1_p!.lb1_bound);

  while (pool.size < D*m) {
    var hasWork = 0;
    var parent = pool.popFront(hasWork);
    if !hasWork then break;

    decompose(lbound1_p, lbound2_p, parent, exploredTree, exploredSol, best, pool);
  }
  timer.stop();
  const res1 = (timer.elapsed(), exploredTree, exploredSol);
  writeln("\nInitial search on CPU completed");
  writeln("Size of the explored tree: ", res1[1]);
  writeln("Number of explored solutions: ", res1[2]);
  writeln("Elapsed time: ", res1[0], " [s]\n");

  /*
    Step 2: We continue the search on GPU in a depth-first manner, until there
    is not enough work.
  */
  timer.start();
  var eachExploredTree, eachExploredSol: [0..#D] uint = noinit;
  var eachBest: [0..#D] int = noinit;

  const poolSize = pool.size;
  const c = poolSize / D;
  const l = poolSize - (D-1)*c;
  const f = pool.front;
  var lock: atomic bool;

  pool.front = 0;
  pool.size = 0;

  coforall (gpuID, gpu) in zip(0..#D, here.gpus) with (ref pool,
    ref eachExploredTree, ref eachExploredSol, ref eachBest) {

    var tree, sol: uint;
    var pool_loc = new SinglePool_ext(Node);
    var best_l = best;

    // each task gets its chunk
    pool_loc.elements[0..#c] = pool.elements[gpuID+f.. by D #c];
    pool_loc.size += c;
    if (gpuID == D-1) {
      pool_loc.elements[c..#(l-c)] = pool.elements[(D*c)+f..#(l-c)];
      pool_loc.size += l-c;
    }

    var lbound1 = new WrapperLB1(jobs, machines); //lb1_bound_data(jobs, machines);
    taillard_get_processing_times(lbound1!.lb1_bound.p_times, id);
    fill_min_heads_tails(lbound1!.lb1_bound);

    const lbound2 = new WrapperLB2(lbound1!.lb1_bound);

    var lbound1_d: lbound1.type;
    var lbound2_d: lbound2.type;

    on gpu {
      lbound1_d = new WrapperLB1(jobs, machines);
      taillard_get_processing_times(lbound1_d!.lb1_bound.p_times, id);
      fill_min_heads_tails(lbound1_d!.lb1_bound);

      lbound2_d = new WrapperLB2(lbound1_d!.lb1_bound);
    }

    while true {
      /*
        Each task gets its parents nodes from the pool.
      */
      var poolSize = pool_loc.size;
      if (poolSize >= m) {
        poolSize = min(poolSize, M);
        var parents: [0..#poolSize] Node = noinit;
        for i in 0..#poolSize {
          var hasWork = 0;
          parents[i] = pool_loc.popBack(hasWork);
          if !hasWork then break;
        }

        /*
          TODO: Optimize 'numBounds' based on the fact that the maximum number of
          generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
          something like that.
        */
        const numBounds = jobs * poolSize;
        var bounds: [0..#numBounds] int = noinit;

        on gpu {
          const parents_d = parents; // host-to-device
          bounds = evaluate_gpu(parents_d, numBounds, best_l, lbound1_d, lbound2_d);
        }

        /*
          Each task generates and inserts its children nodes to the pool.
        */
        generate_children(parents, poolSize, bounds, tree, sol, best_l, pool_loc);
      }
      else {
        break;
      }
    }

    if lock.compareAndSwap(false, true) {
      const poolLocSize = pool_loc.size;
      for p in 0..#poolLocSize {
        var hasWork = 0;
        pool.pushBack(pool_loc.popBack(hasWork));
        if !hasWork then break;
      }
      lock.write(false);
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
    eachBest[gpuID] = best_l;
  }
  timer.stop();

  exploredTree += (+ reduce eachExploredTree);
  exploredSol += (+ reduce eachExploredSol);
  best = (min reduce eachBest);

  const res2 = (timer.elapsed(), exploredTree, exploredSol) - res1;
  writeln("Search on GPU completed");
  writeln("Size of the explored tree: ", res2[1]);
  writeln("Number of explored solutions: ", res2[2]);
  writeln("Elapsed time: ", res2[0], " [s]\n");

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  timer.start();
  while true {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then break;

    decompose(lbound1_p, lbound2_p, parent, exploredTree, exploredSol, best, pool);
  }
  timer.stop();
  elapsedTime = timer.elapsed();
  const res3 = (elapsedTime, exploredTree, exploredSol) - res1 - res2;
  writeln("Search on CPU completed");
  writeln("Size of the explored tree: ", res3[1]);
  writeln("Number of explored solutions: ", res3[2]);
  writeln("Elapsed time: ", res3[0], " [s]");

  optimum = best;

  writeln("\nExploration terminated.");
}

proc main()
{
  check_parameters();
  print_settings();

  var optimum: int;
  var exploredTree: uint = 0;
  var exploredSol: uint = 0;

  var elapsedTime: real;

  startGpuDiagnostics();

  pfsp_search(optimum, exploredTree, exploredSol, elapsedTime);

  stopGpuDiagnostics();

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  writeln("GPU diagnostics:");
  writeln("   kernel_launch: ", getGpuDiagnostics().kernel_launch);
  writeln("   host_to_device: ", getGpuDiagnostics().host_to_device);
  writeln("   device_to_host: ", getGpuDiagnostics().device_to_host);
  writeln("   device_to_device: ", getGpuDiagnostics().device_to_device);

  return 0;
}
