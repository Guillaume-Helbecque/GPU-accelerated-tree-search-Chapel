/*
  Single-GPU B&B to solve Taillard instances of the PFSP in Chapel.
*/

use Time;
use GpuDiagnostics;

config const BLOCK_SIZE = 512;

use Pool;

use Bound_johnson;
use Bound_simple;
use Taillard;

/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

config param MAX_JOBS = 20;

record Node {
  var depth: int;
  var limit1: int; // right limit
  var limit2: int; // left limit
  var prmu: MAX_JOBS*int; //c_array(c_int, JobsMax);

  // default-initializer
  proc init()
  {}

  // root-initializer
  proc init(jobs)
  {
    this.limit1 = -1;
    this.limit2 = jobs;
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

  /* proc deinit()
  {} */
}

/*******************************************************************************
Implementation of the single-GPU PFSP search.
*******************************************************************************/

config const m = 25;
config const M = 50000;

param BEGIN: int    =-1;
param BEGINEND: int = 0;
param END: int      = 1;

config const inst: string = "ta14"; // instance
config const lb: string   = "lb1";  // lower bound function
config const br: string   = "fwd";  // branching rules
config const ub: string   = "opt";  // initial upper bound

const id = inst[2..]:int;
const jobs = taillard_get_nb_jobs(id);
const machines = taillard_get_nb_machines(id);

class WrapperClassLB1 {
  forwarding var lb1_bound: lb1_bound_data;

  proc init(jobs: int, machines: int)
  {
    this.lb1_bound = new lb1_bound_data(jobs, machines);
  }
}

class WrapperClassLB2 {
  forwarding var lb2_bound: lb2_bound_data;

  proc init(const lb1_data: lb1_bound_data)
  {
    this.lb2_bound = new lb2_bound_data(lb1_data);
  }
}

type WrapperLB1 = owned WrapperClassLB1?;
type WrapperLB2 = owned WrapperClassLB2?;

var lbound1 = new WrapperLB1(jobs, machines); //lb1_bound_data(jobs, machines);
taillard_get_processing_times(lbound1!.lb1_bound.p_times, id);
fill_min_heads_tails(lbound1!.lb1_bound);

var lbound2 = new WrapperLB2(lbound1!.lb1_bound);
fill_machine_pairs(lbound2!.lb2_bound/*, LB2_FULL*/);
fill_lags(lbound1!.lb1_bound.p_times, lbound2!.lb2_bound);
fill_johnson_schedules(lbound1!.lb1_bound.p_times, lbound2!.lb2_bound);

const branchingSide = if (br == "fwd") then BEGIN
                      else if (br == "bwd") then END
                      else BEGINEND;

const initUB = if (ub == "opt") then taillard_get_best_ub(id)
               else max(int);

proc check_parameters()
{
  if ((m <= 0) || (M <= 0)) {
    halt("Error: m and M must be positive integers.\n");
  }

  const allowedUpperBounds = ["opt", "inf"];
  const allowedLowerBounds = ["lb1", "lb1_d", "lb2"];
  const allowedBranchingRules = ["fwd", "bwd", "alt", "maxSum", "minMin", "minBranch"];

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

inline proc branchingRule(const lb_begin, const lb_end, const depth, const best)
{
  var branch = br;

  while true {
    select br {
      when "alt" {
        if (depth % 2 == 0) then return BEGIN;
        else return END;
      }
      when "maxSum" {
        var sum1, sum2 = 0;
        for i in 0..#jobs {
          sum1 += lb_begin[i];
          sum2 += lb_end[i];
        }
        if (sum1 >= sum2) then return BEGIN;
        else return END;
      }
      when "minMin" {
        var min0 = max(int);
        for k in 0..#jobs {
          if lb_begin[k] then min0 = min(lb_begin[k], min0);
          if lb_end[k] then min0 = min(lb_end[k], min0);
        }
        var c1, c2 = 0;
        for k in 0..#jobs {
          if (lb_begin[k] == min0) then c1 += 1;
          if (lb_end[k] == min0) then c2 += 1;
        }
        if (c1 < c2) then return BEGIN;
        else if (c1 == c2) then branch = "minBranch";
        else return END;
      }
      when "minBranch" {
        var c, s: int;
        for i in 0..#jobs {
          if (lb_begin[i] >= best) then c += 1;
          if (lb_end[i] >= best) then c -= 1;
          s += (lb_begin[i] - lb_end[i]);
        }
        if (c > 0) then return BEGIN;
        else if (c < 0) then return END;
        else {
          if (s < 0) then return END;
          else return BEGIN;
        }
      }
      otherwise halt("Error - Unsupported branching rule");
    }
  }
  halt("DEADCODE");
}

proc decompose_lb1(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  for i in parent.limit1+1..parent.limit2-1 {
    var child = new Node();
    child.depth = parent.depth;
    child.limit1 = parent.limit1 + 1;
    child.limit2 = parent.limit2; ////////////////////////////////
    child.prmu = parent.prmu;
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth += 1;

    var lowerbound = lb1_bound(lbound1!.lb1_bound, child.prmu, child.limit1, jobs);
    /* writeln(lowerbound); */
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

proc decompose_lb1_d(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  var lb_begin: [0..#jobs] int = noinit; // = allocate(c_int, this.jobs);
  var lb_end: [0..#jobs] int = noinit; // = allocate(c_int, this.jobs);
  /* var prio_begin = allocate(c_int, this.jobs);
  var prio_end = allocate(c_int, this.jobs); */
  var beginEnd = branchingSide;

  lb1_children_bounds(lbound1!.lb1_bound, parent.prmu, parent.limit1, parent.limit2,
    lb_begin, lb_end, /*nil, nil,*/ beginEnd);

  if (branchingSide == BEGINEND) {
    beginEnd = branchingRule(lb_begin, lb_end, parent.depth, best);
  }

  for i in parent.limit1+1..parent.limit2-1 {
    const job = parent.prmu[i];
    const lb = (beginEnd == BEGIN) * lb_begin[job] + (beginEnd == END) * lb_end[job];

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
        child.limit2 = parent.limit2;
        child.prmu = parent.prmu;

        if (beginEnd == BEGIN) {
          child.limit1 += 1;
          child.prmu[child.limit1] <=> child.prmu[i];
        } else if (beginEnd == END) {
          child.limit2 -= 1;
          child.prmu[child.limit2] <=> child.prmu[i];
        }

        pool.pushBack(child);
        tree_loc += 1;
      }
    }
  }

  /* deallocate(lb_begin); deallocate(lb_end); */
  /* deallocate(prio_begin); deallocate(prio_end); */
}

proc decompose_lb2(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  for i in parent.limit1+1..parent.limit2-1 {
    var child = new Node();
    child.depth = parent.depth;
    child.limit1 = parent.limit1 + 1;
    child.limit2 = parent.limit2; ////////////////////////////////
    child.prmu = parent.prmu;
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth += 1;

    var lowerbound = lb2_bound(lbound1!.lb1_bound, lbound2!.lb2_bound, child.prmu, child.limit1, jobs, best);

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
proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  select lb {
    when "lb1" {
      decompose_lb1(parent, tree_loc, num_sol, best, pool);
    }
    when "lb1_d" {
      decompose_lb1_d(parent, tree_loc, num_sol, best, pool);
    }
    when "lb2" {
      decompose_lb2(parent, tree_loc, num_sol, best, pool);
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

    if ((k >= parent.limit1+1) && (k <= parent.limit2-1)) {
      prmu[depth] <=> prmu[k];
      bounds[threadId] = lb1_bound(lbound1_d!.lb1_bound, prmu, parent.limit1+1, jobs);
      prmu[depth] <=> prmu[k];
    }
  }

  /* writeln(bounds); */

  return bounds;
}

// Evaluate a bulk of parent nodes on GPU using lb1_d.
proc evaluate_gpu_lb1_d(const parents_d: [] Node, const size, const best, const lbound1_d)
{
  var bounds: [0..#size] int = noinit;

  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / jobs;
    const k = threadId % jobs;
    var parent = parents_d[parentId];
    const depth = parent.depth;
    var prmu = parent.prmu;

    var lb_begin: MAX_JOBS*int; //[0..#size] int = noinit;
    var lb_end: MAX_JOBS*int; //[0..#size] int = noinit;

    var beginEnd = BEGIN; //branchingSide;

    lb1_children_bounds(lbound1_d!.lb1_bound, parent.prmu, parent.limit1, parent.limit2,
      lb_begin, lb_end, /*nil, nil,*/ beginEnd);

    /* NOTE: branchingRules not GPU eligible */
    /* if (branchingSide == BEGINEND) {
      beginEnd = branchingRule(lb_begin, lb_end, parent.depth, best);
    } */

    if ((k >= parent.limit1+1) && (k <= parent.limit2-1)) {
      const job = parent.prmu[k];
      bounds[threadId] = (beginEnd == BEGIN) * lb_begin[job] + (beginEnd == END) * lb_end[job];
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

    if ((k >= parent.limit1+1) && (k <= parent.limit2-1)) {
      prmu[depth] <=> prmu[k];
      bounds[threadId] = lb2_bound(lbound1_d!.lb1_bound, lbound2_d!.lb2_bound, prmu, parent.limit1+1, jobs, best);
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
      return evaluate_gpu_lb1(parents_d, size, lbound1_d);
    }
    when "lb1_d" {
      return evaluate_gpu_lb1_d(parents_d, size, best, lbound1_d);
    }
    when "lb2" {
      return evaluate_gpu_lb2(parents_d, size, best, lbound1_d, lbound2_d);
    }
    otherwise {
      halt("DEADCODE");
    }
  }
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_children(const ref parents: [] Node, const size: int, const ref bounds: [] int,
  ref exploredTree: uint, ref exploredSol: uint, ref best: int, ref pool: SinglePool(Node))
{
  for i in 0..#size {
    const parent = parents[i];
    const depth = parent.depth;

    for j in parent.limit1+1..parent.limit2-1 {
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
          child.limit2 = parent.limit2; ////////////////////////////////

          pool.pushBack(child);
          exploredTree += 1;
        }
      }
    }
  }
}

// Single-GPU N-Queens search.
proc pfsp_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var best: int = initUB;

  var root = new Node(jobs);

  var pool = new SinglePool(Node);
  pool.pushBack(root);

  var timer: stopwatch;
  timer.start();

  var lbound1_d: lbound1.type;
  var lbound2_d: lbound2.type;

  on here.gpus[0] {
    lbound1_d = new WrapperLB1(jobs, machines);
    taillard_get_processing_times(lbound1_d!.lb1_bound.p_times, id);
    fill_min_heads_tails(lbound1_d!.lb1_bound);

    lbound2_d = new WrapperLB2(lbound1_d!.lb1_bound);
    fill_machine_pairs(lbound2_d!.lb2_bound/*, LB2_FULL*/);
    fill_lags(lbound1_d!.lb1_bound.p_times, lbound2_d!.lb2_bound);
    fill_johnson_schedules(lbound1_d!.lb1_bound.p_times, lbound2_d!.lb2_bound);
  }

  while true {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then {
      writeln("pool size = ", pool.size);
      break;
    }

    decompose(parent, exploredTree, exploredSol, best, pool);

    var poolSize = min(pool.size, M);

    // If 'poolSize' is sufficiently large, we offload the pool on GPU.
    if (poolSize >= m) {
      writeln("ENTER GPU WITH ", poolSize, " PARENT NODES");

      var parents: [0..#poolSize] Node = noinit;
      for i in 0..#poolSize {
        var hasWork = 0;
        parents[i] = pool.popBack(hasWork);
        if !hasWork then break;
      }

      /*
        TODO: Optimize 'numBounds' based on the fact that the maximum number of
        generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
        something like that.
      */
      const numBounds = jobs * poolSize;
      var bounds: [0..#numBounds] int = noinit;

      on here.gpus[0] {
        const parents_d = parents; // host-to-device
        bounds = evaluate_gpu(parents_d, numBounds, best, lbound1_d, lbound2_d);
      }

      /*
        Each task generates and inserts its children nodes to the pool.
      */
      generate_children(parents, poolSize, bounds, exploredTree, exploredSol, best, pool);
    }
  }

  timer.stop();
  elapsedTime = timer.elapsed();

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
