/*
  Sequential B&B to solve Taillard instances of the PFSP in Chapel.
*/

use Time;

use Pool;

use Bound_johnson;
use Bound_simple;
use Taillard;

/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

record Node {
  var depth: int;
  var limit1: int; // right limit
  var limit2: int; // left limit
  var pd: domain(1);
  var prmu: [pd] int; //c_array(c_int, JobsMax);

  // default-initializer
  proc init()
  {}

  // root-initializer
  proc init(jobs)
  {
    this.limit1 = -1;
    this.limit2 = jobs;
    this.pd = {0..#jobs};
    /* init this; */
    this.prmu = 0..#jobs;
  }

  // copy-initializer
  proc init(other: Node)
  {
    this.depth  = other.depth;
    this.limit1 = other.limit1;
    this.limit2 = other.limit2;
    this.pd     = other.pd;
    this.prmu   = other.prmu;
  }

  proc deinit()
  {}
}

/*******************************************************************************
Implementation of the sequential PFSP search.
*******************************************************************************/

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

var lbound1 = new lb1_bound_data(jobs, machines);
taillard_get_processing_times(lbound1.p_times, id);
fill_min_heads_tails(lbound1);

var lbound2 = new lb2_bound_data(lbound1);
fill_machine_pairs(lbound2/*, LB2_FULL*/);
fill_lags(lbound1.p_times, lbound2);
fill_johnson_schedules(lbound1.p_times, lbound2);

const branchingSide = if (br == "fwd") then BEGIN
                      else if (br == "bwd") then END
                      else BEGINEND;

const initUB = if (ub == "opt") then taillard_get_best_ub(id)
               else max(int);

proc check_parameters()
{
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
    var child = new Node(parent);
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth  += 1;
    child.limit1 += 1;

    var lowerbound = lb1_bound(lbound1, child.prmu, child.limit1, jobs);

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

  lb1_children_bounds(lbound1, parent.prmu, parent.limit1, parent.limit2,
    lb_begin, lb_end, nil, nil, beginEnd);

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
        var child = new Node(parent);
        child.depth += 1;

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
    var child = new Node(parent);
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth  += 1;
    child.limit1 += 1;

    var lowerbound = lb2_bound(lbound1, lbound2, child.prmu, child.limit1, jobs, best);

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

// Sequential PFSP search.
proc pfsp_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var best: int = initUB;

  var root = new Node(jobs);

  var pool = new SinglePool(Node);
  pool.pushBack(root);

  var timer: stopwatch;
  timer.start();

  while true {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then break;
    decompose(parent, exploredTree, exploredSol, best, pool);
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

  pfsp_search(optimum, exploredTree, exploredSol, elapsedTime);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  return 0;
}
