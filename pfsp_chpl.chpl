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
  var pd: domain(1);
  var prmu: [pd] int; //c_array(c_int, JobsMax);

  // default-initializer
  proc init()
  {}

  // root-initializer
  proc init(jobs)
  {
    this.limit1 = -1;
    this.pd = {0..#jobs};
    /* init this; */
    this.prmu = 0..#jobs;
  }

  // copy-initializer
  proc init(other: Node)
  {
    this.depth  = other.depth;
    this.limit1 = other.limit1;
    this.pd     = other.pd;
    this.prmu   = other.prmu;
  }

  proc deinit()
  {}
}

/*******************************************************************************
Implementation of the sequential PFSP search.
*******************************************************************************/

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

const initUB = if (ub == "opt") then taillard_get_best_ub(id)
               else max(int);

proc check_parameters()
{
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
proc decompose_lb1(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  for i in parent.limit1+1..(jobs-1) {
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
  /* var prio_begin = allocate(c_int, this.jobs);
  var prio_end = allocate(c_int, this.jobs); */

  lb1_children_bounds(lbound1, parent.prmu, parent.limit1, jobs, lb_begin);

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
        var child = new Node(parent);
        child.depth += 1;
        child.limit1 += 1;
        child.prmu[child.limit1] <=> child.prmu[i];

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
  for i in parent.limit1+1..(jobs-1) {
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
