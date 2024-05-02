/*
  Sequential B&B to solve Taillard instances of the PFSP in Chapel.
*/

use Time;

use Pool;

use PFSP_Node;
use Bound_johnson;
use Bound_simple;
use Taillard;

/*******************************************************************************
Implementation of the sequential PFSP search.
*******************************************************************************/

config const inst: int = 14; // instance
config const lb: int = 1; // lower bound function
config const ub: int = 1; // initial upper bound
/*
  NOTE: Only forward branching is considered because other strategies increase a
  lot the implementation complexity and do not add much contribution.
*/

const jobs = taillard_get_nb_jobs(inst);
const machines = taillard_get_nb_machines(inst);

var lbound1 = new lb1_bound_data(jobs, machines);
taillard_get_processing_times(lbound1.p_times, inst);
fill_min_heads_tails(lbound1);

var lbound2 = new lb2_bound_data(jobs, machines);
fill_machine_pairs(lbound2/*, LB2_FULL*/);
fill_lags(lbound1.p_times, lbound2);
fill_johnson_schedules(lbound1.p_times, lbound2);

const initUB = if (ub == 1) then taillard_get_best_ub(inst) else max(int);

proc check_parameters()
{
  if (inst < 1 || inst > 120) then
    halt("Error: unsupported Taillard's instance");

  if (lb < 0 || lb > 2) then
    halt("Error: unsupported lower bound function");

  if (ub != 0 && ub != 1) then
    halt("Error: unsupported upper bound initialization");
}

proc print_settings(): void
{
  writeln("\n=================================================");
  writeln("Sequential Chapel\n");
  writeln("Resolution of PFSP Taillard's instance: ta", inst, " (m = ", machines, ", n = ", jobs, ")");
  if (ub == 0) then writeln("Initial upper bound: inf");
  else /* if (ub == 1) */ writeln("Initial upper bound: opt");
  if (lb == 0) then writeln("Lower bound function: lb1_d");
  else if (lb == 1) then writeln("Lower bound function: lb1");
  else /* if (lb == 2) */ writeln("Lower bound function: lb2");
  writeln("Branching rule: fwd");
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
    var child = new Node();
    child.depth = parent.depth;
    child.limit1 = parent.limit1 + 1;
    child.prmu = parent.prmu;
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth += 1;

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
  var lb_begin: MAX_JOBS*int(32);

  lb1_children_bounds(lbound1, parent.prmu, parent.limit1, jobs, lb_begin);

  for i in parent.limit1+1..(jobs-1) {
    const job = parent.prmu[i];
    const lowerbound = lb_begin[job];

    if (parent.depth + 1 == jobs) { // if child leaf
      num_sol += 1;

      if (lowerbound < best) { // if child feasible
        best = lowerbound;
      }
    } else { // if not leaf
      if (lowerbound < best) { // if child feasible
        var child = new Node();
        child.depth = parent.depth + 1;
        child.limit1 = parent.limit1 + 1;
        child.prmu = parent.prmu;
        child.prmu[child.limit1] <=> child.prmu[i];

        pool.pushBack(child);
        tree_loc += 1;
      }
    }
  }
}

proc decompose_lb2(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  for i in parent.limit1+1..(jobs-1) {
    var child = new Node();
    child.depth = parent.depth;
    child.limit1 = parent.limit1 + 1;
    child.prmu = parent.prmu;
    child.prmu[child.depth] <=> child.prmu[i];
    child.depth += 1;

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
    when 0 {
      decompose_lb1_d(parent, tree_loc, num_sol, best, pool);
    }
    when 1 {
      decompose_lb1(parent, tree_loc, num_sol, best, pool);
    }
    otherwise { // 2
      decompose_lb2(parent, tree_loc, num_sol, best, pool);
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
