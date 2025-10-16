/*
  Sequential B&B to solve instances of the Qubit Allocation problem in Chapel.
*/
use IO;
use Time;

use util;
use Pool;
use QubitAlloc_node;
use Util_qubitAlloc;
use Problem_qubitAlloc;

config param sizeMax: int(32) = 27;
config param _lb = "hhb";

type Node = if (_lb == "glb") then Node_GLB else Node_HHB;

/*******************************************************************************
Implementation of the sequential Qubit Allocation search.
*******************************************************************************/

config const inter = "10_sqn";
config const dist = "16_melbourne";
config const itmax: int(32) = 10;
config const ub: string = "heuristic"; // heuristic

var filenameInter: string = inter;
var filenameDist: string = dist;
var N: int(32);
var dom: domain(2, idxType = int(32));
var D: [dom] int(32);
var n: int(32);
var F: [dom] int(32);

var priority: [0..<sizeMax] int(32);

var it_max: int(32) = itmax;

var initUB: int(32);

var f = open("./lib/qubitAlloc/instances/inter/" + filenameInter + ".csv", ioMode.r);
var channel = f.reader(locking=false);

channel.read(n);
dom = {0..<n, 0..<n};
channel.read(F);

channel.close();
f.close();

f = open("./lib/qubitAlloc/instances/dist/" + filenameDist + ".csv", ioMode.r);
channel = f.reader(locking=false);

channel.read(N);
assert(n <= N, "More logical qubits than physical ones");
dom = {0..<N, 0..<N};
channel.read(D);

channel.close();
f.close();

Prioritization(priority, F, n, N);

if (ub == "heuristic") then initUB = GreedyAllocation(D, F, priority, n, N);
else {
  try! initUB = ub:int(32);

  // NOTE: If `ub` cannot be cast into `int(32)`, an errow is thrown. For now, we cannot
  // manage it as only catch-less try! statements are allowed in initializers.
  // Ideally, we'd like to do this:

  /* try {
    this.initUB = ub:int(32);
  } catch {
    halt("Error - Unsupported initial upper bound");
  } */
}

proc print_settings(): void
{
  writeln("\n=================================================");
  writeln("Circuit: ", filenameInter);
  writeln("Device: ", filenameDist);
  writeln("Number of logical qubits: ", n);
  writeln("Number of physical qubits: ", N);
  if _lb == "hhb" then
    writeln("Max bounding iterations: ", it_max);
  const heuristic = if (ub == "heuristic") then " (heuristic)" else "";
  writeln("Initial upper bound: ", initUB, heuristic);
  writeln("Lower bound function: ", _lb);
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
  writeln("Optimal allocation: ", optimum, is_better);
  writeln("Elapsed time: ", timer, " [s]");
  writeln("=================================================\n");
}

proc help_message(): void
{
  writeln("\n  Qubit Allocation Problem Parameters:\n");
  writeln("   --inter   str       file containing the coupling distance matrix");
  writeln("   --dist    str       file containing the interaction frequency matrix");
  writeln("   --itmax   int       maximum number of bounding iterations");
  writeln("   --ub      str/int   upper bound initialization ('heuristic' or any integer)\n");
}

// Evaluate and generate children nodes on CPU.
proc decompose_HHB(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  var depth = parent.depth;

  if (parent.depth == n) {
    const eval = ObjectiveFunction(parent.mapping, D, F, n);

    if (eval < best) {
      best = eval;
    }

    num_sol += 1;
  }
  else {
    var i = priority[depth];

    // local index of q_i in the cost matrix
    var k = localLogicalQubitIndex(parent.mapping, i);

    for j in 0..<N by -1 {
      if !parent.available[j] then continue; // skip if not available

      // next available physical qubit
      var l = localPhysicalQubitIndex(parent.available, j);

      // increment lower bound
      var incre = parent.leader[k*(N - depth) + l];
      var lb_new = parent.lower_bound + incre;

      // prune
      if (lb_new > best) {
        continue;
      }

      var child = reduceNode(Node, parent, i, j, k, l, lb_new);

      if (child.depth < n) {
        var lb = bound_HHB(child, best, itmax);
        if (lb <= best) {
          pool.pushBack(child);
          tree_loc += 1;
        }
      }
      else {
        pool.pushBack(child);
        tree_loc += 1;
      }
    }
  }
}

proc decompose_GLB(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  var depth = parent.depth;

  if (parent.depth == n) {
    const eval = ObjectiveFunction(parent.mapping, D, F, n);

    if (eval < best) {
      best = eval;
    }

    num_sol += 1;
  }
  else {
    var i = priority[depth];

    for j in 0..<N by -1 {
      if !parent.available[j] then continue; // skip if not available

      var child = new Node();
      child.mapping = parent.mapping;
      child.depth = parent.depth + 1;
      child.available = parent.available;
      child.mapping[i] = j;
      child.available[j] = false;

      if (child.depth < n) {
        var lb = bound_GLB(child, D, F, n, N);
        if (lb <= best) {
          pool.pushBack(child);
          tree_loc += 1;
        }
      }
      else {
        pool.pushBack(child);
        tree_loc += 1;
      }
    }
  }
}

proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node))
{
  select _lb {
    when "hhb" {
      decompose_HHB(parent, tree_loc, num_sol, best, pool);
    }
    when "glb" {
      decompose_GLB(parent, tree_loc, num_sol, best, pool);
    }
    otherwise {
      halt("DEADCODE");
    }
  }
}

// Sequential Qubit Allocation search.
proc qubitAlloc_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var best: int = initUB;

  var root = if (_lb == "hhb") then new Node(n, N, D, F) else new Node(n);

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

proc main(args: [] string)
{
  // Helper
  for a in args[1..] {
    if (a == "-h" || a == "--help") {
      common_help_message();
      help_message();

      return 1;
    }
  }

  print_settings();

  var optimum: int;
  var exploredTree: uint = 0;
  var exploredSol: uint = 0;

  var elapsedTime: real;

  qubitAlloc_search(optimum, exploredTree, exploredSol, elapsedTime);

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  return 0;
}
