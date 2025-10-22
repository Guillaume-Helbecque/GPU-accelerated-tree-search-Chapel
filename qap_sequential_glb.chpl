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

/*******************************************************************************
Implementation of the sequential Qubit Allocation search.
*******************************************************************************/

config const inter = "10_sqn";
config const dist = "16_melbourne";
config const ub: string = "heuristic"; // heuristic

var n, N: int(32);

var priority: [0..<sizeMax] int(32);

var initUB: int(32);

var f = open("./lib/qap/instances/inter/" + inter + ".csv", ioMode.r);
var channel = f.reader(locking=false);

channel.read(n);
var F: [0..<(n**2)] int(32) = noinit;
channel.read(F);

channel.close();
f.close();

f = open("./lib/qap/instances/dist/" + dist + ".csv", ioMode.r);
channel = f.reader(locking=false);

channel.read(N);
assert(n <= N, "More logical qubits than physical ones");
var D: [0..<(N**2)] int(32) = noinit;
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
  writeln("Circuit: ", inter);
  writeln("Device: ", dist);
  writeln("Number of logical qubits: ", n);
  writeln("Number of physical qubits: ", N);
  const heuristic = if (ub == "heuristic") then " (heuristic)" else "";
  writeln("Initial upper bound: ", initUB, heuristic);
  writeln("Lower bound function: glb");
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
  writeln("   --ub      str/int   upper bound initialization ('heuristic' or any integer)\n");
}

proc decompose(const parent: Node_GLB, ref tree_loc: uint, ref num_sol: uint,
  ref best: int, ref pool: SinglePool(Node_GLB))
{
  var depth = parent.depth;

  if (parent.depth == n) {
    const eval = ObjectiveFunction(parent.mapping, D, F, n, N);

    if (eval < best) {
      best = eval;
    }

    num_sol += 1;
  }
  else {
    var i = priority[depth];

    for j in 0..<N by -1 {
      if !parent.available[j] then continue; // skip if not available

      var child = new Node_GLB();
      child.mapping = parent.mapping;
      child.depth = parent.depth + 1;
      child.available = parent.available;
      child.mapping[i] = j:int(8);
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

// Sequential Qubit Allocation search.
proc qubitAlloc_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var best: int = initUB;

  var root = new Node_GLB(n);

  var pool = new SinglePool(Node_GLB);
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
