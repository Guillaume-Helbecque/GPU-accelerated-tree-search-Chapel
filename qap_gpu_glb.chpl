/*
  Single-GPU B&B to solve instances of the Qubit Allocation problem in Chapel.
*/
use IO;
use Time;
use GpuDiagnostics;

use util;
use Pool;
use QubitAlloc_node;
use Util_qubitAlloc;
use Problem_qubitAlloc;

config param sizeMax: int(32) = 27;

config const BLOCK_SIZE = 512;

/*******************************************************************************
Implementation of the single-GPU Qubit Allocation search.
*******************************************************************************/

config const m = 25;
config const M = 50000;

config const inter = "10_sqn";
config const dist = "16_melbourne";
config const ub: string = "heuristic"; // heuristic

var n, N: int(32);

var initUB: int(32);

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

proc decompose(const parent: Node_GLB, const ref D, const ref F, const ref priority,
  ref tree_loc: uint, ref num_sol: uint, ref best: int, ref pool: SinglePool(Node_GLB))
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

proc prepareChildren(m, M, n, N, const ref D, const ref F, const ref priority,
  ref children, ref pool: SinglePool(Node_GLB), ref best, ref num_sol)
{
  var size = 0;

  if (pool.size < m) then return 0;

  while (size < M-N) {
    var hasWork = 0;
    var parent = pool.popBack(hasWork);
    if !hasWork then break;

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

        children[size] = child;
        size += 1;
      }
    }
  }

  return size;
}

// Evaluate a bulk of parent nodes on GPU.
proc evaluate_gpu(ref children_d: [] Node_GLB, const size, const ref D, const ref F, ref bounds_d)
{
  @assertOnGpu
  foreach threadId in 0..#size {
    bounds_d[threadId] = bound_GLB(children_d[threadId], D, F, n, N);
  }
}

// Generate children nodes (evaluated by GPU) on CPU.
proc generate_children(const ref children: [] Node_GLB, const size: int, const ref bounds: [] int(32),
  ref exploredTree: uint, ref exploredSol: uint, ref best: int, ref pool: SinglePool(Node_GLB))
{
  for i in 0..<size {
    ref child = children[i];

    if (child.depth < n) {
      var lb = bounds[i];
      if (lb <= best) {
        pool.pushBack(child);
        exploredTree += 1;
      }
    }
    else {
      pool.pushBack(child);
      exploredTree += 1;
    }
  }
}

// Single-GPU Qubit Allocation search.
proc qubitAlloc_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  const device = here.gpus[0];

  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();

  var priority: [0..<sizeMax] int(32);

  var f = open("./lib/qubitAlloc/instances/inter/" + inter + ".csv", ioMode.r);
  var channel = f.reader(locking=false);

  channel.read(n);
  var F: [0..<(n**2)] int(32) = noinit;
  channel.read(F);

  channel.close();
  f.close();

  f = open("./lib/qubitAlloc/instances/dist/" + dist + ".csv", ioMode.r);
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

  var best: int = initUB;

  var root = new Node_GLB(n);

  var pool = new SinglePool(Node_GLB);
  pool.pushBack(root);

  while (pool.size < m) {
    var hasWork = 0;
    var parent = pool.popFront(hasWork);
    if !hasWork then break;

    decompose(parent, D, F, priority, exploredTree, exploredSol, best, pool);
  }

  timer.stop();
  const res1 = (timer.elapsed(), exploredTree, exploredSol);

  writeln("\nInitial search on CPU completed");
  writeln("Size of the explored tree: ", res1[1]);
  writeln("Number of explored solutions: ", res1[2]);
  writeln("Elapsed time: ", res1[0], " [s]\n");

  /*
    Step 2: We continue the search on GPU in a depth-first manner until there
    is not enough work.
  */
  timer.start();

  var t1, t2, t3, t4, t5: stopwatch;

  var children: [0..#M] Node_GLB;// = noinit;
  var bounds: [0..#M] int(32);// = noinit;

  on device var children_d: [0..#M] Node_GLB;
  on device var bounds_d: [0..#M] int(32);

  on device const D_d = D;
  on device const F_d = F;
  /* on device const priority_d = priority; */

  while true {
    t1.start();
    var poolSize = prepareChildren(m, M, n, N, D, F, priority, children, pool, best, exploredSol);
    t1.stop();
    /* var poolSize = pool.popBackBulk(m, M, children); */

    if (poolSize > 0) {
      /*
        TODO: Optimize 'numBounds' based on the fact that the maximum number of
        generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
        something like that.
      */
      const numBounds = poolSize;

      t2.start();
      children_d = children; // host-to-device
      t2.stop();
      t3.start();
      on device do evaluate_gpu(children_d, numBounds, D_d, F_d, bounds_d); // GPU kernel
      t3.stop();
      t4.start();
      bounds = bounds_d; // device-to-host
      t4.stop();

      /*
        Each task generates and inserts its children nodes to the pool.
      */
      t5.start();
      generate_children(children, poolSize, bounds, exploredTree, exploredSol, best, pool);
      t5.stop();
    }
    else {
      break;
    }
  }

  timer.stop();
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

    decompose(parent, D, F, priority, exploredTree, exploredSol, best, pool);
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

  writeln("prepare children = ", t1.elapsed(), " (", t1.elapsed()/elapsedTime*100, "%)");
  writeln("H2D              = ", t2.elapsed(), " (", t2.elapsed()/elapsedTime*100, "%)");
  writeln("kernel           = ", t3.elapsed(), " (", t3.elapsed()/elapsedTime*100, "%)");
  writeln("D2H              = ", t4.elapsed(), " (", t4.elapsed()/elapsedTime*100, "%)");
  writeln("gen children     = ", t5.elapsed(), " (", t5.elapsed()/elapsedTime*100, "%)");
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

  // TODO: n, N, and ub are still at 0 here
  print_settings();

  var optimum: int;
  var exploredTree: uint = 0;
  var exploredSol: uint = 0;

  var elapsedTime: real;

  startGpuDiagnostics();

  qubitAlloc_search(optimum, exploredTree, exploredSol, elapsedTime);

  stopGpuDiagnostics();

  print_results(optimum, exploredTree, exploredSol, elapsedTime);

  writeln("GPU diagnostics:");
  writeln("   kernel_launch: ", getGpuDiagnostics().kernel_launch);
  writeln("   host_to_device: ", getGpuDiagnostics().host_to_device);
  writeln("   device_to_host: ", getGpuDiagnostics().device_to_host);
  writeln("   device_to_device: ", getGpuDiagnostics().device_to_device);

  return 0;
}
