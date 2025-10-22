/*
  Multi-GPU B&B to solve instances of the Qubit Allocation problem in Chapel.
*/
use IO;
use Time;
use Random;
use GpuDiagnostics;

use util;
use Pool_par;
use QubitAlloc_node;
use Util_qubitAlloc;
use Problem_qubitAlloc;

config param sizeMax: int(32) = 27;

config const BLOCK_SIZE = 512;

/*******************************************************************************
Implementation of the multi-GPU Qubit Allocation search.
*******************************************************************************/

config const m = 25;
config const M = 50000;
config const D = 1;

config const inter = "10_sqn";
config const dist = "16_melbourne";
config const itmax: int(32) = 10;
config const ub: string = "heuristic"; // heuristic

var n, N: int(32);
const it_max: int(32) = itmax;

var initUB: int(32);

proc print_settings(): void
{
  writeln("\n=================================================");
  writeln("Multi-GPU Chapel (", D, " GPUs)\n");
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
  ref tree_loc: uint, ref num_sol: uint, ref best: int, ref pool: SinglePool_par(Node_GLB))
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
          pool.pushBackFree(child);
          tree_loc += 1;
        }
      }
      else {
        pool.pushBackFree(child);
        tree_loc += 1;
      }
    }
  }
}

proc prepareChildren(m, M, n, N, const ref D, const ref F, const ref priority,
  ref children, ref pool: SinglePool_par(Node_GLB), ref best, ref num_sol)
{
  var size = 0;

  if (pool.size < m) then return 0;

  pool.acquireLock();

  while (size < M-N) {
    var hasWork = 0;
    var parent = pool.popBackFree(hasWork);
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

  pool.releaseLock();

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
  ref exploredTree: uint, ref exploredSol: uint, ref best: int, ref pool: SinglePool_par(Node_GLB))
{
  pool.acquireLock();

  for i in 0..<size {
    ref child = children[i];

    if (child.depth < n) {
      var lb = bounds[i];
      if (lb <= best) {
        pool.pushBackFree(child);
        exploredTree += 1;
      }
    }
    else {
      pool.pushBackFree(child);
      exploredTree += 1;
    }
  }

  pool.releaseLock();
}

// Multi-GPU Qubit Allocation search.
proc qubitAlloc_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();

  /* var dom: domain(1, idxType = int(32)); */
  var priority: [0..<sizeMax] int(32);

  var ff = open("./lib/qubitAlloc/instances/inter/" + inter + ".csv", ioMode.r);
  var channel = ff.reader(locking=false);

  channel.read(n);
  var F: [0..<(n**2)] int(32) = noinit;
  channel.read(F);

  channel.close();
  ff.close();

  ff = open("./lib/qubitAlloc/instances/dist/" + dist + ".csv", ioMode.r);
  channel = ff.reader(locking=false);

  channel.read(N);
  assert(n <= N, "More logical qubits than physical ones");
  var DD: [0..<(N**2)] int(32) = noinit;
  channel.read(DD);

  channel.close();
  ff.close();

  Prioritization(priority, F, n, N);

  if (ub == "heuristic") then initUB = GreedyAllocation(DD, F, priority, n, N);
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

  var pool = new SinglePool_par(Node_GLB);
  pool.pushBackFree(root);

  while (pool.size < D*m) {
    var hasWork = 0;
    var parent = pool.popFrontFree(hasWork);
    if !hasWork then break;

    decompose(parent, DD, F, priority, exploredTree, exploredSol, best, pool);
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

  var eachExploredTree, eachExploredSol: [0..#D] uint = noinit;
  var eachBest: [0..#D] int = noinit;
  var eachTaskState: [0..#D] atomic bool = BUSY; // one task per GPU
  var allTasksIdleFlag: atomic bool = false;

  var eachTime: [1..6, 0..#D] real;

  const poolSize = pool.size;
  const c = poolSize / D;
  const l = poolSize - (D-1)*c;
  const f = pool.front;

  pool.front = 0;
  pool.size = 0;

  var multiPool: [0..#D] SinglePool_par(Node_GLB);

  coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol,
    ref eachBest, ref eachTaskState, ref multiPool, ref eachTime) {

    var t1, t2, t3, t4, t5, t6: stopwatch;

    const device = here.gpus[gpuID];

    var nSteal, nSSteal: int;

    var tree, sol: uint;
    ref pool_loc = multiPool[gpuID];
    var best_l = best;
    var taskState: bool = BUSY;

    // each task gets its chunk
    pool_loc.elements[0..#c] = pool.elements[gpuID+f.. by D #c];
    pool_loc.size += c;
    if (gpuID == D-1) {
      pool_loc.elements[c..#(l-c)] = pool.elements[(D*c)+f..#(l-c)];
      pool_loc.size += l-c;
    }

    var children: [0..#M] Node_GLB;// = noinit;
    var bounds: [0..#M] int(32);// = noinit;

    on device var children_d: [0..#M] Node_GLB;
    on device var bounds_d: [0..#M] int(32);

    on device const D_d = DD;
    on device const F_d = F;

    while true {
      t6.start();
      var poolSize = prepareChildren(m, M, n, N, DD, F, priority, children, pool_loc, best_l, sol);
      t6.stop();
      /* var poolSize = pool.popBackBulk(m, M, children); */

      if (poolSize > 0) {
        if (taskState == IDLE) {
          taskState = BUSY;
          eachTaskState[gpuID].write(BUSY);
        }

        /*
          TODO: Optimize 'numBounds' based on the fact that the maximum number of
          generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
          something like that.
        */
        const numBounds = poolSize;

        t1.start();
        children_d = children; // host-to-device
        t1.stop();
        t2.start();
        on device do evaluate_gpu(children_d, numBounds, D_d, F_d, bounds_d); // GPU kernel
        t2.stop();
        t3.start();
        bounds = bounds_d; // device-to-host
        t3.stop();

        /*
          Each task generates and inserts its children nodes to the pool.
        */
        t4.start();
        generate_children(children, poolSize, bounds, tree, sol, best_l, pool_loc);
        t4.stop();
      }
      else {
        t5.start();
        // work stealing attempts
        var tries = 0;
        var steal = false;
        const victims = permute(0..#D);

        label WS0 while (tries < D && steal == false) {
          const victimID = victims[tries];

          if (victimID != gpuID) { // if not me
            ref victim = multiPool[victimID];
            nSteal += 1;
            var nn = 0;

            label WS1 while (nn < 10) {
              if victim.lock.compareAndSwap(false, true) { // get the lock
                const size = victim.size;

                if (size >= 2*m) {
                  var (hasWork, p) = victim.popFrontBulkFree(m, M);
                  if (hasWork == 0) {
                    victim.lock.write(false); // reset lock
                    halt("DEADCODE in work stealing");
                  }

                  pool_loc.pushBackBulk(p);

                  steal = true;
                  nSSteal += 1;
                  victim.lock.write(false); // reset lock
                  break WS0;
                }

                victim.lock.write(false); // reset lock
                break WS1;
              }

              nn += 1;
              currentTask.yieldExecution();
            }
          }
          tries += 1;
        }

        if (steal == false) {
          // termination
          if (taskState == BUSY) {
            taskState = IDLE;
            eachTaskState[gpuID].write(IDLE);
          }
          if allIdle(eachTaskState, allTasksIdleFlag) {
            writeln("task ", gpuID, " exits normally");
            t5.stop();
            break;
          }
          t5.stop();
          continue;
        } else {
          t5.stop();
          continue;
        }
        t5.stop();
      }
    }

    const poolLocSize = pool_loc.size;
    for p in 0..#poolLocSize {
      var hasWork = 0;
      pool.pushBack(pool_loc.popBack(hasWork));
      if !hasWork then break;
    }

    eachTime[1, gpuID] = t1.elapsed();
    eachTime[2, gpuID] = t2.elapsed();
    eachTime[3, gpuID] = t3.elapsed();
    eachTime[4, gpuID] = t4.elapsed();
    eachTime[5, gpuID] = t5.elapsed();
    eachTime[6, gpuID] = t6.elapsed();
    /* writeln("on GPU ", gpuID, " t1 = ", t1.elapsed());
    writeln("on GPU ", gpuID, " t2 = ", t2.elapsed());
    writeln("on GPU ", gpuID, " t3 = ", t3.elapsed());
    writeln("on GPU ", gpuID, " t4 = ", t4.elapsed());
    writeln("on GPU ", gpuID, " t5 = ", t5.elapsed());
    writeln("on GPU ", gpuID, " t6 = ", t6.elapsed()); */

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
    eachBest[gpuID] = best_l;
  }

  timer.stop();
  const res2 = (timer.elapsed(), exploredTree, exploredSol) - res1;

  writeln("Search on GPU completed");
  writeln("Size of the explored tree: ", res2[1]);
  writeln("Number of explored solutions: ", res2[2]);
  writeln("Elapsed time: ", res2[0], " [s]\n");

  exploredTree += (+ reduce eachExploredTree);
  exploredSol += (+ reduce eachExploredSol);
  best = (min reduce eachBest);

  writeln("workload per GPU: ", 100.0*eachExploredTree/(exploredTree-res1[1]):real);

  /*
    Step 3: We complete the depth-first search on CPU.
  */
  timer.start();

  while true {
    var hasWork = 0;
    var parent = pool.popBackFree(hasWork);
    if !hasWork then break;

    decompose(parent, DD, F, priority, exploredTree, exploredSol, best, pool);
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

  writeln("prepare children = ", (+ reduce eachTime[6, 0..<D])/D, " (", (+ reduce eachTime[6, 0..<D])/D/elapsedTime*100, "%)");
  writeln("H2D              = ", (+ reduce eachTime[1, 0..<D])/D, " (", (+ reduce eachTime[1, 0..<D])/D/elapsedTime*100, "%)");
  writeln("kernel           = ", (+ reduce eachTime[2, 0..<D])/D, " (", (+ reduce eachTime[2, 0..<D])/D/elapsedTime*100, "%)");
  writeln("D2H              = ", (+ reduce eachTime[3, 0..<D])/D, " (", (+ reduce eachTime[3, 0..<D])/D/elapsedTime*100, "%)");
  writeln("gen children     = ", (+ reduce eachTime[4, 0..<D])/D, " (", (+ reduce eachTime[4, 0..<D])/D/elapsedTime*100, "%)");
  writeln("WS               = ", (+ reduce eachTime[5, 0..<D])/D, " (", (+ reduce eachTime[5, 0..<D])/D/elapsedTime*100, "%)");
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
