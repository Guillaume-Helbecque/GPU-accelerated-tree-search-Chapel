/*
  Multi-GPU backtracking to solve instances of the N-Queens problem in Chapel.
*/

use Time;
use Random;
use GpuDiagnostics;

use util;
use Pool_par;
use NQueens_node;

config const BLOCK_SIZE = 512;

/*******************************************************************************
Implementation of the multi-GPU N-Queens search.
*******************************************************************************/

config const N = 14;
config const g = 1;
config const m = 25;
config const M = 50000;
config const D = 1;

proc check_parameters()
{
  if ((N <= 0) || (g <= 0) || (m <= 0) || (M <= 0) || (D <= 0)) {
    halt("All parameters must be positive integers.\n");
  }
}

proc print_settings()
{
  writeln("\n=================================================");
  writeln("Multi-GPU Chapel (", D, " GPUs)\n");
  writeln("Resolution of the ", N, "-Queens instance");
  writeln("  with ", g, " safety check(s) per evaluation");
  writeln("=================================================");
}

proc print_results(const exploredTree: uint, const exploredSol: uint, const timer: real)
{
  writeln("\n=================================================");
  writeln("Size of the explored tree: ", exploredTree);
  writeln("Number of explored solutions: ", exploredSol);
  writeln("Elapsed time: ", timer, " [s]");
  writeln("=================================================\n");
}

proc help_message(): void
{
  writeln("\n  N-Queens Benchmark Parameters:\n");
  writeln("   --N   int   number of queens");
  writeln("   --g   int   number of safety check(s) per evaluation\n");
}

// Check queen's safety.
proc isSafe(const board, const queen_num, const row_pos): uint(8)
{
  var isSafe: uint(8) = 1;

  for i in 0..#queen_num {
    const other_row_pos = board[i];

    for _g in 0..#g {
      if (other_row_pos == row_pos - (queen_num - i) ||
          other_row_pos == row_pos + (queen_num - i)) {
        isSafe = 0;
      }
    }
  }

  return isSafe;
}

// Evaluate and generate children nodes on CPU.
proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint, ref pool)
{
  const depth = parent.depth;

  if (depth == N) {
    num_sol += 1;
  }
  for j in depth..(N-1) {
    if isSafe(parent.board, depth, parent.board[j]) {
      var child = new Node();
      child.depth = depth + 1;
      child.board = parent.board;
      child.board[depth] <=> child.board[j];
      pool.pushBack(child);
      tree_loc += 1;
    }
  }
}

// Evaluate a bulk of parent nodes on GPU.
proc evaluate_gpu(const parents_d: [] Node, const size, ref labels_d)
{
  @assertOnGpu
  foreach threadId in 0..#size {
    const parentId = threadId / N;
    const k = threadId % N;
    const parent = parents_d[parentId];
    const depth = parent.depth;
    const queen_num = parent.board[k];

    var isSafe: uint(8);
    // If child 'k' is not scheduled, we evaluate its safety 'G' times, otherwise 0.
    if (k >= depth) {
      isSafe = 1;
      /* const G_notScheduled = g * (k >= depth); */
      for i in 0..#depth {
        const pbi = parent.board[i];

        for _g in 0..#g {//G_notScheduled {
          isSafe *= (pbi != queen_num - (depth - i) &&
                     pbi != queen_num + (depth - i));
        }
      }
      labels_d[threadId] = isSafe;
    }
  }
}

// Generate children nodes (evaluated on GPU) on CPU.
proc generate_children(const ref parents: [] Node, const size: int, const ref labels: [] uint(8),
  ref exploredTree: uint, ref exploredSol: uint, ref pool)
{
  pool.acquireLock();

  for i in 0..#size  {
    const parent = parents[i];
    const depth = parent.depth;

    if (depth == N) {
      exploredSol += 1;
    }
    else {
      for j in depth..(N-1) {
        if (labels[j + i * N] == 1) {
          var child = new Node();
          child.depth = depth + 1;
          child.board = parent.board;
          child.board[depth] <=> child.board[j];
          pool.pushBackFree(child);
          exploredTree += 1;
        }
      }
    }
  }

  pool.releaseLock();
}

// Multi-GPU N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var root = new Node(N);

  var pool = new SinglePool_par(Node);

  pool.pushBack(root);

  var allTasksIdleFlag: atomic bool = false;
  var eachTaskState: [0..#D] atomic bool = BUSY; // one task per GPU

  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();
  while (pool.size < D*m) {
    var hasWork = 0;
    var parent = pool.popFrontFree(hasWork);
    if !hasWork then break;

    decompose(parent, exploredTree, exploredSol, pool);
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
  var eachExploredTree, eachExploredSol: [0..#D] uint;

  const poolSize = pool.size;
  const c = poolSize / D;
  const l = poolSize - (D-1)*c;
  const f = pool.front;
  var lock: atomic bool;

  pool.front = 0;
  pool.size = 0;

  var multiPool: [0..#D] SinglePool_par(Node);

  coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol,
    ref eachTaskState, ref multiPool) {

    const device = here.gpus[gpuID];

    var nSteal, nSSteal: int;

    var tree, sol: uint;
    ref pool_loc = multiPool[gpuID];
    var taskState: bool = BUSY;

    // each task gets its chunk
    pool_loc.elements[0..#c] = pool.elements[gpuID+f.. by D #c];
    pool_loc.size += c;
    if (gpuID == D-1) {
      pool_loc.elements[c..#(l-c)] = pool.elements[(D*c)+f..#(l-c)];
      pool_loc.size += l-c;
    }

    var parents: [0..#M] Node = noinit;
    var labels: [0..#(M*N)] uint(8) = noinit;

    on device var parents_d: [0..#M] Node;
    on device var labels_d: [0..#(M*N)] uint(8);

    while true {
      /*
        Each task gets its parents nodes from the pool.
      */
      var poolSize = pool_loc.popBackBulk(m, M, parents);

      if (poolSize > 0) {
        if (taskState == IDLE) {
          taskState = BUSY;
          eachTaskState[gpuID].write(BUSY);
        }

        /* poolSize = min(poolSize, M);

        for i in 0..#poolSize {
          var hasWork = 0;
          parents[i] = pool_loc.popBack(hasWork);
          if !hasWork then break;
        } */

        const numLabels = N * poolSize;

        parents_d = parents; // host-to-device
        on device {
          evaluate_gpu(parents_d, numLabels, labels_d);
        }
        labels = labels_d; // device-to-host

        /*
          Each task generates and inserts its children nodes to the pool.
        */
        generate_children(parents, poolSize, labels, tree, sol, pool_loc);
      }
      else {
        // work stealing
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

                  /* for i in 0..#(size/2) {
                    pool_loc.pushBack(p[i]);
                  } */
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
            break;
          }
          continue;
        } else {
          continue;
        }
      }
    }

    const poolLocSize = pool_loc.size;
    for p in 0..#poolLocSize {
      var hasWork = 0;
      pool.pushBack(pool_loc.popBack(hasWork));
      if !hasWork then break;
    }

    eachExploredTree[gpuID] = tree;
    eachExploredSol[gpuID] = sol;
  }
  timer.stop();

  exploredTree += (+ reduce eachExploredTree);
  exploredSol += (+ reduce eachExploredSol);

  writeln("workload per GPU: ", 100.0*eachExploredTree/(exploredTree-res1[1]):real, "\n");

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
    var parent = pool.popBackFree(hasWork);
    if !hasWork then break;

    decompose(parent, exploredTree, exploredSol, pool);
  }
  timer.stop();
  elapsedTime = timer.elapsed();
  const res3 = (elapsedTime, exploredTree, exploredSol) - res1 - res2;
  writeln("Search on CPU completed");
  writeln("Size of the explored tree: ", res3[1]);
  writeln("Number of explored solutions: ", res3[2]);
  writeln("Elapsed time: ", res3[0], " [s]");

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

  check_parameters();
  print_settings();

  var exploredTree: uint = 0;
  var exploredSol: uint = 0;

  var elapsedTime: real;

  startGpuDiagnostics();

  nqueens_search(exploredTree, exploredSol, elapsedTime);

  stopGpuDiagnostics();

  print_results(exploredTree, exploredSol, elapsedTime);

  writeln("GPU diagnostics:");
  writeln("   kernel_launch: ", getGpuDiagnostics().kernel_launch);
  writeln("   host_to_device: ", getGpuDiagnostics().host_to_device);
  writeln("   device_to_host: ", getGpuDiagnostics().device_to_host);
  writeln("   device_to_device: ", getGpuDiagnostics().device_to_device);

  return 0;
}
