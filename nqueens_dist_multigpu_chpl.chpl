/*
  Distributed multi-GPU backtracking to solve instances of the N-Queens problem in Chapel.
*/

use Time;

use Pool;
use GpuDiagnostics;

use NQueens_node;

config const BLOCK_SIZE = 512;

/*******************************************************************************
Implementation of the distributed multi-GPU N-Queens search.
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
  writeln("Distributed multi-GPU Chapel (", numLocales, "x", D, " GPUs)\n");
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
proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint, ref pool: SinglePool(Node))
{
  const depth = parent.depth;

  if (depth == N) {
    num_sol += 1;
  }
  for j in depth..(N-1) {
    if isSafe(parent.board, depth, parent.board[j]) {
      var child = new Node();
      child.depth = parent.depth;
      child.board = parent.board;
      child.board[depth] <=> child.board[j];
      child.depth += 1;
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
  ref exploredTree: uint, ref exploredSol: uint, ref pool: SinglePool(Node))
{
  for i in 0..#size  {
    const parent = parents[i];
    const depth = parent.depth;

    if (depth == N) {
      exploredSol += 1;
    }
    for j in depth..(N-1) {
      if (labels[j + i * N] == 1) {
        var child = new Node();
        child.depth = depth + 1;
        child.board = parent.board;
        child.board[depth] <=> child.board[j];
        pool.pushBack(child);
        exploredTree += 1;
      }
    }
  }
}

// Distributed multi-GPU N-Queens search.
proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
{
  var root = new Node(N);

  var pool = new SinglePool(Node);

  pool.pushBack(root);

  var timer: stopwatch;

  /*
    Step 1: We perform a partial breadth-first search on CPU in order to create
    a sufficiently large amount of work for GPU computation.
  */
  timer.start();
  while (pool.size < D*m*numLocales) {
    var hasWork = 0;
    var parent = pool.popFront(hasWork);
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
  var eachLocaleExploredTree, eachLocaleExploredSol: [0..#numLocales] uint = noinit;

  const poolSize = pool.size;
  const c = poolSize / numLocales;
  const l = poolSize - (numLocales-1)*c;
  const f = pool.front;
  var lock: atomic bool;

  pool.front = 0;
  pool.size = 0;

  coforall (locID, loc) in zip(0..#numLocales, Locales) with (ref pool,
    ref eachLocaleExploredTree, ref eachLocaleExploredSol) do on loc {

    var eachExploredTree, eachExploredSol: [0..#D] uint = noinit;

    var pool_lloc = new SinglePool(Node);

    // each locale gets its chunk
    pool_lloc.elements[0..#c] = pool.elements[locID+f.. by numLocales #c];
    pool_lloc.size += c;
    if (locID == numLocales-1) {
      pool_lloc.elements[c..#(l-c)] = pool.elements[(numLocales*c)+f..#(l-c)];
      pool_lloc.size += l-c;
    }

    const poolSize_l = pool_lloc.size;
    const c_l = poolSize_l / D;
    const l_l = poolSize_l - (D-1)*c_l;
    const f_l = pool_lloc.front;

    pool_lloc.front = 0;
    pool_lloc.size = 0;

    coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol) {

      const device = here.gpus[gpuID];

      var tree, sol: uint;
      var pool_loc = new SinglePool(Node);

      // each task gets its chunk
      pool_loc.elements[0..#c_l] = pool_lloc.elements[gpuID+f_l.. by D #c_l];
      pool_loc.size += c_l;
      if (gpuID == D-1) {
        pool_loc.elements[c_l..#(l_l-c_l)] = pool_lloc.elements[(D*c_l)+f_l..#(l_l-c_l)];
        pool_loc.size += l_l-c_l;
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

          const numLabels = N * poolSize;
          var labels: [0..#numLabels] uint(8) = noinit;

          on device {
            const parents_d = parents; // host-to-device
            var labels_d: [0..#numLabels] uint(8) = noinit;
            evaluate_gpu(parents_d, numLabels, labels_d);
            labels = labels_d; // device-to-host
          }

          /*
            Each task generates and inserts its children nodes to the pool.
          */
          generate_children(parents, poolSize, labels, tree, sol, pool_loc);
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
    }

    eachLocaleExploredTree[locID] = (+ reduce eachExploredTree);
    eachLocaleExploredSol[locID] = (+ reduce eachExploredSol);
  }
  timer.stop();

  exploredTree += (+ reduce eachLocaleExploredTree);
  exploredSol += (+ reduce eachLocaleExploredSol);

  writeln("workload per Locale: ", 100.0*eachLocaleExploredTree/(exploredTree-res1[1]):real, "\n");

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

proc main()
{
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
