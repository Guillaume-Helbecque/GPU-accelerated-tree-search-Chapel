module qap_search_multigpu_glb
{
  /*
    Multi-GPU B&B to solve instances of the QAP in Chapel.
  */
  use Time;
  use Random;
  use GpuDiagnostics;

  use util;
  use Pool_par;
  use QAP_node;
  use Util_qap;
  use Problem_qap;

  import main_qap.m as m;
  import main_qap.M as M;
  import main_qap.D as D;

  import main_qap.inst as inst;
  import main_qap.itmax as itmax;
  import main_qap.lb as lb;
  import main_qap.ub as ub;

  config param sizeMax: int(32) = 27;

  config const BLOCK_SIZE = 512;

  /*******************************************************************************
  Implementation of the multi-GPU QAP search.
  *******************************************************************************/

  var benchmark: string = "qubitAlloc";

  var n, N: int(32);

  var initUB: int;

  proc decompose(const parent: Node_GLB, const ref D, const ref F, const ref priority,
    ref tree_loc: uint, ref num_sol: uint, ref best: int, ref pool: SinglePool_par(Node_GLB))
  {
    const depth = parent.depth;

    if (depth == n) {
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
        child.depth = depth + 1;
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

      const depth = parent.depth;

      if (depth == n) {
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
          child.depth = depth + 1;
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
  proc generate_children(const ref children: [] Node_GLB, const size: int, const ref bounds: [] int,
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

  // Multi-GPU QAP search.
  proc qap_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
  {
    var timer: stopwatch;

    // read instance
    var domF, domD: domain(1, idxType = int(32));
    var F: [domF] int(32);
    var DD: [domD] int(32);

    readInstance(inst, n, N, domF, domD, F, DD, benchmark);

    /*
      Step 0 (preprocessing): Compute a variable prioritization order used by the
      search and compute a heuristic solution for the initial upper bound.
    */
    timer.start();

    var priority: [0..<sizeMax] int(32);
    Prioritization(priority, F, n);

    if (ub == "heuristic") then initUB = GreedyAllocation(DD, F, priority, n, N);
    else {
      try! initUB = ub:int;

      // NOTE: If `ub` cannot be cast into `int`, an errow is thrown. For now, we cannot
      // manage it as only catch-less try! statements are allowed in initializers.
      // Ideally, we'd like to do this:

      /* try {
        this.initUB = ub:int;
      } catch {
        halt("Error - Unsupported initial upper bound");
      } */
    }

    timer.stop();
    const res0 = timer.elapsed();

    print_settings(benchmark, inst, n, N, itmax, lb, ub, initUB);

    writeln("\nPreprocessing completed");
    writeln("Elapsed time: ", res0, " [s]\n");

    var best: int = initUB;

    /*
      Step 1: We perform a partial breadth-first search on CPU in order to create
      a sufficiently large amount of work for GPU computation.
    */
    timer.start();

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
    const res1 = (timer.elapsed() - res0, exploredTree, exploredSol);

    writeln("Initial search on CPU completed");
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

    const poolSize = pool.size;
    const c = poolSize / D;
    const l = poolSize - (D-1)*c;
    const f = pool.front;

    pool.front = 0;
    pool.size = 0;

    var multiPool: [0..#D] SinglePool_par(Node_GLB);

    coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol,
      ref eachBest, ref eachTaskState, ref multiPool) {

      const device = here.gpus[gpuID];

      /* var nSteal, nSSteal: int; */

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
      var bounds: [0..#M] int;// = noinit;

      on device var children_d: [0..#M] Node_GLB;
      on device var bounds_d: [0..#M] int;

      on device const D_d = DD;
      on device const F_d = F;

      while true {
        var poolSize = prepareChildren(m, M, n, N, DD, F, priority, children, pool_loc, best_l, sol);

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

          children_d = children; // host-to-device
          on device do evaluate_gpu(children_d, numBounds, D_d, F_d, bounds_d); // GPU kernel
          bounds = bounds_d; // device-to-host

          /*
            Each task generates and inserts its children nodes to the pool.
          */
          generate_children(children, poolSize, bounds, tree, sol, best_l, pool_loc);
        }
        else {
          // work stealing attempts
          var tries = 0;
          var steal = false;
          const victims = permute(0..#D);

          label WS0 while (tries < D && steal == false) {
            const victimID = victims[tries];

            if (victimID != gpuID) { // if not me
              ref victim = multiPool[victimID];
              /* nSteal += 1; */
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
                    /* nSSteal += 1; */
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
  }

  proc search_multigpu_glb()
  {
    writeln("Multi-GPU execution mode using GLB with ", D, " GPUs");

    var optimum: int;
    var exploredTree: uint = 0;
    var exploredSol: uint = 0;

    var elapsedTime: real;

    startGpuDiagnostics();

    qap_search(optimum, exploredTree, exploredSol, elapsedTime);

    stopGpuDiagnostics();

    print_results(optimum, exploredTree, exploredSol, elapsedTime, initUB);

    writeln("GPU diagnostics:");
    writeln("   kernel_launch: ", getGpuDiagnostics().kernel_launch);
    writeln("   host_to_device: ", getGpuDiagnostics().host_to_device);
    writeln("   device_to_host: ", getGpuDiagnostics().device_to_host);
    writeln("   device_to_device: ", getGpuDiagnostics().device_to_device);

    return 0;
  }
}
