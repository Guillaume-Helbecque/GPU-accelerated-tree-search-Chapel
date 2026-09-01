module qap_search_distributed_glb
{
  /*
    Distributed multi-GPU B&B to solve instances of the QAP in Chapel.
  */
  use Time;
  use Random;
  use PrivateDist;
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

  config const count = 2;

  config const BLOCK_SIZE = 512;

  /*******************************************************************************
  Implementation of the distributed multi-GPU QAP search.
  *******************************************************************************/

  var benchmark: string;

  var n, N: int(32);

  var initUB: int;

  proc decompose(const parent: Node_GLB, const ref D, const ref F, const ref priority_fac,
    const ref priority_loc, ref tree_loc: uint, ref num_sol: uint, ref best: int,
    ref pool: SinglePool_par(Node_GLB))
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
      var i = priority_fac[depth];

      for j0 in 0..<N by -1 {
        const j = priority_loc[j0];

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

  proc prepareChildren(m, M, n, N, const ref D, const ref F, const ref priority_fac,
    const ref priority_loc, ref children, ref pool: SinglePool_par(Node_GLB), ref best,
    ref num_sol)
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
        var i = priority_fac[depth];

        for j0 in 0..<N by -1 {
          const j = priority_loc[j0];

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

  // Distributed multi-GPU QAP search.
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

    var priority_fac: [0..<sizeMax] int(32);
    var priority_loc: [0..<sizeMax] int(32);

    Prioritization(priority_fac, F, n, ascend = false);
    if (benchmark == "qubitAlloc") then
      Prioritization_loc_connec(priority_loc, DD, N);
    else
      Prioritization(priority_loc, DD, N);

    if (ub == "heuristic") then initUB = GreedyAllocation(DD, F, priority_fac, n, N);
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

    while (pool.size < numLocales*D*m) {
      var hasWork = 0;
      var parent = pool.popFrontFree(hasWork);
      if !hasWork then break;

      decompose(parent, DD, F, priority_fac, priority_loc, exploredTree, exploredSol, best, pool);
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

    var eachLocaleExploredTree, eachLocaleExploredSol: [PrivateSpace] uint = noinit;
    var eachLocaleBest: [PrivateSpace] int = noinit;

    var eachLocaleState: [PrivateSpace] atomic bool = BUSY; // one locale per compute node
    var allLocalesIdleFlag: atomic bool = false;

    const poolSize = pool.size;
    const c = poolSize / numLocales;
    const l = poolSize - (numLocales-1)*c;
    const f = pool.front;

    pool.front = 0;
    pool.size = 0;

    var distMultiPool: [PrivateSpace][0..#D] SinglePool_par(Node_GLB);

    coforall (locID, loc) in zip(0..#numLocales, Locales) with (ref pool,
      ref eachLocaleExploredTree, ref eachLocaleExploredSol, ref eachLocaleBest,
      ref eachLocaleState, ref distMultiPool, const ref DD, const ref F,
      const ref priority_fac, const ref priority_loc) do on loc {

      var eachExploredTree, eachExploredSol: [0..#D] uint = noinit;
      var eachBest: [0..#D] int = noinit;
      var eachTaskState: [0..#D] atomic bool = BUSY; // one task per GPU
      var allTasksIdleFlag: atomic bool = false;

      var pool_lloc = new SinglePool_par(Node_GLB);

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

      ref multiPool = distMultiPool[locID];

      /* var eachTime: [1..6, 0..#D] real; */

      coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol,
        ref eachBest, ref eachTaskState, ref multiPool, const ref DD, const ref F,
        const ref priority_fac, const ref priority_loc/*, ref eachTime*/) {

        writeln("Hello from gpu ", gpuID, " of locale ", locID);

        var tree, sol: uint;
        var best_l = best;

        /* var t1, t2, t3, t4, t5, t6: stopwatch; */

        const device = here.gpus[gpuID];

        /* var nSteal, nSSteal: int; */

        /* var tree, sol: uint; */
        ref pool_loc = multiPool[gpuID];
        /* var best_l = best; */
        var taskState, locState: bool = BUSY;

        // each task gets its chunk
        pool_loc.elements[0..#c_l] = pool_lloc.elements[gpuID+f_l.. by D #c_l];
        pool_loc.size += c_l;
        if (gpuID == D-1) {
          pool_loc.elements[c_l..#(l_l-c_l)] = pool_lloc.elements[(D*c_l)+f_l..#(l_l-c_l)];
          pool_loc.size += l_l-c_l;
        }

        writeln("BEFORE on loc ", locID, " gpu ", gpuID, " pool size = ", pool_loc.size);

        var children: [0..#M] Node_GLB = noinit;
        var bounds: [0..#M] int = noinit;

        const DD_loc = DD;
        const F_loc = F;
        const priority_fac_loc = priority_fac;
        const priority_loc_loc = priority_loc;

        on device var children_d: [0..#M] Node_GLB;
        on device var bounds_d: [0..#M] int;

        on device const D_d = DD;
        on device const F_d = F;

        var c = 0;

        writeln("AFTER on loc ", locID, " gpu ", gpuID, " pool size = ", pool_loc.size);

        while true {
          /* t6.start(); */
            ////////////////////////////
          /* local { */
            var poolSize = prepareChildren(m, M, n, N, DD_loc, F_loc, priority_fac_loc, priority_loc_loc, children, pool_loc, best_l, sol);
            /* t6.stop(); */
            /* var poolSize = pool.popBackBulk(m, M, children); */
          /* } */

          if (poolSize > 0) {
            /* local { */
              if (taskState == IDLE) {
                taskState = BUSY;
                eachTaskState[gpuID].write(BUSY);
              }
              if (locState == IDLE) {
                locState = BUSY;
                eachLocaleState[locID].write(BUSY);
              }

              /*
                TODO: Optimize 'numBounds' based on the fact that the maximum number of
                generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
                something like that.
              */
              const numBounds = poolSize;
              /* writeln("break avant kernel et comms");
              break; */
              /* t1.start(); */
              children_d = children; // host-to-device
              /* t1.stop(); */
              /* t2.start(); */
              on device do evaluate_gpu(children_d, numBounds, D_d, F_d, bounds_d); // GPU kernel
              /* t2.stop(); */
              /* t3.start(); */
              bounds = bounds_d; // device-to-host
              /* t3.stop(); */

              /* for threadId in 0..#numBounds {
                bounds[threadId] = bound_GLB(children[threadId], DD_loc, F_loc, n, N);
              } */

              /*
                Each task generates and inserts its children nodes to the pool.
              */
              /* t4.start(); */
              generate_children(children, poolSize, bounds, tree, sol, best_l, pool_loc);
              /* t4.stop(); */
            /* } */
            /* break; */
            c += 1;
            if (c == count) then break;
          }
          else {
            /* break; */
            var localSteal, globalSteal = false;

            local {
              // local work stealing attempts
              const victimTasks = permute(0..#D);

              label WS0 for i in 0..#D {
                const victimTaskID = victimTasks[i];

                if (victimTaskID != gpuID) { // if not me
                  ref victim = multiPool[victimTaskID];
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

                        /* for i in 0..#(size/2) {
                          pool_loc.pushBack(p[i]);
                        } */
                        pool_loc.pushBackBulk(p);

                        localSteal = true;
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
              }

              if (localSteal == false && numLocales != 1) {
                // global work stealing attempts
                const victimLocales = permute(0..#numLocales);

                label WS00 for i in 0..#numLocales {
                  const victimLocaleID = victimLocales[i];

                  if (victimLocaleID != locID) { // if not me
                    ref victimMultiPool = distMultiPool[victimLocaleID];
                    const victimTasks = permute(0..#D);

                    for j in 0..#D {
                      const victimTaskID = victimTasks[j];
                      ref victim = victimMultiPool[victimTaskID];
                      var nn = 0;

                      label WS11 while (nn < 10) {
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

                            globalSteal = true;
                            /* nSSteal += 1; */
                            /* victim.lock.write(false); // reset lock */
                          }

                          victim.lock.write(false); // reset lock
                          break WS00;
                        }

                        nn += 1;
                        currentTask.yieldExecution();
                      }
                    }
                  }
                }
              }

              if (localSteal == false && globalSteal == false) {
                // termination
                if (taskState == BUSY) {
                  taskState = IDLE;
                  eachTaskState[gpuID].write(IDLE);
                }
                if allIdle(eachTaskState, allTasksIdleFlag) {
                  if (locState == BUSY) {
                    locState = IDLE;
                    eachLocaleState[locID].write(IDLE);
                  }
                  if allIdle(eachLocaleState, allLocalesIdleFlag) {
                    break;
                  }
                }
                continue;
              } else {
                continue;
              }
            }
          }
        }

        /* const poolLocSize = pool_loc.size;
        for p in 0..#poolLocSize {
          var hasWork = 0;
          pool.pushBack(pool_loc.popBack(hasWork));
          if !hasWork then break;
        } */

        /* eachTime[1, gpuID] = t1.elapsed();
        eachTime[2, gpuID] = t2.elapsed();
        eachTime[3, gpuID] = t3.elapsed();
        eachTime[4, gpuID] = t4.elapsed();
        eachTime[5, gpuID] = t5.elapsed();
        eachTime[6, gpuID] = t6.elapsed(); */

        eachExploredTree[gpuID] = tree;
        eachExploredSol[gpuID] = sol;
        eachBest[gpuID] = best_l;
      }

      eachLocaleExploredTree[locID] = (+ reduce eachExploredTree);
      eachLocaleExploredSol[locID] = (+ reduce eachExploredSol);
      eachLocaleBest[locID] = (min reduce eachBest);
    }

    exploredTree += (+ reduce eachLocaleExploredTree);
    exploredSol += (+ reduce eachLocaleExploredSol);
    best = (min reduce eachLocaleBest);

    timer.stop();
    const res2 = (timer.elapsed(), exploredTree, exploredSol) - res1;

    writeln("Search on GPU completed");
    writeln("Size of the explored tree: ", res2[1]);
    writeln("Number of explored solutions: ", res2[2]);
    writeln("Elapsed time: ", res2[0], " [s]\n");

    writeln("workload per Locale: ", 100.0*eachLocaleExploredTree/(exploredTree-res1[1]):real, "\n");

    /*
      Step 3: We complete the depth-first search on CPU.
    */
    timer.start();

    /* while true {
      var hasWork = 0;
      var parent = pool.popBackFree(hasWork);
      if !hasWork then break;

      decompose(parent, DD, F, priority, exploredTree, exploredSol, best, pool);
    } */

    timer.stop();
    elapsedTime = timer.elapsed();
    const res3 = (elapsedTime, exploredTree, exploredSol) - res1 - res2;

    writeln("Search on CPU completed");
    writeln("Size of the explored tree: ", res3[1]);
    writeln("Number of explored solutions: ", res3[2]);
    writeln("Elapsed time: ", res3[0], " [s]");

    optimum = best;

    writeln("\nExploration terminated.");

    /* writeln("prepare children = ", (+ reduce eachTime[6, 0..<D])/D, " (", (+ reduce eachTime[6, 0..<D])/D/elapsedTime*100, "%)");
    writeln("H2D              = ", (+ reduce eachTime[1, 0..<D])/D, " (", (+ reduce eachTime[1, 0..<D])/D/elapsedTime*100, "%)");
    writeln("kernel           = ", (+ reduce eachTime[2, 0..<D])/D, " (", (+ reduce eachTime[2, 0..<D])/D/elapsedTime*100, "%)");
    writeln("D2H              = ", (+ reduce eachTime[3, 0..<D])/D, " (", (+ reduce eachTime[3, 0..<D])/D/elapsedTime*100, "%)");
    writeln("gen children     = ", (+ reduce eachTime[4, 0..<D])/D, " (", (+ reduce eachTime[4, 0..<D])/D/elapsedTime*100, "%)");
    writeln("WS               = ", (+ reduce eachTime[5, 0..<D])/D, " (", (+ reduce eachTime[5, 0..<D])/D/elapsedTime*100, "%)"); */
  }

  proc search_distributed_glb()
  {
    writeln("Distributed multi-GPU (", numLocales, " locales x ", D, " GPUs)\n");

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
