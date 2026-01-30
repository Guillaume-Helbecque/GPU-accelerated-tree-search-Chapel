module qap_search_gpu_hhb
{
  /*
    Single-GPU B&B to solve instances of the QAP in Chapel.
  */
  use Time;
  use GpuDiagnostics;

  use util;
  use Pool;
  use QAP_node;
  use Util_qap;
  use Problem_qap;

  import main_qap.m as m;
  import main_qap.M as M;

  import main_qap.inst as inst;
  import main_qap.itmax as itmax;
  import main_qap.lb as lb;
  import main_qap.ub as ub;

  config param sizeMax: int(32) = 27;

  config const BLOCK_SIZE = 512;

  /*******************************************************************************
  Implementation of the single-GPU QAP search.
  *******************************************************************************/

  var benchmark: string = "qubitAlloc";

  var n, N: int(32);

  var initUB: int;

  // Evaluate and generate children nodes on CPU.
  proc decompose(const parent: Node_HHB, const ref D, const ref F, const ref priority,
    ref tree_loc: uint, ref num_sol: uint, ref best: int, ref pool: SinglePool(Node_HHB))
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

        var child = reduceNode(Node_HHB, parent, i, j, k, l, lb_new);

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

  proc prepareChildren(m, M, n, N, const ref D, const ref F, const ref priority,
    ref children, ref pool, ref best, ref num_sol)
  {
    var size = 0;

    if (pool.size < m) then return 0;

    while (size < M-N) {
      var hasWork = 0;
      var parent = pool.popBack(hasWork);
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

          var child = reduceNode(Node_HHB, parent, i, j, k, l, lb_new);

          children[size] = child;
          size += 1;
        }
      }
    }

    return size;
  }

  // Evaluate a bulk of parent nodes on GPU.
  proc evaluate_gpu(ref children_d: [] Node_HHB, const size, const best, ref bounds_d)
  {
    @assertOnGpu
    foreach threadId in 0..#size {
      bounds_d[threadId] = bound_HHB(children_d[threadId], best, itmax);
    }
  }

  // Generate children nodes (evaluated by GPU) on CPU.
  proc generate_children(const ref children: [] Node_HHB, const size: int, const ref bounds: [] int,
    ref exploredTree: uint, ref exploredSol: uint, ref best: int, ref pool: SinglePool(Node_HHB))
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

  // Single-GPU QAP search.
  proc qap_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
  {
    const device = here.gpus[0];
    var timer: stopwatch;

    // read instance
    var domF, domD: domain(1, idxType = int(32));
    var F: [domF] int(32);
    var D: [domD] int(32);

    readInstance(inst, n, N, domF, domD, F, D, benchmark);

    /*
      Step 0 (preprocessing): Compute a variable prioritization order used by the
      search and compute a heuristic solution for the initial upper bound.
    */
    timer.start();

    var priority: [0..<sizeMax] int(32);
    Prioritization(priority, F, n);

    if (ub == "heuristic") then initUB = GreedyAllocation(D, F, priority, n, N);
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

    var root = new Node_HHB(n, N, D, F);
    var pool = new SinglePool(Node_HHB);
    pool.pushBack(root);

    while (pool.size < m) {
      var hasWork = 0;
      var parent = pool.popFront(hasWork);
      if !hasWork then break;

      decompose(parent, D, F, priority, exploredTree, exploredSol, best, pool);
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

    var children: [0..#M] Node_HHB;// = noinit;
    var bounds: [0..#M] int;// = noinit;

    on device var children_d: [0..#M] Node_HHB;
    on device var bounds_d: [0..#M] int;

    while true {
      var poolSize = prepareChildren(m, M, n, N, D, F, priority, children, pool, best, exploredSol);

      if (poolSize > 0) {
        /*
          TODO: Optimize 'numBounds' based on the fact that the maximum number of
          generated children for a parent is 'parent.limit2 - parent.limit1 + 1' or
          something like that.
        */
        const numBounds = poolSize;

        children_d = children; // host-to-device
        on device do evaluate_gpu(children_d, numBounds, best, bounds_d); // GPU kernel
        // TODO: can we avoid this copy?
        children = children_d;
        bounds = bounds_d; // device-to-host

        /*
          Each task generates and inserts its children nodes to the pool.
        */
        generate_children(children, poolSize, bounds, exploredTree, exploredSol, best, pool);
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
  }

  proc search_gpu_hhb()
  {
    writeln("Single-GPU execution mode using HHB");

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
