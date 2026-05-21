module qap_search_sequential_glb
{
  /*
    Sequential B&B to solve instances of the QAP in Chapel.
  */
  use Time;

  use util;
  use Pool;
  use QAP_node;
  use Util_qap;
  use Problem_qap;

  import main_qap.inst as inst;
  import main_qap.itmax as itmax;
  import main_qap.lb as lb;
  import main_qap.ub as ub;

  config param sizeMax: int(32) = 27;

  /*******************************************************************************
  Implementation of the sequential QAP search.
  *******************************************************************************/

  var benchmark: string;

  var n, N: int(32);

  var initUB: int;

  proc decompose(const parent: Node_GLB, const ref D, const ref F, const ref priority_fac,
    const ref priority_loc, ref tree_loc: uint, ref num_sol: uint, ref best: int,
    ref pool: SinglePool(Node_GLB))
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
      var i = priority_fac[depth];

      for j0 in 0..<N by -1 {
        const j = priority_loc[j0];

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

  // Sequential QAP search.
  proc qap_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
  {
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

    var priority_fac: [0..<sizeMax] int(32);
    var priority_loc: [0..<sizeMax] int(32);

    Prioritization(priority_fac, F, n, ascend = false);
    if (benchmark == "qubitAlloc") then
      Prioritization_loc_connec(D, N);
    else
      Prioritization(priority_loc, D, N);

    if (ub == "heuristic") then initUB = GreedyAllocation(D, F, priority_fac, n, N);
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
      Step 1: Sequential depth-first search.
    */
    timer.start();

    var root = new Node_GLB(n);
    var pool = new SinglePool(Node_GLB);
    pool.pushBack(root);

    while true {
      var hasWork = 0;
      var parent = pool.popBack(hasWork);
      if !hasWork then break;
      decompose(parent, D, F, priority_fac, priority_loc, exploredTree, exploredSol, best, pool);
    }

    timer.stop();
    elapsedTime = timer.elapsed();
    optimum = best;

    writeln("Search on CPU completed");
    writeln("Elapsed time: ", elapsedTime - res0, " [s]");

    writeln("\nExploration terminated.");
  }

  proc search_sequential_glb()
  {
    writeln("Sequential execution mode");

    var optimum: int;
    var exploredTree: uint = 0;
    var exploredSol: uint = 0;

    var elapsedTime: real;

    qap_search(optimum, exploredTree, exploredSol, elapsedTime);

    print_results(optimum, exploredTree, exploredSol, elapsedTime, initUB);

    return 0;
  }
}
