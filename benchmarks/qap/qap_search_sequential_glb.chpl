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

  var benchmark: string = "qubitAlloc";

  var n, N: int(32);

  var initUB: int;

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

  // Sequential QAP search.
  proc qap_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
  {
    var timer: stopwatch;

    timer.start();

    var priority: [0..<sizeMax] int(32);

    var domF, domD: domain(1, idxType = int(32));
    var F: [domF] int(32);
    var D: [domD] int(32);

    readInstance(inst, n, N, domF, domD, F, D, benchmark);

    Prioritization(priority, F, n, N);

    if (ub == "heuristic") then initUB = GreedyAllocation(D, F, priority, n, N);
    else {
      try! initUB = ub:int;

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

    while true {
      var hasWork = 0;
      var parent = pool.popBack(hasWork);
      if !hasWork then break;
      decompose(parent, D, F, priority, exploredTree, exploredSol, best, pool);
    }

    timer.stop();
    elapsedTime = timer.elapsed();
    optimum = best;

    writeln("\nExploration terminated.");
  }

  proc search_sequential_glb()
  {
    writeln("Sequential execution mode using GLB");
    print_settings(benchmark, inst, n, N, itmax, lb, ub, initUB);

    var optimum: int;
    var exploredTree: uint = 0;
    var exploredSol: uint = 0;

    var elapsedTime: real;

    qap_search(optimum, exploredTree, exploredSol, elapsedTime);

    print_results(optimum, exploredTree, exploredSol, elapsedTime, initUB);

    return 0;
  }
}
