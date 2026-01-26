/* NOTE: implement a proper way to handke throwing functions */
/* module qap_search_sequential_glb
{ */
  /*
    Sequential B&B to solve instances of the QAP in Chapel.
  */
  use IO;
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

  const getFilenames = inst.split(",");
  const inter = getFilenames[0];
  const dist = getFilenames[1];

  var n, N: int(32);

  var priority: [0..<sizeMax] int(32);

  var initUB: int(32);

  var f = open("./benchmarks/qap/instances/data_QubitAlloc/inter/" + inter + ".csv", ioMode.r);
  var channel = f.reader(locking=false);

  channel.read(n);
  var F: [0..<(n**2)] int(32) = noinit;
  channel.read(F);

  channel.close();
  f.close();

  f = open("./benchmarks/qap/instances/data_QubitAlloc/dist/" + dist + ".csv", ioMode.r);
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

  // Sequential QAP search.
  proc qap_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
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
/* } */
