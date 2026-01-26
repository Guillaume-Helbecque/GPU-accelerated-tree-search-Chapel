/* NOTE: implement a proper way to handke throwing functions */
/* module qap_search_sequential_hhb
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

  var n, N: int(32);

  var initUB: int(32);

  // Evaluate and generate children nodes on CPU.
  proc decompose(const parent: Node_HHB, const ref D, const ref F, const ref priority,
    ref tree_loc: uint, ref num_sol: uint, ref best: int, ref pool: SinglePool(Node_HHB))
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

  // Sequential QAP search.
  proc qap_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
  {
    var timer: stopwatch;

    const getFilenames = inst.split(",");
    const inter = getFilenames[0];
    const dist = getFilenames[1];

    timer.start();

    var priority: [0..<sizeMax] int(32);

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

    var best: int = initUB;

    var root = new Node_HHB(n, N, D, F);

    var pool = new SinglePool(Node_HHB);
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

  proc search_sequential_hhb()
  {
    writeln("Sequential execution mode using HHB");
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
