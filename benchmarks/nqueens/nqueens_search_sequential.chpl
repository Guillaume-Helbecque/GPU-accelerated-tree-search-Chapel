module nqueens_search_sequential
{
  /*
    Sequential backtracking to solve instances of the N-Queens problem in Chapel.
  */

  use Time;

  use util;
  use Pool;
  use NQueens_node;

  /*******************************************************************************
  Implementation of the sequential N-Queens search.
  *******************************************************************************/

  config const N = 14;
  config const g = 1;

  proc check_parameters()
  {
    if ((N <= 0) || (g <= 0)) {
      halt("All parameters must be positive integers.\n");
    }
  }

  proc print_settings()
  {
    writeln("\n=================================================");
    writeln("Sequential Chapel\n");
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
  proc decompose(const parent: Node, ref tree_loc: uint, ref num_sol: uint, ref pool)
  {
    const depth = parent.depth;

    if (depth == N) {
      num_sol += 1;
    }
    else {
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
  }

  // Sequential N-Queens search.
  proc nqueens_search(ref exploredTree: uint, ref exploredSol: uint, ref elapsedTime: real)
  {
    var root = new Node(N);

    var pool = new SinglePool(Node);
    pool.pushBack(root);

    var timer: stopwatch;
    timer.start();

    while true {
      var hasWork = 0;
      var parent = pool.popBack(hasWork);
      if !hasWork then break;
      decompose(parent, exploredTree, exploredSol, pool);
    }

    timer.stop();
    elapsedTime = timer.elapsed();

    writeln("\nExploration terminated.");
  }

  proc search_sequential()
  {
    check_parameters();
    print_settings();

    var exploredTree: uint = 0;
    var exploredSol: uint = 0;

    var elapsedTime: real;

    nqueens_search(exploredTree, exploredSol, elapsedTime);

    print_results(exploredTree, exploredSol, elapsedTime);

    return 0;
  }
}
