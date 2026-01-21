module nqueens_problem
{
  proc print_settings(const N, const g)
  {
    writeln("\n=================================================");
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

  proc nqueens_help_message() {
    writeln("\n  N-Queens Benchmark Parameters:\n");
    writeln("   --N   int   number of queens");
    writeln("   --g   int   number of safety check(s) per evaluation\n");
  }
}
