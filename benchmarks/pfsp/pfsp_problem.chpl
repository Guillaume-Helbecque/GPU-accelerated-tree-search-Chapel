module pfsp_problem
{
  proc print_settings(const jobs, const machines, const inst, const ub, const lb): void
  {
    writeln("\n=================================================");
    writeln("Resolution of PFSP Taillard's instance: ta", inst, " (m = ", machines, ", n = ", jobs, ")");
    if (ub == 0) then writeln("Initial upper bound: inf");
    else /* if (ub == 1) */ writeln("Initial upper bound: opt");
    writeln("Lower bound function: ", lb);
    writeln("Branching rule: fwd");
    writeln("=================================================");
  }

  proc print_results(const optimum: int, const exploredTree: uint, const exploredSol: uint,
    const timer: real, const initUB)
  {
    writeln("\n=================================================");
    writeln("Size of the explored tree: ", exploredTree);
    writeln("Number of explored solutions: ", exploredSol);
    const is_better = if (optimum < initUB) then " (improved)"
                                            else " (not improved)";
    writeln("Optimal makespan: ", optimum, is_better);
    writeln("Elapsed time: ", timer, " [s]");
    writeln("=================================================\n");
  }

  proc pfsp_help_message(): void
  {
    writeln("\n  PFSP Benchmark Parameters:\n");
    writeln("   --inst   int   Taillard's instance to solve (between 001 and 120)");
    writeln("   --lb     str   lower bound function (lb1, lb1_d, lb2)");
    writeln("   --ub     int   initial upper bound (0, 1)\n");
  }
}
