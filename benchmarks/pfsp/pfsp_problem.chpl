module pfsp_problem
{
  proc pfsp_help_message(): void
  {
    writeln("\n  PFSP Benchmark Parameters:\n");
    writeln("   --inst   int   Taillard's instance to solve (between 001 and 120)");
    writeln("   --lb     str   lower bound function (lb1, lb1_d, lb2)");
    writeln("   --ub     int   initial upper bound (0, 1)\n");
  }
}
