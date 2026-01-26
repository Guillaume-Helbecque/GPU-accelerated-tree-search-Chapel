module main_pfsp
{
  // Common modules
  use util;

  // Problem-specific modules
  use pfsp_problem;
  use pfsp_search_sequential;
  use pfsp_search_gpu;
  use pfsp_search_multigpu;
  use pfsp_search_distributed;

  // Common options
  config const mode: string = "multigpu";
  config const m = 25;
  config const M = 50000;
  config const D = 1;

  // Problem-specific option
  config const inst: int = 14; // instance
  config const lb: string = "lb1"; // lower bound function
  config const ub: int = 1; // initial upper bound
  /*
    NOTE: Only forward branching is considered because other strategies increase a
    lot the implementation complexity and do not add much contribution.
  */

  proc main(args: [] string): int
  {
    // Helper
    for a in args[1..] {
      if (a == "-h" || a == "--help") {
        common_help_message(args[0]);
        pfsp_help_message();

        return 1;
      }
    }

    // Search
    select mode {
      when "sequential" {
        search_sequential();
      }
      when "gpu" {
        search_gpu();
      }
      when "multigpu" {
        search_multigpu();
      }
      when "distributed" {
        search_distributed();
      }
      otherwise {
        halt("unknown execution mode");
      }
    }

    return 0;
  }
}
