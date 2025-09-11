module main_nqueens
{
  use util;

  use nqueens_problem;
  use nqueens_search_sequential;
  use nqueens_search_gpu;
  use nqueens_search_multigpu;
  use nqueens_search_distributed;

  config const mode: string = "multigpu";

  proc main(args: [] string): int
  {
    // Helper
    for a in args[1..] {
      if (a == "-h" || a == "--help") {
        common_help_message();
        nqueens_help_message();

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
