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
