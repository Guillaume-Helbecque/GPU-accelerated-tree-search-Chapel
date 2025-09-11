module main_nqueens
{
  use nqueens_search_sequential;
  use nqueens_search_gpu;
  use nqueens_search_multigpu;
  use nqueens_search_distributed;

  config const mode: string = "multigpu";

  proc main(args: [] string): int
  {
    // Search
    select mode {
      when "sequential" {
        search_sequential(args);
      }
      when "gpu" {
        search_gpu(args);
      }
      when "multigpu" {
        search_multigpu(args);
      }
      when "distributed" {
        search_distributed(args);
      }
      otherwise {
        halt("unknown execution mode");
      }
    }

    return 0;
  }
}
