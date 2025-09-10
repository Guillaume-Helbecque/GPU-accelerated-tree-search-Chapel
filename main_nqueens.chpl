module main_nqueens
{
  use nqueens_chpl;
  use nqueens_gpu_chpl;
  use nqueens_multigpu_chpl;
  use nqueens_dist_multigpu_chpl;

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
