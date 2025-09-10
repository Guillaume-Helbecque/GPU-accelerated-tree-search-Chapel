module main_pfsp
{
  use pfsp_chpl;
  use pfsp_gpu_chpl;
  use pfsp_multigpu_chpl;
  use pfsp_dist_multigpu_chpl;

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
