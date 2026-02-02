module main_qap
{
  // Common modules
  use util;

  // Problem-specific modules
  use Problem_qap;
  use qap_search_sequential_glb;
  use qap_search_sequential_hhb;
  use qap_search_gpu_glb;
  use qap_search_gpu_iglb;
  use qap_search_gpu_hhb;
  use qap_search_multigpu_glb;
  use qap_search_multigpu_iglb;
  use qap_search_distributed_glb;

  // Common options
  config const mode: string = "multigpu";
  config const m = 25;
  config const M = 50000;
  config const D = 1;

  // Problem-specific option
  config const inst = "10_sqn,16_melbourne";
  config const itmax: int(32) = 10;
  config const lb: string = "glb";
  config const ub: string = "heuristic"; // heuristic

  proc main(args: [] string): int
  {
    // Helper
    for a in args[1..] {
      if (a == "-h" || a == "--help") {
        common_help_message(args[0]);
        qap_help_message();

        return 1;
      }
    }

    // Search
    select mode {
      when "sequential" {
        if lb == "glb" then search_sequential_glb();
        else if lb == "iglb" then halt("'sequential' execution mode with IGLB not yet implemented");
        else if lb == "hhb" then search_sequential_hhb();
        else halt("unknown bounding function");
      }
      when "gpu" {
        if lb == "glb" then search_gpu_glb();
        else if lb == "iglb" then search_gpu_iglb();
        else if lb == "hhb" then search_gpu_hhb();
        else halt("unknown bounding function");
      }
      when "multigpu" {
        if lb == "glb" then search_multigpu_glb();
        else if lb == "iglb" then search_multigpu_iglb();
        else if lb == "hhb" then halt("'multigpu' execution mode with HHB not yet implemented");
        else halt("unknown bounding function");
      }
      when "distributed" {
        if lb == "glb" then search_distributed_glb();
        else if lb == "iglb" then halt("'distributed' execution mode with IGLB not yet implemented");
        else if lb == "hhb" then halt("'distributed' execution mode with HHB not yet implemented");
      }
      otherwise {
        halt("unknown execution mode");
      }
    }

    return 0;
  }
}
