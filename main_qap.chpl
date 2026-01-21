module main_qap
{
  // Common modules
  use util;

  // Problem-specific modules
  use Problem_qap;
  use qap_search_sequential_glb;
  use qap_search_sequential_hhb;
  use qap_search_gpu_glb;
  use qap_search_gpu_hhb;
  use qap_search_multigpu_glb;

  // Common options
  config const mode: string = "multigpu";

  // Problem-specific option
  config const lb: string = "glb";

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
        else if lb == "hhb" then search_sequential_hhb();
        else halt("unknown bounding function");
      }
      when "gpu" {
        if lb == "glb" then search_gpu_glb();
        else if lb == "hhb" then search_gpu_hhb();
        else halt("unknown bounding function");
      }
      when "multigpu" {
        if lb == "glb" then search_multigpu_glb();
        else if lb == "hhb" then halt("'multigpu' execution mode with HHB not yet implemented");
        else halt("unknown bounding function");
      }
      when "distributed" {
        halt("'distributed' execution mode not yet implemented");
      }
      otherwise {
        halt("unknown execution mode");
      }
    }

    return 0;
  }
}
