module main_qap
{
  // Common modules
  use util;

  // Problem-specific modules
  use Problem_qap;
  use qap_search_sequential_glb;
  use qap_search_sequential_rlt1;
  use qap_search_gpu_glb;
  use qap_search_gpu_rlt1;
  use qap_search_gpu_qpb;
  use qap_search_multigpu_glb;
  use qap_search_multigpu_qpb;
  use qap_search_distributed_qpb;

  // Common options
  config const mode: string = "multigpu";
  config const m = 25;
  // M = batched buffer (max kernel-input size). The QPB driver scales
  // well up to ~200K; RLT1 has larger per-node state and may need a
  // smaller M (override with --M=50000 if you hit out-of-memory).
  config const M = 200000;
  config const D = 1;

  // Problem-specific option
  config const inst = "10_sqn,16_melbourne";
  config const itmax: int(32) = 10;
  config const lb: string = "GLB";
  config const ub: string = "heuristic"; // heuristic

  // QPB-specific options.
  //
  // qpb_maxFW, qpb_sinkIter, qpb_tol are declared `config param` (compile-
  // time) instead of `config const` so the inner FW / Sinkhorn loops can
  // be unrolled and the FW convergence test against `tol` can be folded
  // by the compiler. To experiment with different values, recompile with
  // `-sqpb_maxFW=N`, `-sqpb_sinkIter=N`, `-sqpb_tol=...`.
  config param qpb_maxFW:    int      = 15;
  config param qpb_tol:      real(64) = 1.0e-5;
  config param qpb_sinkIter: int      = 5;
  // qpb_xPoolSize is a runtime const (does not enter any loop bound);
  // sized for the per-task pool of bounded nodes carrying their qpbX.
  config const qpb_xPoolSize: int     = 500000;

  // Per-phase timing of the single-GPU QPB main loop. When true, qap_search
  // accumulates wall time across six phases (prep / h2d / eigen-setup-kernel
  // / qpb-kernel / d2h / gen) plus iteration/batch counters and prints a
  // summary block at the end. Default false so production runs are
  // unaffected. Toggle with `--qpb_profile=true`.
  config const qpb_profile: bool = false;

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
        if lb == "GLB" then search_sequential_glb();
        else if lb == "RLT1" then search_sequential_rlt1();
        else if lb == "QPB" then halt("'sequential' execution mode with QPB not implemented (QPB is GPU-only)");
        else halt("unknown bounding function");
      }
      when "gpu" {
        if lb == "GLB" then search_gpu_glb();
        else if lb == "RLT1" then search_gpu_rlt1();
        else if lb == "QPB" then search_gpu_qpb();
        else halt("unknown bounding function");
      }
      when "multigpu" {
        if lb == "GLB" then search_multigpu_glb();
        else if lb == "RLT1" then halt("'multigpu' execution mode with RLT1 not yet implemented");
        else if lb == "QPB" then search_multigpu_qpb();
        else halt("unknown bounding function");
      }
      when "distributed" {
        if lb == "GLB" then halt("'distributed' execution mode with GLB not yet implemented");
        else if lb == "RLT1" then halt("'distributed' execution mode with RLT1 not yet implemented");
        else if lb == "QPB" then search_distributed_qpb();
        else halt("unknown bounding function");
      }
      otherwise {
        halt("unknown execution mode");
      }
    }

    return 0;
  }
}
