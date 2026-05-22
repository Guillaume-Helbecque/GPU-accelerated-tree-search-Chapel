module qap_search_multigpu_qpb
{
  /*
    Multi-GPU B&B for QAP using the QPB lower-bound operator.

    Architectural notes:
      * One coforall task per GPU. Each task owns its local pool, its own
        persistent qpbXPool, its own per-batch eigen / C caches, and its
        own free-slot allocator. There is no shared device memory between
        tasks -- each on-device array lives on `here.gpus[gpuID]`.
      * The host-side multi-pool array uses a work-stealing protocol: each
        worker pops from the back of its own local pool and steals from
        the front of a victim's pool when its own runs dry.
      * When a node is stolen, the stolen copy has its `hasBoundData` flag
        cleared and its `qpbXSlot` invalidated. The original slot in the
        victim's qpbXPool is left allocated (not returned to the victim's
        free list) for two reasons:
          1. The victim's free list is accessed inside its prepareChildren
             without taking the multi-pool lock, so a thief pushing onto
             it would race.
          2. The leakage is bounded by the work-steal count, which is far
             smaller than the per-GPU qpb_xPoolSize.
        Stolen nodes therefore re-bound from a cold (uniform 1/m) FW start.
        This costs a few FW iterations relative to the in-pool warm-start
        path but keeps the cross-task interface lock-free.
  */

  use Time;
  use Math;
  use Random;
  use GpuDiagnostics;

  use util;
  use Pool_par;
  use QAP_node;
  use Util_qap;
  use Problem_qap;
  use Problem_qpb;

  import main_qap.m as m;
  import main_qap.M as M;
  import main_qap.D as D;

  import main_qap.inst as inst;
  import main_qap.lb as lb;
  import main_qap.ub as ub;
  import main_qap.qpb_xPoolSize as qpb_xPoolSize;
  import main_qap.qpb_maxFW    as qpb_maxFW;
  import main_qap.qpb_tol      as qpb_tol;
  import main_qap.qpb_sinkIter as qpb_sinkIter;

  config param sizeMax: int(32) = 32;

  config const BLOCK_SIZE = 32;

  var benchmark: string;
  var n, N: int(32);
  var initUB: int;

  /******************** Helpers shared with single-GPU *************************/

  // Per-GPU free-slot allocator. LIFO stack of qpbXPool slot indices.
  record SlotStack_par {
    var dom: domain(1);
    var elements: [dom] int(32);
    var top: int;

    proc init() {
      this.dom = {0..#1};
    }

    proc ref init_with_capacity(cap: int) {
      this.dom = {0..#cap};
      for s in 0..#cap do this.elements[s] = (cap - 1 - s):int(32);
      this.top = cap;
    }

    proc ref push(slot: int(32)) {
      this.elements[this.top] = slot;
      this.top += 1;
    }

    proc ref pop(): int(32) {
      this.top -= 1;
      return this.elements[this.top];
    }

    proc count(): int { return this.top; }
  }

  record SlotList_par {
    var dom: domain(1);
    var elements: [dom] int(32);
    var size: int;

    proc init() {
      this.dom = {0..#16};
    }

    proc ref push(slot: int(32)) {
      if (this.size >= this.dom.size) {
        this.dom = {0..#(this.dom.size * 2)};
      }
      this.elements[this.size] = slot;
      this.size += 1;
    }

    proc ref clear() { this.size = 0; }
  }

  /******************** CPU-side helpers ***************************************/

  proc decompose_nobound_par(const ref parent: Node_QPB,
                             const ref priority_fac: [] int(32),
                             const ref priority_loc: [] int(32),
                             ref tree_loc: uint,
                             ref pool: SinglePool_par(Node_QPB))
  {
    const depth = parent.depth;
    if (depth == n) then return;

    const i = priority_fac[depth];
    for j0 in 0..<N by -1 {
      const j = priority_loc[j0];
      if !parent.available[j] then continue;

      var child: Node_QPB;
      child.mapping     = parent.mapping;
      child.depth       = depth + 1;
      child.available   = parent.available;
      child.mapping[i]    = j:int(8);
      child.available[j]  = false;
      child.hasBoundData  = false;
      child.qpbXSlot      = -1;
      child.qpbBoundContLast = 0.0;
      child.qpbFixedCost = 0;
      child.lb = 0;
      pool.pushBackFree(child);
      tree_loc += 1;
    }
  }

  proc prepareChildren_par(ref inputs: [] Node_QPB_In,
                           ref parent_first_child_idx: [] int(32),
                           ref parent_count: int(32),
                           ref freeSlots: SlotStack_par,
                           ref deferred_free: SlotList_par,
                           const ref queue_fac: [] int(32),
                           const ref queue_loc: [] int(32),
                           const ref F: [] int(32),
                           const ref DD: [] int(32),
                           ref pool: SinglePool_par(Node_QPB),
                           ref UB: int,
                           ref exploredTree: uint,
                           ref exploredSol: uint): int(32)
  {
    var size: int(32) = 0;
    parent_count = 0;

    pool.acquireLock();

    while (true) {
      if (size + N > M)            then break;
      if (freeSlots.count() < N)   then break;

      var hasWork = 0;
      var parent = pool.popBackFree(hasWork);
      if !hasWork then break;

      if (parent.lb >= UB) {
        if (parent.hasBoundData && parent.qpbXSlot >= 0) then
          freeSlots.push(parent.qpbXSlot);
        continue;
      }

      const dp = parent.depth;
      if (dp == n) {
        const cost = ObjectiveFunction(parent.mapping, DD, F, n, N);
        if (cost < UB) then UB = cost;
        exploredSol += 1;
        if (parent.hasBoundData && parent.qpbXSlot >= 0) then
          freeSlots.push(parent.qpbXSlot);
        continue;
      }

      var loc_to_idx: sizeMax*int(32);
      for k in 0..<sizeMax do loc_to_idx[k] = -1:int(32);
      var m_parent: int(32) = 0;
      for l in 0..<N {
        if parent.available[l] {
          loc_to_idx[l] = m_parent;
          m_parent += 1;
        }
      }

      var parent_i_idx:      int(32) = -1;
      var parent_next_i_idx: int(32) = -1;
      const branching_fac      = queue_fac[dp:int(32)];
      const next_branching_fac: int(32) =
          if (dp:int(32) + 1) < n then queue_fac[dp:int(32) + 1] else (-1):int(32);
      {
        var idx: int(32) = 0;
        for f in 0..<n {
          if (parent.mapping[f] == -1:int(8)) {
            if (f == branching_fac)      then parent_i_idx      = idx;
            if (f == next_branching_fac) then parent_next_i_idx = idx;
            idx += 1;
          }
        }
      }

      var child_next_bf_idx: int(8) = -1;
      if (parent_next_i_idx >= 0 && parent_i_idx >= 0) {
        child_next_bf_idx =
          (if parent_next_i_idx < parent_i_idx
             then parent_next_i_idx else parent_next_i_idx - 1):int(8);
      }

      const i = branching_fac;
      var parent_eigen_slot: int(32) = -1;
      const parent_can_warmstart =
        parent.hasBoundData && (parent_i_idx >= 0) && (m_parent > 2)
        && (parent.qpbXSlot >= 0);

      for j0 in 0..<N by -1 {
        const j = queue_loc[j0];
        if !parent.available[j] then continue;

        if (parent.hasBoundData && parent_i_idx >= 0) {
          const j_idx = loc_to_idx[j];
          if (j_idx >= 0) {
            const est = parent.qpbFixedCost:real(64)
                      + parent.qpbBoundContLast
                      + parent.qpbReducedCostsRow[j_idx]:real(64);
            const vf_tol = max(HOST_QPB_VF_TOL_BASE, HOST_QPB_VF_TOL_REL * abs(est));
            if (ceil(est - vf_tol):int >= UB) then continue;
          }
        }

        if (dp:int(32) + 1 == n) {
          var leaf_map: sizeMax*int(8) = parent.mapping;
          leaf_map[i] = j:int(8);
          const cost = ObjectiveFunction(leaf_map, DD, F, n, N);
          if (cost < UB) then UB = cost;
          exploredSol += 1;
          exploredTree += 1;
          continue;
        }

        const child_out_slot = freeSlots.pop();

        ref ip = inputs[size];
        ip.mapping   = parent.mapping;
        ip.available = parent.available;
        ip.depth     = (dp + 1):int(8);
        ip.mapping[i]   = j:int(8);
        ip.available[j] = false;
        ip.parentIdxFac     = -1;
        ip.parentIdxLoc     = -1;
        ip.nextBranchFacIdx = child_next_bf_idx;
        ip.parentEigenIdx   = -1;
        ip.parentXSlot      = -1;
        ip.outSlot          = child_out_slot;

        if parent_can_warmstart {
          const j_idx = loc_to_idx[j];
          if (j_idx >= 0) {
            if (parent_eigen_slot < 0) {
              parent_eigen_slot = parent_count;
              parent_first_child_idx[parent_count] = size;
              parent_count += 1;
            }
            ip.parentIdxFac   = parent_i_idx:int(8);
            ip.parentIdxLoc   = j_idx:int(8);
            ip.parentEigenIdx = parent_eigen_slot;
            ip.parentXSlot    = parent.qpbXSlot;
          }
        }

        size += 1;
        exploredTree += 1;
      }

      if (parent.hasBoundData && parent.qpbXSlot >= 0) {
        if (parent_eigen_slot >= 0) then deferred_free.push(parent.qpbXSlot);
        else                            freeSlots.push(parent.qpbXSlot);
      }
    }

    pool.releaseLock();
    return size;
  }

  proc generate_children_par(const ref inputs:  [] Node_QPB_In,
                             const ref outputs: [] Node_QPB_Out,
                             const ref bounds:  [] int,
                             const size: int,
                             ref freeSlots: SlotStack_par,
                             ref pool: SinglePool_par(Node_QPB),
                             const UB: int)
  {
    pool.acquireLock();
    for k in 0..<size {
      const ref ip = inputs[k];
      const ref op = outputs[k];
      const lb_k = bounds[k];
      if (lb_k < UB) {
        var child: Node_QPB;
        child.mapping   = ip.mapping;
        child.available = ip.available;
        child.depth     = ip.depth:uint(8);
        child.hasBoundData = true;
        child.qpbXSlot     = ip.outSlot;
        const m_child = n - ip.depth:int(32);
        for j in 0..<m_child do
          child.qpbReducedCostsRow[j] = op.qpbReducedCostsRow[j];
        child.qpbBoundContLast = op.qpbBoundContLast;
        child.qpbFixedCost     = op.qpbFixedCost;
        child.lb               = lb_k;
        pool.pushBackFree(child);
      } else {
        if (ip.outSlot >= 0) then freeSlots.push(ip.outSlot);
      }
    }
    pool.releaseLock();
  }

  /******************** GPU kernel wrappers ************************************/

  // Per-parent eigen + C-matrix setup kernel. One thread per parent slot.
  // `@gpu.blockSize(32)` is required: the per-thread working set inside
  // parent_eigen_setup_proc (A_hat / tmpM / Fa / FtF / evec_F plus the
  // Helmert prefix-sum scratch) is too large to leave headroom for the
  // runtime's default block size, which can otherwise hit a "too many
  // resources requested for launch" failure at kernel dispatch.
  proc parent_eigen_setup_gpu_par(const ref inputs_d:       [] Node_QPB_In,
                                  const ref parent_first_d: [] int(32),
                                  const parent_count:       int(32),
                                  const ref F_d:            [] int(32),
                                  const ref DD_d:           [] int(32),
                                  const ref queue_fac_d:    [] int(32),
                                  ref parentW_d:       [] real(32),
                                  ref parentSigma_d:   [] real(64),
                                  ref parentVW_d:      [] real(32),
                                  ref parentFnormFa_d: [] real(64),
                                  ref parentSingFa_d:  [] real(64),
                                  ref parent_C_d:      [] real(32))
  {
    @assertOnGpu
    @gpu.blockSize(32)
    foreach slot in 0..<parent_count {
      const first_child = parent_first_d[slot];
      parent_eigen_setup_proc(inputs_d[first_child], F_d, DD_d, queue_fac_d,
                              n, N, slot,
                              parentW_d, parentSigma_d, parentVW_d,
                              parentFnormFa_d, parentSingFa_d, parent_C_d);
    }
  }

  // Per-child QPB bound kernel. Same `@gpu.blockSize` rationale; the
  // per-thread working set here is even larger (~11 sizeMaxSq float
  // matrices + LAP scratch).
  proc evaluate_qpb_gpu_par(const ref inputs_d:  [] Node_QPB_In,
                            ref outputs_d:       [] Node_QPB_Out,
                            ref bounds_d:        [] int,
                            const numChildren:   int(32),
                            ref parentXPool_d:        [] real(32),
                            const ref parentW_d:      [] real(32),
                            const ref parentSigma_d:  [] real(64),
                            const ref parentVW_d:     [] real(32),
                            const ref parentFnormFa_d:[] real(64),
                            const ref parentSingFa_d: [] real(64),
                            const ref parent_C_d:     [] real(32),
                            const ref F_d:            [] int(32),
                            const ref DD_d:           [] int(32),
                            const ref queue_fac_d:    [] int(32),
                            const upperBound:    int,
                            param maxFW:         int,
                            param tol:           real(64),
                            param sinkIter:      int)
  {
    @assertOnGpu
    @gpu.blockSize(32)
    foreach tid in 0..<numChildren {
      bound_QPB_kernel(inputs_d[tid], outputs_d[tid], bounds_d[tid],
                       parentXPool_d,
                       parentW_d, parentSigma_d, parentVW_d,
                       parentFnormFa_d, parentSingFa_d, parent_C_d,
                       F_d, DD_d, queue_fac_d,
                       n, N, upperBound, maxFW, tol, sinkIter);
    }
  }

  /******************** Main entry point ***************************************/

  proc qap_search(ref optimum: int, ref exploredTree: uint, ref exploredSol: uint,
                  ref elapsedTime: real)
  {
    var timer: stopwatch;

    var domF, domD: domain(1, idxType = int(32));
    var F:  [domF] int(32);
    var DD: [domD] int(32);

    readInstance(inst, n, N, domF, domD, F, DD, benchmark);

    // ---- Step 0: preprocessing ----
    timer.start();
    var priority_fac: [0..<sizeMax] int(32);
    var priority_loc: [0..<sizeMax] int(32);
    Prioritization(priority_fac, F, n, ascend = false);
    if (benchmark == "qubitAlloc") then
      Prioritization_loc_connec(priority_loc, DD, N);
    else
      Prioritization(priority_loc, DD, N);

    if (ub == "heuristic") then initUB = GreedyAllocation(DD, F, priority_fac, n, N);
    else {
      try! initUB = ub:int;
    }

    timer.stop();
    const res0 = timer.elapsed();
    print_settings_mqpb(benchmark, inst, n, N, qpb_maxFW, qpb_tol,
                        m, M, BLOCK_SIZE, lb, ub, initUB);

    var best: int = initUB;

    // ---- Step 1: initial CPU BFS to populate D*m nodes ----
    timer.start();
    var root = new Node_QPB(n);
    var pool = new SinglePool_par(Node_QPB);
    pool.pushBackFree(root);

    while (pool.size < D*m) {
      var hasWork = 0;
      var parent = pool.popFrontFree(hasWork);
      if !hasWork then break;
      decompose_nobound_par(parent, priority_fac, priority_loc, exploredTree, pool);
    }

    timer.stop();
    const res1 = (timer.elapsed() - res0, exploredTree, exploredSol);

    // ---- Step 2: distribute pool across D workers and run multi-GPU loop ----
    timer.start();

    var eachExploredTree, eachExploredSol: [0..#D] uint = noinit;
    var eachBest: [0..#D] int = noinit;
    var eachTaskState: [0..#D] atomic bool = BUSY;
    var allTasksIdleFlag: atomic bool = false;

    const poolSize = pool.size;
    const c = poolSize / D;
    const l = poolSize - (D-1)*c;
    const f = pool.front;

    pool.front = 0;
    pool.size = 0;

    var multiPool: [0..#D] SinglePool_par(Node_QPB);

    coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol,
      ref eachBest, ref eachTaskState, ref multiPool) {

      const device = here.gpus[gpuID];

      var tree, sol: uint;
      ref pool_loc = multiPool[gpuID];
      var best_l = best;
      var taskState: bool = BUSY;

      // each task gets its chunk (stride-D split, last task gets the tail)
      pool_loc.elements[0..#c] = pool.elements[gpuID+f.. by D #c];
      pool_loc.size += c;
      if (gpuID == D-1) {
        pool_loc.elements[c..#(l-c)] = pool.elements[(D*c)+f..#(l-c)];
        pool_loc.size += l-c;
      }

      // Host-side per-batch buffers (per-task).
      var inputs:   [0..#M] Node_QPB_In;
      var outputs:  [0..#M] Node_QPB_Out;
      var bounds:   [0..#M] int;
      var parent_first_h: [0..#M] int(32);

      var freeSlots: SlotStack_par;
      freeSlots.init_with_capacity(qpb_xPoolSize);
      var deferred_free: SlotList_par;

      // Device-side per-batch buffers (per-task, on this task's GPU).
      on device var inputs_d:   [0..#M] Node_QPB_In;
      on device var outputs_d:  [0..#M] Node_QPB_Out;
      on device var bounds_d:   [0..#M] int;
      on device var parent_first_d: [0..#M] int(32);

      on device var parentW_d:       [0..#(M*sizeMaxSq)] real(32);
      on device var parentSigma_d:   [0..#(M*sizeMax)]   real(64);
      on device var parentVW_d:      [0..#(M*sizeMaxSq)] real(32);
      on device var parentFnormFa_d: [0..#M]             real(64);
      on device var parentSingFa_d:  [0..#(M*sizeMax)]   real(64);
      on device var parent_C_d:      [0..#(M*sizeMaxSq)] real(32);

      on device var qpbXPool_d: [0..#(qpb_xPoolSize*sizeMaxSq)] real(32);

      on device const F_d  = F;
      on device const DD_d = DD;
      on device const queue_fac_d = priority_fac;

      while true {
        var parent_count: int(32) = 0;
        const numChildren = prepareChildren_par(
            inputs, parent_first_h, parent_count,
            freeSlots, deferred_free,
            priority_fac, priority_loc, F, DD,
            pool_loc, best_l, tree, sol);

        if (numChildren > 0) {
          if (taskState == IDLE) {
            taskState = BUSY;
            eachTaskState[gpuID].write(BUSY);
          }

          inputs_d = inputs;
          if (parent_count > 0) then
            parent_first_d = parent_first_h;

          if (parent_count > 0) then
            on device do parent_eigen_setup_gpu_par(
                inputs_d, parent_first_d, parent_count,
                F_d, DD_d, queue_fac_d,
                parentW_d, parentSigma_d, parentVW_d,
                parentFnormFa_d, parentSingFa_d, parent_C_d);

          on device do evaluate_qpb_gpu_par(
              inputs_d, outputs_d, bounds_d, numChildren,
              qpbXPool_d,
              parentW_d, parentSigma_d, parentVW_d,
              parentFnormFa_d, parentSingFa_d, parent_C_d,
              F_d, DD_d, queue_fac_d,
              best_l, qpb_maxFW, qpb_tol, qpb_sinkIter);

          outputs = outputs_d;
          bounds  = bounds_d;

          generate_children_par(inputs, outputs, bounds, numChildren,
                                freeSlots, pool_loc, best_l);

          for k in 0..<deferred_free.size do
            freeSlots.push(deferred_free.elements[k]);
          deferred_free.clear();
        } else {
          // Work stealing.
          var tries = 0;
          var stole = false;
          const victims = permute(0..#D);

          label WS0 while (tries < D && stole == false) {
            const victimID = victims[tries];

            if (victimID != gpuID) {
              ref victim = multiPool[victimID];
              var nn = 0;

              label WS1 while (nn < 10) {
                if victim.lock.compareAndSwap(false, true) {
                  const size = victim.size;

                  if (size >= 2*m) {
                    var (hasWork, p) = victim.popFrontBulkFree(m, M);
                    if (hasWork == 0) {
                      victim.lock.write(false);
                      halt("DEADCODE in QPB work stealing");
                    }

                    // Invalidate qpbX warm-start on stolen nodes. The
                    // victim's pool slots are left allocated (leaked) --
                    // see the file-header comment for the rationale.
                    for k in 0..<hasWork {
                      if p[k].hasBoundData {
                        p[k].hasBoundData = false;
                        p[k].qpbXSlot = -1;
                      }
                    }

                    pool_loc.pushBackBulk(p);
                    stole = true;
                    victim.lock.write(false);
                    break WS0;
                  }

                  victim.lock.write(false);
                  break WS1;
                }
                nn += 1;
                currentTask.yieldExecution();
              }
            }
            tries += 1;
          }

          if (stole == false) {
            // termination
            if (taskState == BUSY) {
              taskState = IDLE;
              eachTaskState[gpuID].write(IDLE);
            }
            if allIdle(eachTaskState, allTasksIdleFlag) {
              /* writeln("task ", gpuID, " exits normally"); */
              break;
            }
            continue;
          } else {
            continue;
          }
        }
      }

      // Migrate any leftover local pool entries back to the global pool.
      const poolLocSize = pool_loc.size;
      for p in 0..#poolLocSize {
        var hasWork = 0;
        const elt = pool_loc.popBack(hasWork);
        if !hasWork then break;
        pool.pushBack(elt);
      }

      eachExploredTree[gpuID] = tree;
      eachExploredSol[gpuID]  = sol;
      eachBest[gpuID]         = best_l;
    }

    timer.stop();
    const res2 = (timer.elapsed(), exploredTree, exploredSol) - res1;

    exploredTree += (+ reduce eachExploredTree);
    exploredSol  += (+ reduce eachExploredSol);
    best          = (min reduce eachBest);

    // Per-GPU workload share (% of GPU-loop nodes explored per task).
    // Printed after "Exploration terminated." so it sits between the settings
    // and results blocks rather than inside either.
    const workloadArr = 100.0*eachExploredTree/(exploredTree-res1[1]):real;

    // ---- Step 3: final CPU DFS drain ----
    timer.start();
    while true {
      var hasWork = 0;
      var parent = pool.popBackFree(hasWork);
      if !hasWork then break;
      const depth = parent.depth;
      if (depth == n) {
        const cost = ObjectiveFunction(parent.mapping, DD, F, n, N);
        if (cost < best) then best = cost;
        exploredSol += 1;
        continue;
      }
      decompose_nobound_par(parent, priority_fac, priority_loc, exploredTree, pool);
    }

    timer.stop();
    elapsedTime = timer.elapsed();
    const res3 = (elapsedTime, exploredTree, exploredSol) - res1 - res2;

    optimum = best;
    writeln("\nExploration terminated.");
    writeln("Workload per GPU: ", workloadArr);
  }

  proc search_multigpu_qpb()
  {
    writeln("Multi-GPU execution mode (QPB) with ", D, " GPUs");

    var optimum: int;
    var exploredTree: uint = 0;
    var exploredSol:  uint = 0;
    var elapsedTime:  real;

    startGpuDiagnostics();
    qap_search(optimum, exploredTree, exploredSol, elapsedTime);
    stopGpuDiagnostics();

    print_results(optimum, exploredTree, exploredSol, elapsedTime, initUB);

    return 0;
  }

  // QPB multi-GPU settings printer. Labels are padded to 29 chars (values
  // start at column 30) so the settings block prints as an aligned table.
  proc print_settings_mqpb(const benchmark: string, const inst: string,
                           const n: int(32), const N: int(32),
                           const maxFW: int, const tol: real(64),
                           const minBatch: int, const maxBatch: int,
                           const blockSize: int,
                           const lb: string, const ub: string, const initUB: int): void
  {
    writeln("\n=================================================");
    if (benchmark == "qap") {
      writeln("QAP instance:                ", inst);
      writeln("Number of facilities:        ", N);
    } else if (benchmark == "qubitAlloc") {
      var getFilenames = inst.split(",");
      writeln("Circuit:                     ", getFilenames[0]);
      writeln("Device:                      ", getFilenames[1]);
      writeln("Number of logical qubits:    ", n);
      writeln("Number of physical qubits:   ", N);
    }
    writeln("Number of GPUs:              ", D);
    writeln("Min GPU batch size (m):      ", minBatch);
    writeln("Max GPU batch size (M):      ", maxBatch);
    writeln("GPU block size:              ", blockSize);
    const heuristic = if (ub == "heuristic") then " (heuristic)" else "";
    writeln("Initial upper bound:         ", initUB, heuristic);
    writeln("Lower bound function:        ", lb);
    writeln("Max Frank-Wolfe iterations:  ", maxFW);
    writeln("Relative FW duality gap tol: ", tol);
    writeln("=================================================");
  }
}
