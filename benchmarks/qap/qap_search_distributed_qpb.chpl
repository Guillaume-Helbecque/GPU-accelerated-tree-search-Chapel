module qap_search_distributed_qpb
{
  /*
    Distributed multi-GPU B&B for QAP using the QPB lower-bound operator.

    Architecture (three-level):
      * One coforall task per locale (via `on loc`). Each locale owns its own
        partition of the work pool and runs `D` GPU tasks on its `D` local
        GPUs.
      * One coforall task per GPU within each locale. Each GPU task owns its
        local pool, its persistent qpbXPool, its per-batch eigen / C caches,
        and its free-slot allocator. There is no shared device memory
        between tasks: each on-device array lives on `here.gpus[gpuID]` of
        the current locale.
      * Work stealing operates at both levels: a per-GPU task first tries
        to steal from other GPU tasks on its own locale; if that fails it
        tries to steal from GPU tasks on remote locales via the
        distributed array `distMultiPool[PrivateSpace][0..#D]`.

    Stolen-node handling:
      Whether a steal is intra- or inter-locale, the stolen node copy has
      its `hasBoundData` flag cleared and its `qpbXSlot` invalidated so the
      thief re-bounds from a cold (uniform 1/m) FW start. The victim's
      qpbXPool slot is left allocated -- the victim's free list cannot be
      mutated from another task / locale without lock contention or remote
      atomic ops, and the leak is bounded by the total work-steal count,
      which is far smaller than the per-GPU qpb_xPoolSize.

    Termination:
      Per-task IDLE/BUSY state is tracked in `eachTaskState[0..#D]` (one
      array per locale). Per-locale IDLE/BUSY state is tracked in
      `eachLocaleState[PrivateSpace]` (one slot per locale, accessible
      across locales). A task is idle when its pool is empty and all steal
      attempts failed; a locale is idle when all its tasks are idle;
      termination fires when all locales are idle.
  */

  use Time;
  use Math;
  use Random;
  use PrivateDist;
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

  // Module-level locale-0 state (set in qap_search, read by the print
  // helpers on locale 0). Worker procs receive n / N as explicit args
  // so the values survive an `on loc` jump.
  var benchmark: string;
  var initUB: int;

  /******************** Slot-allocator helper records **************************/

  /*
    SlotStack_dist: LIFO pool of free qpbXPool slot indices. One instance
    per GPU task on each locale.
  */
  record SlotStack_dist {
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

  /*
    SlotList_dist: append-only buffer for parent slots deferred until the
    current batch's kernel has drained. Cleared each iteration.
  */
  record SlotList_dist {
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

  // CPU-side expansion (no bound). Used by the initial BFS to populate the
  // pool with children that the first GPU batch will bound from scratch.
  proc decompose_nobound_dist(const ref parent: Node_QPB,
                              const ref priority_fac: [] int(32),
                              const ref priority_loc: [] int(32),
                              ref tree_loc: uint,
                              ref pool: SinglePool_par(Node_QPB),
                              const n: int(32), const N: int(32))
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

  /*
    prepareChildren_dist: pop parents off the back of `pool`, generate children
    (with QPB variable-fixing against the parent's stored reduced costs),
    write each child's Node_QPB_In into `inputs`, and assign each child a
    fresh slot in the persistent qpbXPool free list.

    Leaves are evaluated inline. Parents whose post-VF children contain at
    least one warm-startable child are deferred-freed.

    Returns the number of children written; sets `parent_count` to the
    number of eigen-cache slots assigned.
  */
  proc prepareChildren_dist(ref inputs: [] Node_QPB_In,
                            ref parent_first_child_idx: [] int(32),
                            ref parent_count: int(32),
                            ref freeSlots: SlotStack_dist,
                            ref deferred_free: SlotList_dist,
                            const ref queue_fac: [] int(32),
                            const ref queue_loc: [] int(32),
                            const ref F: [] int(32),
                            const ref DD: [] int(32),
                            ref pool: SinglePool_par(Node_QPB),
                            ref UB: int,
                            ref exploredTree: uint,
                            ref exploredSol: uint,
                            const n: int(32), const N: int(32)): int(32)
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

  /*
    generate_children_dist: after the kernel drains, combine inputs + outputs
    + bounds into a Node_QPB for each survivor and push back into the pool.
    Pruned children's qpbX slots are reclaimed.
  */
  proc generate_children_dist(const ref inputs:  [] Node_QPB_In,
                              const ref outputs: [] Node_QPB_Out,
                              const ref bounds:  [] int,
                              const size: int,
                              ref freeSlots: SlotStack_dist,
                              ref pool: SinglePool_par(Node_QPB),
                              const UB: int,
                              const n: int(32))
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
  // parent_eigen_setup_proc (several sizeMaxSq float tuples for A_hat /
  // tmpM / Fa / FtF / evec_F plus the Helmert prefix-sum scratch) is too
  // large to leave headroom for the runtime's default block size, which
  // can otherwise hit a "too many resources requested for launch" failure
  // at kernel dispatch.
  proc parent_eigen_setup_gpu_dist(const ref inputs_d:       [] Node_QPB_In,
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
                                   ref parent_C_d:      [] real(32),
                                   const n: int(32), const N: int(32))
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

  // Per-child QPB bound kernel. One thread per child. Same `@gpu.blockSize`
  // rationale; the per-thread working set here is even larger (~11
  // sizeMaxSq float matrices + LAP scratch).
  proc evaluate_qpb_gpu_dist(const ref inputs_d:  [] Node_QPB_In,
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
                             const n: int(32), const N: int(32),
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

    var n, N: int(32);
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
    print_settings_dqpb(benchmark, inst, n, N, qpb_maxFW, qpb_tol,
                        m, M, BLOCK_SIZE, lb, ub, initUB);

    var best: int = initUB;

    // ---- Step 1: initial CPU BFS (no bounds) to D*m*numLocales nodes ----
    timer.start();
    var root = new Node_QPB(n);
    var pool = new SinglePool_par(Node_QPB);
    pool.pushBackFree(root);

    while (pool.size < D*m*numLocales) {
      var hasWork = 0;
      var parent = pool.popFrontFree(hasWork);
      if !hasWork then break;
      decompose_nobound_dist(parent, priority_fac, priority_loc,
                             exploredTree, pool, n, N);
    }

    timer.stop();
    const res1 = (timer.elapsed() - res0, exploredTree, exploredSol);

    // ---- Step 2: split pool across locales, then across GPUs per locale ----
    timer.start();

    var eachLocaleExploredTree, eachLocaleExploredSol: [PrivateSpace] uint = noinit;
    var eachLocaleBest: [PrivateSpace] int = noinit;
    var eachLocaleState: [PrivateSpace] atomic bool = BUSY;
    var allLocalesIdleFlag: atomic bool = false;

    const poolSize = pool.size;
    const c = poolSize / numLocales;
    const l = poolSize - (numLocales-1)*c;
    const f = pool.front;

    pool.front = 0;
    pool.size = 0;

    // distMultiPool[locID][gpuID] is the per-GPU pool. The outer index is
    // distributed across locales (PrivateSpace), the inner index is local.
    // Cross-locale work stealing reaches into distMultiPool[victimLocaleID]
    // directly; Chapel's distributed array semantics route the access.
    var distMultiPool: [PrivateSpace][0..#D] SinglePool_par(Node_QPB);

    coforall (locID, loc) in zip(0..#numLocales, Locales) with (ref pool,
      ref eachLocaleExploredTree, ref eachLocaleExploredSol, ref eachLocaleBest,
      ref eachLocaleState, ref distMultiPool) do on loc {

      var eachExploredTree, eachExploredSol: [0..#D] uint = noinit;
      var eachBest: [0..#D] int = noinit;
      var eachTaskState: [0..#D] atomic bool = BUSY;
      var allTasksIdleFlag: atomic bool = false;

      var pool_lloc = new SinglePool_par(Node_QPB);

      // Each locale gets its chunk (stride-numLocales split; last locale
      // also receives the tail).
      pool_lloc.elements[0..#c] = pool.elements[locID+f.. by numLocales #c];
      pool_lloc.size += c;
      if (locID == numLocales-1) {
        pool_lloc.elements[c..#(l-c)] = pool.elements[(numLocales*c)+f..#(l-c)];
        pool_lloc.size += l-c;
      }

      const poolSize_l = pool_lloc.size;
      const c_l = poolSize_l / D;
      const l_l = poolSize_l - (D-1)*c_l;
      const f_l = pool_lloc.front;

      pool_lloc.front = 0;
      pool_lloc.size = 0;

      ref multiPool = distMultiPool[locID];

      coforall gpuID in 0..#D with (ref pool, ref eachExploredTree, ref eachExploredSol,
        ref eachBest, ref eachTaskState, ref multiPool) {

        const device = here.gpus[gpuID];

        var tree, sol: uint;
        ref pool_loc = multiPool[gpuID];
        var best_l = best;
        var taskState, locState: bool = BUSY;

        // Each GPU task gets its chunk of the locale's slice.
        pool_loc.elements[0..#c_l] = pool_lloc.elements[gpuID+f_l.. by D #c_l];
        pool_loc.size += c_l;
        if (gpuID == D-1) {
          pool_loc.elements[c_l..#(l_l-c_l)] = pool_lloc.elements[(D*c_l)+f_l..#(l_l-c_l)];
          pool_loc.size += l_l-c_l;
        }

        // Host-side per-batch buffers (per-task).
        var inputs:   [0..#M] Node_QPB_In;
        var outputs:  [0..#M] Node_QPB_Out;
        var bounds:   [0..#M] int;
        var parent_first_h: [0..#M] int(32);

        var freeSlots: SlotStack_dist;
        freeSlots.init_with_capacity(qpb_xPoolSize);
        var deferred_free: SlotList_dist;

        // Device-side per-batch buffers (on this task's GPU).
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
          const numChildren = prepareChildren_dist(
              inputs, parent_first_h, parent_count,
              freeSlots, deferred_free,
              priority_fac, priority_loc, F, DD,
              pool_loc, best_l, tree, sol, n, N);

          if (numChildren > 0) {
            if (taskState == IDLE) {
              taskState = BUSY;
              eachTaskState[gpuID].write(BUSY);
            }
            if (locState == IDLE) {
              locState = BUSY;
              eachLocaleState[locID].write(BUSY);
            }

            inputs_d = inputs;
            if (parent_count > 0) then
              parent_first_d = parent_first_h;

            if (parent_count > 0) then
              on device do parent_eigen_setup_gpu_dist(
                  inputs_d, parent_first_d, parent_count,
                  F_d, DD_d, queue_fac_d,
                  parentW_d, parentSigma_d, parentVW_d,
                  parentFnormFa_d, parentSingFa_d, parent_C_d,
                  n, N);

            on device do evaluate_qpb_gpu_dist(
                inputs_d, outputs_d, bounds_d, numChildren,
                qpbXPool_d,
                parentW_d, parentSigma_d, parentVW_d,
                parentFnormFa_d, parentSingFa_d, parent_C_d,
                F_d, DD_d, queue_fac_d,
                best_l, n, N, qpb_maxFW, qpb_tol, qpb_sinkIter);

            outputs = outputs_d;
            bounds  = bounds_d;

            generate_children_dist(inputs, outputs, bounds, numChildren,
                                   freeSlots, pool_loc, best_l, n);

            for k in 0..<deferred_free.size do
              freeSlots.push(deferred_free.elements[k]);
            deferred_free.clear();
          } else {
            // Work stealing -- first local (other GPUs on this locale),
            // then global (GPU pools on remote locales).
            var localSteal  = false;
            var globalSteal = false;

            // ---- Local steal (intra-locale, between GPUs). ----
            const victimTasks = permute(0..#D);

            label WS0 for i in 0..#D {
              const victimTaskID = victimTasks[i];
              if (victimTaskID == gpuID) then continue;

              ref victim = multiPool[victimTaskID];
              var nn = 0;

              label WS1 while (nn < 10) {
                if victim.lock.compareAndSwap(false, true) {
                  const size = victim.size;
                  if (size >= 2*m) {
                    var (hasWork, p) = victim.popFrontBulkFree(m, M);
                    if (hasWork == 0) {
                      victim.lock.write(false);
                      halt("DEADCODE in QPB distributed local work stealing");
                    }

                    // Stolen nodes lose their warm-start; victim pool slots
                    // are leaked (see module header).
                    for k in 0..<hasWork {
                      if p[k].hasBoundData {
                        p[k].hasBoundData = false;
                        p[k].qpbXSlot     = -1;
                      }
                    }

                    pool_loc.pushBackBulk(p);
                    localSteal = true;
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

            // ---- Global steal (inter-locale, between GPU pools). ----
            if (localSteal == false && numLocales != 1) {
              const victimLocales = permute(0..#numLocales);

              label WS00 for i in 0..#numLocales {
                const victimLocaleID = victimLocales[i];
                if (victimLocaleID == locID) then continue;

                ref victimMultiPool = distMultiPool[victimLocaleID];
                const victimTasks2 = permute(0..#D);

                for j in 0..#D {
                  const victimTaskID = victimTasks2[j];
                  ref victim = victimMultiPool[victimTaskID];
                  var nn = 0;

                  label WS11 while (nn < 10) {
                    if victim.lock.compareAndSwap(false, true) {
                      const size = victim.size;
                      if (size >= 2*m) {
                        var (hasWork, p) = victim.popFrontBulkFree(m, M);
                        if (hasWork == 0) {
                          victim.lock.write(false);
                          halt("DEADCODE in QPB distributed global work stealing");
                        }

                        // Same invalidation as local steal; cross-locale
                        // qpbXSlots are meaningless on the thief's GPU.
                        for k in 0..<hasWork {
                          if p[k].hasBoundData {
                            p[k].hasBoundData = false;
                            p[k].qpbXSlot     = -1;
                          }
                        }

                        pool_loc.pushBackBulk(p);
                        globalSteal = true;
                      }
                      victim.lock.write(false);
                      break WS00;
                    }
                    nn += 1;
                    currentTask.yieldExecution();
                  }
                }
              }
            }

            if (localSteal == false && globalSteal == false) {
              // Termination -- propagate IDLE up the task / locale chain.
              if (taskState == BUSY) {
                taskState = IDLE;
                eachTaskState[gpuID].write(IDLE);
              }
              if allIdle(eachTaskState, allTasksIdleFlag) {
                if (locState == BUSY) {
                  locState = IDLE;
                  eachLocaleState[locID].write(IDLE);
                }
                if allIdle(eachLocaleState, allLocalesIdleFlag) {
                  break;
                }
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

      eachLocaleExploredTree[locID] = (+ reduce eachExploredTree);
      eachLocaleExploredSol[locID]  = (+ reduce eachExploredSol);
      eachLocaleBest[locID]         = (min reduce eachBest);
    }

    exploredTree += (+ reduce eachLocaleExploredTree);
    exploredSol  += (+ reduce eachLocaleExploredSol);
    best          = (min reduce eachLocaleBest);

    timer.stop();
    const res2 = (timer.elapsed(), exploredTree, exploredSol) - res1;

    // Per-locale workload share (% of GPU-loop nodes explored per locale).
    // Printed between the settings and results blocks (see search_distributed_qpb).
    const workloadLocaleArr =
        100.0*eachLocaleExploredTree/(exploredTree-res1[1]):real;

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
      decompose_nobound_dist(parent, priority_fac, priority_loc,
                             exploredTree, pool, n, N);
    }

    timer.stop();
    elapsedTime = timer.elapsed();
    const res3 = (elapsedTime, exploredTree, exploredSol) - res1 - res2;

    optimum = best;
    writeln("\nExploration terminated.");
    writeln("Workload per locale: ", workloadLocaleArr);
  }

  proc search_distributed_qpb()
  {
    writeln("Distributed multi-GPU execution mode (QPB) with ", numLocales,
            " locales and ", D, " GPUs each");

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

  // QPB distributed-multi-GPU settings printer. Labels are padded to 29
  // chars (values start at column 30) so the settings block prints as an
  // aligned table.
  proc print_settings_dqpb(const benchmark: string, const inst: string,
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
    writeln("Number of locales:           ", numLocales);
    writeln("Number of GPUs per locale:   ", D);
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
