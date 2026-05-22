module qap_search_gpu_qpb
{
  /*
    Single-GPU B&B for QAP using the QPB lower-bound operator.

    Algorithm pipeline:

      1. Preprocessing (priorities, heuristic UB) -- shared with GLB.
      2. Initial CPU BFS to populate the pool. No bounds computed (there is
         no CPU QPB); we just expand until pool.size >= m.
      3. Main GPU loop:
         - prepareChildren: pop parents, generate children with variable-
           fixing against parent's stored reduced costs, allocate pool slots
           for each child's qpbX, decide which parents are "warm-startable"
           and allocate a per-batch eigen-cache slot for each.
         - parent_eigen_setup_gpu (per surviving parent): build the A-side
           eigendecomp (W/sigma/VW), the asym A-side cache (fnormFa/singFa),
           and the parent's cross-cost matrix C, all reused across siblings.
         - evaluate_qpb_gpu (per child): full QPB bound. Reads parent caches
           and warm-start X; writes integer bound + Node_QPB_Out payload +
           the child's primal X into its assigned qpbXPool slot.
         - generate_children: push survivors back into the pool; reclaim
           pool slots of pruned children. Reclaim parent slots that were
           deferred while their batch's kernel was in flight.
      4. Final CPU DFS to drain leaves (no further GPU dispatches).

    All GPU kernels are written as @assertOnGpu foreach blocks. Per-thread
    working set lives in `Problem_qpb` procedures via local tuples; the
    persistent qpbXPool and per-batch caches are managed here.
  */

  use Time;
  use Math;
  use GpuDiagnostics;

  use util;
  use Pool;
  use QAP_node;
  use Util_qap;
  use Problem_qap;
  use Problem_qpb;

  import main_qap.m as m;
  import main_qap.M as M;

  import main_qap.inst as inst;
  import main_qap.lb as lb;
  import main_qap.ub as ub;
  import main_qap.qpb_xPoolSize as qpb_xPoolSize;
  import main_qap.qpb_maxFW    as qpb_maxFW;
  import main_qap.qpb_tol      as qpb_tol;
  import main_qap.qpb_sinkIter as qpb_sinkIter;
  import main_qap.qpb_profile  as qpb_profile;

  config param sizeMax: int(32) = 32;

  config const BLOCK_SIZE = 32;

  var benchmark: string;
  var n, N: int(32);
  var initUB: int;

  /******************** CPU-side helpers ***************************************/

  // CPU-side expansion (no bound). Used by the initial BFS to populate the
  // pool with children that the first GPU batch will bound from scratch.
  proc decompose_nobound(const ref parent: Node_QPB,
                         const ref priority_fac: [] int(32),
                         const ref priority_loc: [] int(32),
                         ref tree_loc: uint,
                         ref pool: SinglePool(Node_QPB))
  {
    const depth = parent.depth;
    if (depth == n) then return; // leaves are handled inline elsewhere

    const i = priority_fac[depth];
    for j0 in 0..<N by -1 {
      const j = priority_loc[j0];
      if !parent.available[j] then continue;

      var child: Node_QPB;
      child.mapping  = parent.mapping;
      child.depth    = depth + 1;
      child.available = parent.available;
      child.mapping[i]   = j:int(8);
      child.available[j] = false;
      child.hasBoundData = false;
      child.qpbXSlot     = -1;
      child.qpbBoundContLast = 0.0;
      child.qpbFixedCost = 0;
      child.lb = 0;
      pool.pushBack(child);
      tree_loc += 1;
    }
  }

  /*
    prepareChildren: pop parents off the back of the pool, generate children
    (with QPB variable-fixing against the parent's stored reduced costs),
    write each child's Node_QPB_In into `inputs`, and assign each child a
    fresh slot in the persistent qpbXPool free list.

    Leaves are evaluated inline (no GPU dispatch). Parents are queued for
    deferred-free of their qpbXSlot if at least one warm-startable child was
    emitted (in which case the kernel will read their slot as warm-start).

    Returns the number of children written; sets `parent_count` to the number
    of eigen-cache slots assigned.
  */
  proc prepareChildren(ref inputs: [] Node_QPB_In,
                       ref parent_first_child_idx: [] int(32),
                       ref parent_count: int(32),
                       ref freeSlots: SlotStack,
                       ref deferred_free: SlotList,
                       const ref queue_fac: [] int(32),
                       const ref queue_loc: [] int(32),
                       const ref F: [] int(32),
                       const ref DD: [] int(32),
                       ref pool: SinglePool(Node_QPB),
                       ref UB: int,
                       ref exploredTree: uint,
                       ref exploredSol: uint): int(32)
  {
    var size: int(32) = 0;
    parent_count = 0;

    while (true) {
      // Stop conditions: kernel-input buffer would overflow, or we'd run
      // out of qpbXPool slots in the next parent. Both must reserve N slots
      // because that's the max children one parent can produce.
      if (size + N > M)            then break;
      if (freeSlots.count() < N)   then break;

      var hasWork = 0;
      var parent = pool.popBack(hasWork);
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

      // Precompute parent's unassigned-list positions of the branching fac
      // and the NEXT branching fac (needed for the child's nextBranchFacIdx).
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

        // ---- QPB variable fixing using parent's stored reduced costs. ----
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

        // ---- Leaf: evaluate inline, don't dispatch. ----
        if (dp:int(32) + 1 == n) {
          var leaf_map: sizeMax*int(8) = parent.mapping;
          leaf_map[i] = j:int(8);
          const cost = ObjectiveFunction(leaf_map, DD, F, n, N);
          if (cost < UB) then UB = cost;
          exploredSol += 1;
          exploredTree += 1;
          continue;
        }

        // ---- Allocate child's output qpbX pool slot. ----
        const child_out_slot = freeSlots.pop();

        // ---- Write Node_QPB_In ----
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

      // Parent's qpbX slot is reclaimable. If at least one warm-startable
      // child was emitted the kernel will read it -- defer to post-drain.
      // Otherwise return it now.
      if (parent.hasBoundData && parent.qpbXSlot >= 0) {
        if (parent_eigen_slot >= 0) then deferred_free.push(parent.qpbXSlot);
        else                            freeSlots.push(parent.qpbXSlot);
      }
    }

    return size;
  }

  /*
    generate_children: after the kernel drains, combine inputs + outputs +
    bounds into a Node_QPB for each survivor and push back into the pool.
    Pruned children's qpbX slots are reclaimed.
  */
  proc generate_children(const ref inputs:  [] Node_QPB_In,
                         const ref outputs: [] Node_QPB_Out,
                         const ref bounds:  [] int,
                         const size: int,
                         ref freeSlots: SlotStack,
                         ref pool: SinglePool(Node_QPB),
                         const UB: int)
  {
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
        pool.pushBack(child);
      } else {
        if (ip.outSlot >= 0) then freeSlots.push(ip.outSlot);
      }
    }
  }

  /******************** Slot-allocator helper records **************************/

  /*
    SlotStack: LIFO pool of free qpbXPool slot indices. Backed by an array
    initialized to [0..#qpbXPoolSize] and a top-of-stack counter. push() /
    pop() / count() are O(1).
  */
  record SlotStack {
    var dom: domain(1);
    var elements: [dom] int(32);
    var top: int;  // next free index (one past the top)

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
    SlotList: simple append-only buffer for deferring parent slot releases
    until the current batch's kernel has drained. Cleared on each iteration
    after we reclaim its contents.
  */
  record SlotList {
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

  /******************** GPU kernel wrappers ************************************/

  // Per-parent eigen + C-matrix setup kernel. One thread per parent slot.
  // `@gpu.blockSize(32)` is required: the per-thread working set inside
  // parent_eigen_setup_proc (several sizeMaxSq float tuples for A_hat /
  // tmpM / Fa / FtF / evec_F plus the Helmert prefix-sum scratch) is too
  // large to leave headroom for the runtime's default block size, which
  // can otherwise hit a "too many resources requested for launch" failure
  // at kernel dispatch.
  proc parent_eigen_setup_gpu(const ref inputs_d:       [] Node_QPB_In,
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

  // Per-child QPB bound kernel. One thread per child. Same `@gpu.blockSize`
  // rationale as parent_eigen_setup_gpu; the per-thread working set here is
  // even larger (~11 sizeMaxSq float matrices + LAP scratch).
  proc evaluate_qpb_gpu(const ref inputs_d:  [] Node_QPB_In,
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
    const device = here.gpus[0];
    var timer: stopwatch;

    // read instance
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
    print_settings_qpb(benchmark, inst, n, N, qpb_maxFW, qpb_tol,
                       m, M, BLOCK_SIZE, lb, ub, initUB);

    var best: int = initUB;

    // ---- Step 1: initial CPU BFS (no bounds) ----
    timer.start();
    var root = new Node_QPB(n);
    var pool = new SinglePool(Node_QPB);
    pool.pushBack(root);

    while (pool.size < m) {
      var hasWork = 0;
      var parent = pool.popFront(hasWork);
      if !hasWork then break;
      decompose_nobound(parent, priority_fac, priority_loc, exploredTree, pool);
    }

    timer.stop();
    const res1 = (timer.elapsed() - res0, exploredTree, exploredSol);

    // ---- Step 2: main GPU loop ----
    timer.start();

    // Host-side per-batch buffers.
    var inputs:   [0..#M] Node_QPB_In;
    var outputs:  [0..#M] Node_QPB_Out;
    var bounds:   [0..#M] int;
    var parent_first_h: [0..#M] int(32);

    // Persistent qpbXPool free-list (host-side allocator).
    var freeSlots: SlotStack;
    freeSlots.init_with_capacity(qpb_xPoolSize);
    var deferred_free: SlotList;

    // Device-side per-batch buffers. Declared with `on device var` so that
    // their storage lives on the GPU; we assign host arrays to them for H2D.
    on device var inputs_d:   [0..#M] Node_QPB_In;
    on device var outputs_d:  [0..#M] Node_QPB_Out;
    on device var bounds_d:   [0..#M] int;
    on device var parent_first_d: [0..#M] int(32);

    // Device-side per-batch eigen / C caches (stride sizeMax per slot).
    on device var parentW_d:       [0..#(M*sizeMaxSq)] real(32);
    on device var parentSigma_d:   [0..#(M*sizeMax)]   real(64);
    on device var parentVW_d:      [0..#(M*sizeMaxSq)] real(32);
    on device var parentFnormFa_d: [0..#M]             real(64);
    on device var parentSingFa_d:  [0..#(M*sizeMax)]   real(64);
    on device var parent_C_d:      [0..#(M*sizeMaxSq)] real(32);

    // Persistent device-side qpbX pool.
    on device var qpbXPool_d: [0..#(qpb_xPoolSize*sizeMaxSq)] real(32);

    // Read-only problem data + ordering queue, copied once to the device.
    on device const F_d  = F;
    on device const DD_d = DD;
    on device const queue_fac_d = priority_fac;

    // ---- Per-phase profile state (active only when qpb_profile=true). ----
    // Six stopwatches accumulate wall time across the main-loop phases;
    // counters track iteration count and mean / max batch and parent-slot
    // sizes. All updates are gated on qpb_profile.
    var prep_sw, h2d_sw, eigen_sw, qpb_sw, d2h_sw, gen_sw: stopwatch;
    var iter_count: int    = 0;
    var nc_sum:     int    = 0;
    var nc_max:     int    = 0;
    var pc_sum:     int    = 0;
    var pc_max:     int    = 0;
    var eigen_iter: int    = 0;   // iterations where parent_count > 0

    while true {
      var parent_count: int(32) = 0;
      if qpb_profile then prep_sw.start();
      const numChildren = prepareChildren(
          inputs, parent_first_h, parent_count,
          freeSlots, deferred_free,
          priority_fac, priority_loc, F, DD,
          pool, best, exploredTree, exploredSol);
      if qpb_profile then prep_sw.stop();

      if (numChildren > 0) {
        if qpb_profile {
          iter_count += 1;
          nc_sum += numChildren;
          if (numChildren > nc_max) then nc_max = numChildren;
          pc_sum += parent_count;
          if (parent_count > pc_max) then pc_max = parent_count;
          if (parent_count > 0)      then eigen_iter += 1;
        }

        // H2D: ship the input record buffer and the parent-first-child index
        // buffer in bulk. Only the first `numChildren` / `parent_count`
        // entries are valid; uploading the full M-sized buffer lets the
        // Chapel runtime issue one contiguous copy per array.
        if qpb_profile then h2d_sw.start();
        inputs_d = inputs;
        if (parent_count > 0) then
          parent_first_d = parent_first_h;
        if qpb_profile then h2d_sw.stop();

        // Per-parent setup kernel.
        if (parent_count > 0) {
          if qpb_profile then eigen_sw.start();
          on device do parent_eigen_setup_gpu(
              inputs_d, parent_first_d, parent_count,
              F_d, DD_d, queue_fac_d,
              parentW_d, parentSigma_d, parentVW_d,
              parentFnormFa_d, parentSingFa_d, parent_C_d);
          if qpb_profile then eigen_sw.stop();
        }

        // Main per-child bound kernel.
        if qpb_profile then qpb_sw.start();
        on device do evaluate_qpb_gpu(
            inputs_d, outputs_d, bounds_d, numChildren,
            qpbXPool_d,
            parentW_d, parentSigma_d, parentVW_d,
            parentFnormFa_d, parentSingFa_d, parent_C_d,
            F_d, DD_d, queue_fac_d,
            best, qpb_maxFW, qpb_tol, qpb_sinkIter);
        if qpb_profile then qpb_sw.stop();

        // D2H of outputs and bounds; qpbX stays in qpbXPool_d.
        if qpb_profile then d2h_sw.start();
        outputs = outputs_d;
        bounds  = bounds_d;
        if qpb_profile then d2h_sw.stop();

        // Filter survivors back into the pool.
        if qpb_profile then gen_sw.start();
        generate_children(inputs, outputs, bounds, numChildren,
                          freeSlots, pool, best);

        // Reclaim the qpbX slots of parents whose batch has now drained.
        for k in 0..<deferred_free.size do
          freeSlots.push(deferred_free.elements[k]);
        deferred_free.clear();
        if qpb_profile then gen_sw.stop();
      } else {
        break;
      }
    }

    timer.stop();
    const res2 = (timer.elapsed(), exploredTree, exploredSol) - res1;

    // ---- Step 3: final CPU DFS drain (no bound) ----
    timer.start();
    while true {
      var hasWork = 0;
      var parent = pool.popBack(hasWork);
      if !hasWork then break;

      const depth = parent.depth;
      if (depth == n) {
        const cost = ObjectiveFunction(parent.mapping, DD, F, n, N);
        if (cost < best) then best = cost;
        exploredSol += 1;
        continue;
      }
      decompose_nobound(parent, priority_fac, priority_loc, exploredTree, pool);
    }

    timer.stop();
    elapsedTime = timer.elapsed();
    const res3 = (elapsedTime, exploredTree, exploredSol) - res1 - res2;

    // ---- Optional per-phase profile summary (--qpb_profile=true). ----
    if qpb_profile {
      const total_gpu = res2[0];
      const inv_total = if total_gpu > 0.0 then 100.0 / total_gpu else 0.0;
      writeln();
      writeln("=================================================");
      writeln("QPB Profile (single-GPU)");
      writeln("=================================================");
      writef("%<31s %10.4r s\n",                "Preprocessing:",            res0);
      writef("%<31s %10.4r s\n",                "CPU initial BFS:",          res1[0]);
      writef("%<31s %10.4r s\n",                "GPU main loop:",            total_gpu);
      writef("   %<28s %10.4r s  (%5.2r %%)\n", "prepareChildren:",          prep_sw.elapsed(),  inv_total * prep_sw.elapsed());
      writef("   %<28s %10.4r s  (%5.2r %%)\n", "H2D inputs/parent_first:",  h2d_sw.elapsed(),   inv_total * h2d_sw.elapsed());
      writef("   %<28s %10.4r s  (%5.2r %%)\n", "parent_eigen_setup kernel:",eigen_sw.elapsed(), inv_total * eigen_sw.elapsed());
      writef("   %<28s %10.4r s  (%5.2r %%)\n", "evaluate_qpb kernel:",      qpb_sw.elapsed(),   inv_total * qpb_sw.elapsed());
      writef("   %<28s %10.4r s  (%5.2r %%)\n", "D2H outputs/bounds:",       d2h_sw.elapsed(),   inv_total * d2h_sw.elapsed());
      writef("   %<28s %10.4r s  (%5.2r %%)\n", "generate_children:",        gen_sw.elapsed(),   inv_total * gen_sw.elapsed());
      writef("%<31s %10.4r s\n",                "CPU final DFS drain:",      res3[0]);
      writeln("-------------------------------------------------");
      writeln("Iterations: ", iter_count, "  (", eigen_iter, " ran parent_eigen_setup)");
      const mean_nc = if iter_count > 0 then nc_sum / iter_count else 0;
      const mean_pc = if iter_count > 0 then pc_sum / iter_count else 0;
      writeln("Batch size  (numChildren): mean ", mean_nc, "  max ", nc_max);
      writeln("Parent slot (parent_cnt):  mean ", mean_pc, "  max ", pc_max);
      writeln("=================================================");
    }

    optimum = best;
    writeln("\nExploration terminated.");
  }

  proc search_gpu_qpb()
  {
    writeln("Single-GPU execution mode (QPB)");

    var optimum: int;
    var exploredTree: uint = 0;
    var exploredSol:  uint = 0;
    var elapsedTime:  real;

    startGpuDiagnostics();

    qap_search(optimum, exploredTree, exploredSol, elapsedTime);

    stopGpuDiagnostics();

    print_results(optimum, exploredTree, exploredSol, elapsedTime, initUB);

    if qpb_profile {
      writeln("GPU diagnostics:");
      writeln("   kernel_launch:   ", getGpuDiagnostics().kernel_launch);
      writeln("   host_to_device:  ", getGpuDiagnostics().host_to_device);
      writeln("   device_to_host:  ", getGpuDiagnostics().device_to_host);
      writeln("   device_to_device:", getGpuDiagnostics().device_to_device);
    }

    return 0;
  }

  // QPB settings printer. Labels are padded to 29 chars (values start at
  // column 30) so the settings block prints as an aligned table.
  proc print_settings_qpb(const benchmark: string, const inst: string,
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
