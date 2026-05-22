/*******************************************************************************
  QPB (Quadratic Programming Bound) lower bound for the GPU-batched B&B.

  One Chapel "thread" (one iteration of the @assertOnGpu foreach in the search
  module) bounds one child node; everything below runs in thread-local storage
  with mixed precision (real(32) matrix storage, real(64) accumulators).

  GPU entry-point procedures:
    bound_QPB_kernel        -- per-child bound; reads per-batch eigen / C
                               caches and the persistent qpbX pool, writes the
                               integer bound + the Node_QPB_Out payload + the
                               child's primal X into its assigned pool slot.
    parent_eigen_setup_proc -- per-parent setup; one call per parent slot in
                               the batch. Computes A-side eigendecomp (W,
                               sigma, V*W), the asymmetry-correction cache
                               (fnormFa, singFa), and the parent's C cross-
                               cost matrix that every sibling reuses.

  See Problem_qap.chpl for the GLB / RLT1 counterparts.
*******************************************************************************/

module Problem_qpb
{
  use Math;

  use Util_qap;
  use QAP_node;

  /*****************************************************************************
    Compile-time bounds and numerical tolerances.

    The values below are calibrated for the mixed-precision policy (real(32)
    matrix storage, real(64) accumulators). Tightening any of them without
    instance-aware analysis can silently prune optimum-bearing subtrees, so
    treat them as a floor.
  *****************************************************************************/

  config param sizeMax: int(32) = 32;
  param sizeMaxSq: int(32) = sizeMax * sizeMax;

  // Kernel-side tolerances. BIG_DBL is the LAP "infinity" sentinel; the rest
  // gate convergence inside FW, Jacobi, Sinkhorn, and the integer-bound floor.
  param BIG_DBL:          real(64) = 1.0e30;
  param SK_EPS:           real(64) = 1.0e-10;
  param Q_CURV_EPS:       real(64) = 1.0e-9;
  param ASYM_EPS:         real(64) = 1.0e-6;
  param JAC_OFF_TOL:      real(64) = 1.0e-10;
  param JAC_MAX_SWEEPS:   int      = 60;
  param QPB_VF_TOL_BASE:  real(64) = 1.0e-6;
  param QPB_VF_TOL_REL:   real(64) = 1.0e-5;

  // Host-side variable-fixing tolerances. Looser than the kernel-side floor
  // because the host's estimate
  //   est = qpbFixedCost + qpbBoundContLast + reducedCost
  // accumulates more drift than the kernel's `corrected` value (FW + Sinkhorn
  // + LAP warm-start drift on top of the per-cell FP32 cast roundoff).
  param HOST_QPB_VF_TOL_BASE: real(64) = 1.0;
  param HOST_QPB_VF_TOL_REL:  real(64) = 1.0e-6;

  /*****************************************************************************
    Per-batch transfer records.

    Node_QPB is split into kernel-input (Node_QPB_In) and kernel-output
    (Node_QPB_Out) so each batch ships a small input record H2D per child
    and returns a small output record D2H per child. The primal X never
    round-trips: it stays in the persistent device-side qpbX pool.
  *****************************************************************************/

  record Node_QPB_In
  {
    var mapping:    sizeMax*int(8);
    var available:  sizeMax*bool;
    var depth:      int(8);
    var parentIdxFac:     int(8);   // pos of branching_fac in parent's unassigned_fac list
    var parentIdxLoc:     int(8);   // pos of branching_loc in parent's unassigned_loc list
    var nextBranchFacIdx: int(8);   // row of dual reduced-costs the kernel emits, -1 if no further VF
    var parentEigenIdx:   int(32);  // slot in per-batch eigen cache, -1 if not warm-startable
    var parentXSlot:      int(32);  // slot in persistent qpbXPool, -1 if cold-start
    var outSlot:          int(32);  // slot the kernel writes this child's X into
  }

  record Node_QPB_Out
  {
    var qpbReducedCostsRow: sizeMax*real(32);
    var qpbBoundContLast:   real(64);
    var qpbFixedCost:       int;
  }

  /*****************************************************************************
    Helpers shared by setup and main kernels.

    These are written so the Chapel compiler inlines them inside the @assertOnGpu
    foreach. The big matrices are passed as `ref` tuples; small scratch arrays
    are local tuples declared inside the proc.
  *****************************************************************************/

  /*
    Serial dense matmul, C = A * B with row strides sA / sB / sC. FP64 inner
    accumulator, FP32 stores. The k-loop is intentionally not hand-unrolled;
    the backend's adaptive unroll handles a variable trip count better than
    a fixed factor.
  */
  inline proc matmul(const ref A: (sizeMax**2)*real(32),
                     const ref B: (sizeMax**2)*real(32),
                     ref C:       (sizeMax**2)*real(32),
                     m: int(32), k: int(32), p: int(32),
                     sA: int(32), sB: int(32), sC: int(32))
  {
    for i in 0..<m {
      for j in 0..<p {
        var sum: real(64) = 0.0;
        for l in 0..<k {
          sum += A[i*sA + l]:real(64) * B[l*sB + j]:real(64);
        }
        C[i*sC + j] = sum:real(32);
      }
    }
  }

  /*
    Transposed-A matmul: C = A^T * B. Shapes are (m, k) = A^T means A is
    stored as (k, m) and we want the (l, i) -> i*sA + l layout flipped to
    A[l*sA + i] for the read.
  */
  inline proc matmul_TN(const ref A: (sizeMax**2)*real(32),
                        const ref B: (sizeMax**2)*real(32),
                        ref C:       (sizeMax**2)*real(32),
                        m: int(32), k: int(32), p: int(32),
                        sA: int(32), sB: int(32), sC: int(32))
  {
    for i in 0..<m {
      for j in 0..<p {
        var sum: real(64) = 0.0;
        for l in 0..<k {
          sum += A[l*sA + i]:real(64) * B[l*sB + j]:real(64);
        }
        C[i*sC + j] = sum:real(32);
      }
    }
  }

  /*
    helmert_VTAV: out = V^T * M * V, where V is the (m x (m-1)) Helmert basis.
    Avoids materializing V by using a streaming 2-row prefix-sum on M.
    Coefficients c_j = 1/sqrt((j+1)(j+2)), d_j = (j+1) c_j.

    `M_stride` lets the caller pass A (stride m) or a stride-sizeMax cache slot.
  */
  inline proc helmert_VTAV_dev(const ref M: (sizeMax**2)*real(32),
                               m: int(32), M_stride: int(32),
                               ref outM: (sizeMax**2)*real(32),
                               out_stride: int(32))
  {
    const mm = m - 1;
    if (mm <= 0) then return;

    // Precompute c, d (length mm).
    var c: sizeMax*real(64);
    var d: sizeMax*real(64);
    for j in 0..<mm {
      const inv_norm = 1.0 / sqrt((j+1):real(64) * (j+2):real(64));
      c[j] = inv_norm;
      d[j] = (j+1):real(64) * inv_norm;
    }

    // P_prev / P_curr each hold one row of the 2D prefix sum
    //   P(k, l) = sum_{i<=k, j<=l} M[i, j]
    // sliding k from 0 .. mm-1.
    var P_prev: sizeMax*real(64);
    var P_curr: sizeMax*real(64);

    {
      var running: real(64) = 0.0;
      for l in 0..<m {
        running += M[0*M_stride + l]:real(64);
        P_prev[l] = running;
      }
    }

    for k in 0..<mm {
      // P_curr[l] = P_prev[l] + cumsum_{j<=l} M[k+1, j]
      var running: real(64) = 0.0;
      for l in 0..<m {
        running += M[(k+1)*M_stride + l]:real(64);
        P_curr[l] = P_prev[l] + running;
      }

      const c_k = c[k];
      const d_k = d[k];

      for l in 0..<mm {
        const c_l = c[l];
        const d_l = d[l];
        const rect    = P_prev[l];
        const col_lp1 = P_prev[l + 1] - rect;
        const row_kp1 = P_curr[l]     - rect;
        const corner  = M[(k+1)*M_stride + (l+1)]:real(64);

        const val = c_k * c_l * rect
                  - c_k * d_l * col_lp1
                  - d_k * c_l * row_kp1
                  + d_k * d_l * corner;
        outM[k*out_stride + l] = val:real(32);
      }

      // slide P_prev <- P_curr
      for l in 0..<m do P_prev[l] = P_curr[l];
    }
  }

  /*
    helmert_VA: out = V * U, V the m x (m-1) Helmert basis, U is (m-1) x (m-1).
    Walks down rows updating a suffix sum T(i, *) = sum_{k>=i} c[k] * U[k, *].
  */
  inline proc helmert_VA_dev(const ref U: (sizeMax**2)*real(32),
                             m: int(32), U_stride: int(32),
                             ref outM: (sizeMax**2)*real(32),
                             out_stride: int(32))
  {
    const mm = m - 1;
    if (mm <= 0) then return;

    var c: sizeMax*real(64);
    var d: sizeMax*real(64);
    for j in 0..<mm {
      const inv_norm = 1.0 / sqrt((j+1):real(64) * (j+2):real(64));
      c[j] = inv_norm;
      d[j] = (j+1):real(64) * inv_norm;
    }

    // T(0, l) = sum_{k=0..mm-1} c[k] * U[k, l]
    var T: sizeMax*real(64);
    for l in 0..<mm {
      var sum: real(64) = 0.0;
      for k in 0..<mm {
        sum += c[k] * U[k*U_stride + l]:real(64);
      }
      T[l] = sum;
    }

    // out[0, l] = T(0, l)
    for l in 0..<mm do
      outM[0*out_stride + l] = T[l]:real(32);

    // For i = 1..m-1:
    //   out[i, l] = T(i, l) - d_{i-1} * U[i-1, l]
    //   T(i, l)   = T(i-1, l) - c_{i-1} * U[i-1, l]
    for i in 1..<m {
      const c_prev = c[i - 1];
      const d_prev = d[i - 1];
      for l in 0..<mm {
        const u = U[(i-1)*U_stride + l]:real(64);
        T[l] -= c_prev * u;
        outM[i*out_stride + l] = (T[l] - d_prev * u):real(32);
      }
    }
  }

  /*
    Sinkhorn-Knopp normalization. Clamps negatives, then alternates row /
    column normalization for `maxIter` passes. Accumulates row/col sums in
    double, stores back in float.
  */
  inline proc sinkhorn_normalize(ref X: (sizeMax**2)*real(32), n: int(32), param maxIter: int)
  {
    const cells = n * n;
    for idx in 0..<cells {
      if (X[idx] < 0.0:real(32)) then X[idx] = 0.0:real(32);
    }
    for it in 0..<maxIter {
      for i in 0..<n {
        var rs: real(64) = 0.0;
        for j in 0..<n do rs += X[i*n + j]:real(64);
        if (rs > SK_EPS) {
          const inv: real(32) = (1.0 / rs):real(32);
          for j in 0..<n do X[i*n + j] *= inv;
        }
      }
      for j in 0..<n {
        var cs: real(64) = 0.0;
        for i in 0..<n do cs += X[i*n + j]:real(64);
        if (cs > SK_EPS) {
          const inv: real(32) = (1.0 / cs):real(32);
          for i in 0..<n do X[i*n + j] *= inv;
        }
      }
    }
  }

  /*
    Warm-start: strip row ri / col ci of parentX (stored at stride m_parent
    inside its sizeMaxSq-cell pool slot) into Xchild (m_child = m_parent - 1
    at stride m_child), then renormalize via Sinkhorn.

    `parentSliceStart` is the start offset of the parent's slot inside the
    persistent qpbXPool flat array.
  */
  inline proc extract_warm_start_from_pool(const ref parentXPool: [] real(32),
                                           parentSliceStart: int(32),
                                           ref Xchild: (sizeMax**2)*real(32),
                                           m_parent: int(32), ri: int(32), ci: int(32),
                                           param sinkIter: int)
  {
    const mc = m_parent - 1;
    for r2 in 0..<mc {
      const r = if r2 < ri then r2 else r2 + 1;
      for c2 in 0..<mc {
        const c = if c2 < ci then c2 else c2 + 1;
        Xchild[r2*mc + c2] = parentXPool[parentSliceStart + r*m_parent + c];
      }
    }
    sinkhorn_normalize(Xchild, mc, sinkIter);
  }

  /*
    Bidirectional dual repair for LAP warm-start. Row trim then column tighten.
    Restores feasibility u[i] + v[j] <= cost[i, j] for the NEW cost matrix
    against duals (u, v) carried from the previous LAP solve.

    Indices are 1-based to match the solve_lap_double convention.
  */
  inline proc repair_lap_duals(const ref cost: (sizeMax**2)*real(32),
                               n: int(32),
                               ref u: (sizeMax+1)*real(64),
                               ref v: (sizeMax+1)*real(64))
  {
    // Phase 1: row trim.
    for i in 1..n {
      const ui   = u[i];
      var viol: real(64) = 0.0;
      for j in 1..n {
        const cand = ui + v[j] - cost[(i-1)*n + (j-1)]:real(64);
        if (cand > viol) then viol = cand;
      }
      u[i] = ui - viol;
    }
    // Phase 2: column tighten. v[j] := min_i(c[i,j] - u[i]).
    for j in 1..n {
      var minv: real(64) = cost[0*n + (j-1)]:real(64) - u[1];
      for i in 2..n {
        const slack = cost[(i-1)*n + (j-1)]:real(64) - u[i];
        if (slack < minv) then minv = slack;
      }
      v[j] = minv;
    }
  }

  /*
    Jonker-Volgenant LAP, 1-based indexing inside. Reads an n x n float cost
    matrix at stride n. Writes perm[i] = col assigned to row i (0-based).
    rowDual / colDual receive u[1..n] / v[1..n] (the optimal duals).

    `warm_start = true` reuses u, v from the previous call; the caller is
    responsible for restoring feasibility (call repair_lap_duals first).
  */
  proc solve_lap_double(const ref cost: (sizeMax**2)*real(32), n: int(32),
                        ref perm:    sizeMax*int(32),
                        ref rowDual: sizeMax*real(64),
                        ref colDual: sizeMax*real(64),
                        ref u:       (sizeMax+1)*real(64),
                        ref v:       (sizeMax+1)*real(64),
                        ref p:       (sizeMax+1)*int(32),
                        ref way:     (sizeMax+1)*int(32),
                        ref minv:    (sizeMax+1)*real(64),
                        ref used:    (sizeMax+1)*int(8),
                        warm_start: bool)
  {
    if (n == 0) then return;
    if (n == 1) {
      perm[0]    = 0:int(32);
      rowDual[0] = cost[0]:real(64);
      colDual[0] = 0.0;
      u[1] = cost[0]:real(64);
      v[1] = 0.0;
      return;
    }
    if !warm_start {
      for k in 0..n { u[k] = 0.0; v[k] = 0.0; }
    }
    for k in 0..n { p[k] = 0:int(32); way[k] = 0:int(32); }
    for i in 1..n {
      for k in 0..n { minv[k] = BIG_DBL; used[k] = 0:int(8); }
      p[0] = i:int(32);
      var j0: int(32) = 0;
      do {
        used[j0] = 1:int(8);
        const i0 = p[j0]:int(32);
        var delta: real(64) = BIG_DBL;
        var j1: int(32) = 0;
        for j in 1..n {
          if (used[j] == 0:int(8)) {
            const cur = cost[(i0-1)*n + (j-1)]:real(64) - u[i0] - v[j];
            if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
            if (minv[j] < delta) { delta = minv[j]; j1 = j:int(32); }
          }
        }
        for j in 0..n {
          if (used[j] != 0:int(8)) { u[p[j]] += delta; v[j] -= delta; }
          else                     { minv[j] -= delta; }
        }
        j0 = j1;
      } while (p[j0] != 0);
      do {
        const j1 = way[j0]:int(32);
        p[j0] = p[j1];
        j0    = j1;
      } while (j0 != 0);
    }
    for j in 1..n {
      const i = p[j]:int(32);
      perm[i - 1] = (j - 1):int(32);
    }
    for i in 0..<n {
      rowDual[i] = u[i + 1];
      colDual[i] = v[i + 1];
    }
  }

  /*
    Cyclic Jacobi eigendecomposition of a real symmetric n x n matrix.
    Destroys A in place; eigvecs (stride n) holds eigenvectors in columns;
    eigvals (length n) returned sorted ascending.

    Storage in float, rotations computed in double, stored back in float.
  */
  proc jacobi_eig_sym(ref A:       (sizeMax**2)*real(32),
                      ref eigvecs: (sizeMax**2)*real(32),
                      ref eigvals: sizeMax*real(64),
                      n: int(32))
  {
    // V <- I
    for i in 0..<n {
      for j in 0..<n {
        eigvecs[i*n + j] = if i == j then 1.0:real(32) else 0.0:real(32);
      }
    }
    if (n <= 1) {
      if (n == 1) then eigvals[0] = A[0]:real(64);
      return;
    }
    for sweep in 0..<JAC_MAX_SWEEPS {
      // off-diagonal squared sum
      var off: real(64) = 0.0;
      for i in 0..<n {
        for j in (i+1)..<n {
          const a = A[i*n + j]:real(64);
          off += a * a;
        }
      }
      if (off < JAC_OFF_TOL) then break;

      for p in 0..<(n-1) {
        for q in (p+1)..<n {
          const apq = A[p*n + q]:real(64);
          if (abs(apq) < 1.0e-30) then continue;

          const app = A[p*n + p]:real(64);
          const aqq = A[q*n + q]:real(64);
          const theta = (aqq - app) / (2.0 * apq);
          var t: real(64);
          if (abs(theta) > 1.0e15) {
            t = 1.0 / (2.0 * theta);
          } else {
            const at  = abs(theta);
            const sgn: real(64) = if theta >= 0.0 then 1.0 else -1.0;
            t = sgn / (at + sqrt(theta*theta + 1.0));
          }
          const cc  = 1.0 / sqrt(1.0 + t*t);
          const ss  = t * cc;
          const tau = ss / (1.0 + cc);

          A[p*n + p] = (app - t * apq):real(32);
          A[q*n + q] = (aqq + t * apq):real(32);
          A[p*n + q] = 0.0:real(32);
          A[q*n + p] = 0.0:real(32);

          for k in 0..<n {
            if (k == p || k == q) then continue;
            const akp = A[k*n + p]:real(64);
            const akq = A[k*n + q]:real(64);
            const newKp = akp - ss * (akq + tau * akp);
            const newKq = akq + ss * (akp - tau * akq);
            A[k*n + p] = newKp:real(32);
            A[p*n + k] = newKp:real(32);
            A[k*n + q] = newKq:real(32);
            A[q*n + k] = newKq:real(32);
          }
          for k in 0..<n {
            const vkp = eigvecs[k*n + p]:real(64);
            const vkq = eigvecs[k*n + q]:real(64);
            eigvecs[k*n + p] = (vkp - ss * (vkq + tau * vkp)):real(32);
            eigvecs[k*n + q] = (vkq + ss * (vkp - tau * vkq)):real(32);
          }
        }
      }
    }
    // Read eigenvalues off the diagonal.
    for i in 0..<n do eigvals[i] = A[i*n + i]:real(64);
    // Insertion sort ascending; swap eigvecs columns in lockstep.
    for i in 1..<n {
      const ei = eigvals[i];
      var j = i - 1;
      while (j >= 0 && eigvals[j] > ei) {
        eigvals[j + 1] = eigvals[j];
        for r in 0..<n {
          const tmp = eigvecs[r*n + (j+1)];
          eigvecs[r*n + (j+1)] = eigvecs[r*n + j];
          eigvecs[r*n + j]     = tmp;
        }
        j -= 1;
      }
      eigvals[j + 1] = ei;
    }
  }

  /*
    Asymmetry correction. Returns -sum_k sigma_k(Fa) sigma_k(Da) where
    Fa = 0.5*(A - A^T), Da = 0.5*(B - B^T).

    A-side data (fnormFa + sing(Fa)) can be supplied from a per-parent cache
    via the `has_cached_A` flag; B-side is always per-child.
  */
  proc compute_asym_correction(const ref A_raw: (sizeMax**2)*real(32),
                               const ref B_raw: (sizeMax**2)*real(32),
                               m: int(32),
                               ref Fa:     (sizeMax**2)*real(32),
                               ref Da:     (sizeMax**2)*real(32),
                               ref FtF:    (sizeMax**2)*real(32),
                               ref DtD:    (sizeMax**2)*real(32),
                               ref evec_F: (sizeMax**2)*real(32),
                               ref evec_D: (sizeMax**2)*real(32),
                               ref evF:    sizeMax*real(64),
                               ref evD:    sizeMax*real(64),
                               has_cached_A: bool,
                               cached_fnormFa: real(64),
                               const ref cached_singFa: sizeMax*real(64)): real(64)
  {
    // fnorm_Fa: cache or compute.
    var fnorm_Fa: real(64);
    if has_cached_A {
      fnorm_Fa = cached_fnormFa;
    } else {
      fnorm_Fa = 0.0;
      for i in 0..<m {
        for j in (i+1)..<m {
          const dF = A_raw[i*m + j]:real(64) - A_raw[j*m + i]:real(64);
          fnorm_Fa += dF * dF;
        }
      }
    }
    if (fnorm_Fa < ASYM_EPS) then return 0.0;

    // fnorm_Da: always per-child.
    var fnorm_Da: real(64) = 0.0;
    for i in 0..<m {
      for j in (i+1)..<m {
        const dD = B_raw[i*m + j]:real(64) - B_raw[j*m + i]:real(64);
        fnorm_Da += dD * dD;
      }
    }
    if (fnorm_Da < ASYM_EPS) then return 0.0;

    // Singular values of Fa: from cache or recompute (Jacobi(FtF)).
    if !has_cached_A {
      for i in 0..<m {
        for j in 0..<m {
          Fa[i*m + j] = 0.5:real(32) * (A_raw[i*m + j] - A_raw[j*m + i]);
        }
      }
      matmul_TN(Fa, Fa, FtF, m, m, m, m, m, m);
      for i in 0..<m {
        for j in (i+1)..<m {
          const sF = 0.5:real(32) * (FtF[i*m + j] + FtF[j*m + i]);
          FtF[i*m + j] = sF; FtF[j*m + i] = sF;
        }
      }
      jacobi_eig_sym(FtF, evec_F, evF, m);
    }

    // Singular values of Da: always per-child.
    for i in 0..<m {
      for j in 0..<m {
        Da[i*m + j] = 0.5:real(32) * (B_raw[i*m + j] - B_raw[j*m + i]);
      }
    }
    matmul_TN(Da, Da, DtD, m, m, m, m, m, m);
    for i in 0..<m {
      for j in (i+1)..<m {
        const sD = 0.5:real(32) * (DtD[i*m + j] + DtD[j*m + i]);
        DtD[i*m + j] = sD; DtD[j*m + i] = sD;
      }
    }
    jacobi_eig_sym(DtD, evec_D, evD, m);

    var asym_norm: real(64) = 0.0;
    for k in 0..<m {
      var sF: real(64);
      if has_cached_A {
        sF = cached_singFa[k];
      } else {
        var eF = evF[k]; if (eF < 0.0) then eF = 0.0;
        sF = sqrt(eF);
      }
      var eD = evD[k]; if (eD < 0.0) then eD = 0.0;
      asym_norm += sF * sqrt(eD);
    }
    return -asym_norm;
  }

  /*
    QPB setup: builds Sp, Tp, gamma_s from symmetric m x m A, B.

    Slot map (all are caller-provided ref scratch):
      W      -- A-side eigvecs (descending after the flip below). Also used
                transiently as A_hat = V^T A V.
      U_eig  -- B-side eigvecs (ascending). Transiently B_hat = V^T B V.
      tmpM   -- Jacobi eigvec output / V*W / V*U scratch.

    If has_cached_A is true the A-side eigendecomp (W, sigma) and VW are
    read from the per-parent cache (W_cache / sigma_cache / VW_cache).
  */
  proc qpb_setup(const ref A: (sizeMax**2)*real(32),
                 const ref B: (sizeMax**2)*real(32),
                 m: int(32),
                 ref Sp: (sizeMax**2)*real(32),
                 ref Tp: (sizeMax**2)*real(32),
                 ref gamma_s: real(64),
                 ref W:     (sizeMax**2)*real(32),
                 ref U_eig: (sizeMax**2)*real(32),
                 ref tmpM:  (sizeMax**2)*real(32),
                 ref sigma: sizeMax*real(64),
                 ref lam:   sizeMax*real(64),
                 ref sbar:  sizeMax*real(64),
                 ref tbar:  sizeMax*real(64),
                 has_cached_A: bool,
                 const ref W_cache:     (sizeMax**2)*real(32),
                 const ref sigma_cache: sizeMax*real(64),
                 const ref VW_cache:    (sizeMax**2)*real(32))
  {
    const mm = m - 1;

    // ---- A side ----
    if has_cached_A {
      for i in 0..<mm {
        for k in 0..<mm {
          W[i*mm + k] = W_cache[i*sizeMax + k];
        }
      }
      for i in 0..<mm do sigma[i] = sigma_cache[i];
    } else {
      helmert_VTAV_dev(A, m, m, W, mm);                  // W = A_hat (transient)
      for i in 0..<mm {
        for j in (i+1)..<mm {
          const s = 0.5:real(32) * (W[i*mm + j] + W[j*mm + i]);
          W[i*mm + j] = s; W[j*mm + i] = s;
        }
      }
      // Jacobi writes ascending into sigma; flip to descending.
      jacobi_eig_sym(W, tmpM, sigma, mm);
      for i in 0..<(mm / 2) {
        const tmp = sigma[i];
        sigma[i] = sigma[mm - 1 - i];
        sigma[mm - 1 - i] = tmp;
      }
      for i in 0..<mm {
        for k in 0..<mm {
          W[i*mm + k] = tmpM[i*mm + (mm - 1 - k)];
        }
      }
    }

    // ---- B side ----
    helmert_VTAV_dev(B, m, m, U_eig, mm);                // U_eig = B_hat (transient)
    for i in 0..<mm {
      for j in (i+1)..<mm {
        const s = 0.5:real(32) * (U_eig[i*mm + j] + U_eig[j*mm + i]);
        U_eig[i*mm + j] = s; U_eig[j*mm + i] = s;
      }
    }
    // Jacobi writes ascending into lam (no flip).
    jacobi_eig_sym(U_eig, tmpM, lam, mm);
    for i in 0..<mm {
      for k in 0..<mm {
        U_eig[i*mm + k] = tmpM[i*mm + k];
      }
    }

    // ---- gamma, sbar, tbar recurrences ----
    var g: real(64) = 0.0;
    for i in 0..<mm do g += lam[i] * sigma[i];
    gamma_s = g;
    if (mm > 0) {
      tbar[0] = 0.0;
      for k in 0..<(mm - 1) {
        sbar[k]     = lam[k] * sigma[k] - tbar[k];
        tbar[k + 1] = sigma[k + 1] * (lam[k + 1] - lam[k]) + tbar[k];
      }
      sbar[mm - 1] = lam[mm - 1] * sigma[mm - 1] - tbar[mm - 1];
    }

    // ---- Sp[i,j] = sum_k VW[i,k] * sbar[k] * VW[j,k] ----
    if has_cached_A {
      for i in 0..<m {
        for k in 0..<mm {
          tmpM[i*mm + k] = VW_cache[i*sizeMax + k];
        }
      }
    } else {
      helmert_VA_dev(W, m, mm, tmpM, mm);                // tmpM = V * W  (m x mm)
    }
    for i in 0..<m {
      for j in 0..<m {
        var sum: real(64) = 0.0;
        for k in 0..<mm {
          sum += tmpM[i*mm + k]:real(64) * sbar[k] * tmpM[j*mm + k]:real(64);
        }
        Sp[i*m + j] = sum:real(32);
      }
    }
    // ---- Tp[i,j] = sum_k VU[i,k] * tbar[k] * VU[j,k] ----
    helmert_VA_dev(U_eig, m, mm, tmpM, mm);              // tmpM = V * U_eig
    for i in 0..<m {
      for j in 0..<m {
        var sum: real(64) = 0.0;
        for k in 0..<mm {
          sum += tmpM[i*mm + k]:real(64) * tbar[k] * tmpM[j*mm + k]:real(64);
        }
        Tp[i*m + j] = sum:real(32);
      }
    }
  }

  /*
    QPB Frank-Wolfe loop. Iterates the FW step on the QPB relaxation
    starting from X (warm-start from parent if has_warm, else uniform 1/m),
    tightens via the sparse-direction trick (one matmul per iter instead of
    five) and the across-iter recurrence on the X-side caches
    (T1 = Sp*X, T2 = A*X*B, T3 = X*Tp). LAP duals (u_lap, v_lap) are
    persisted across iters and trimmed via repair_lap_duals each call.

    On exit:
      * z_best holds the best continuous bound across FW iterations
      * z_last holds the very last z
      * G holds the LAST iter's gradient (rebuilt from caches on maxFW exit)
      * X has been updated to the FW solution
      * (rowD, colD) hold the LAST iter's LAP duals (used to derive the row
        of dual reduced costs for VF)

    Slot map (during FW): G=S1, T1=S2, T2=S3, T3=S5, d=S4.
  */
  proc qpb_fw(const ref A: (sizeMax**2)*real(32),
              const ref B: (sizeMax**2)*real(32),
              const ref C: (sizeMax**2)*real(32),
              const ref Sp: (sizeMax**2)*real(32),
              const ref Tp: (sizeMax**2)*real(32),
              gamma_s: real(64), m: int(32),
              ref X: (sizeMax**2)*real(32),
              has_warm: bool,
              const ref parentXPool: [] real(32),
              parentXSliceStart: int(32),
              piIdx: int(32), pjIdx: int(32), mParentWarm: int(32),
              upperBound: real(64), param maxFW: int, param tol: real(64), param sinkIter: int,
              ref z_best: real(64), ref z_last: real(64),
              ref G:  (sizeMax**2)*real(32),
              ref T1: (sizeMax**2)*real(32),
              ref T2: (sizeMax**2)*real(32),
              ref T3: (sizeMax**2)*real(32),
              ref d:  (sizeMax**2)*real(32),
              ref perm:    sizeMax*int(32),
              ref rowD:    sizeMax*real(64),
              ref colD:    sizeMax*real(64),
              ref u_lap:   (sizeMax+1)*real(64),
              ref v_lap:   (sizeMax+1)*real(64),
              ref p_lap:   (sizeMax+1)*int(32),
              ref way_lap: (sizeMax+1)*int(32),
              ref minv_lap:(sizeMax+1)*real(64),
              ref used_lap:(sizeMax+1)*int(8))
  {
    const cells = m * m;

    // ---- X init ----
    if has_warm {
      extract_warm_start_from_pool(parentXPool, parentXSliceStart,
                                   X, mParentWarm, piIdx, pjIdx, sinkIter);
    } else {
      const u: real(32) = (1.0 / m:real(64)):real(32);
      for idx in 0..<cells do X[idx] = u;
    }

    z_best = -BIG_DBL;
    z_last = -BIG_DBL;

    var lap_warm: bool = false;

    // perm_inv persists across iters; int(8) is safe (m <= sizeMax = 32 << int8 max).
    var perm_inv: sizeMax*int(8);

    var have_prev: bool = false;
    var t_prev:    real(64) = 0.0;
    var g_clobbered: bool = false;

    for it in 0..<maxFW {
      // ---- X-side caches for this iter ----
      if !have_prev {
        matmul(A,  X, G,  m, m, m, m, m, m);       // G  = A * X (scratch)
        matmul(G,  B, T2, m, m, m, m, m, m);       // T2 = A * X * B  (cache)
        matmul(Sp, X, T1, m, m, m, m, m, m);       // T1 = Sp * X     (cache)
        matmul(X, Tp, T3, m, m, m, m, m, m);       // T3 = X * Tp     (cache)
      } else {
        const one_minus_t = 1.0 - t_prev;
        const t_d         = t_prev;
        // T2 (AXB) <- (1 - t) * T2_old + t * G_holding_APB
        for idx in 0..<cells {
          const v = one_minus_t * T2[idx]:real(64) + t_d * G[idx]:real(64);
          T2[idx] = v:real(32);
        }
        // T1 (SpX) <- (1 - t) * T1_old + t * Sp[i, perm_inv[j]]
        for i in 0..<m {
          for j in 0..<m {
            const piv = perm_inv[j]:int(32);
            const v = one_minus_t * T1[i*m + j]:real(64)
                    + t_d         * Sp[i*m + piv]:real(64);
            T1[i*m + j] = v:real(32);
          }
        }
        // T3 (XTp) <- (1 - t) * T3_old + t * Tp[perm[i], j]
        for i in 0..<m {
          const pi = perm[i]:int(32);
          for j in 0..<m {
            const v = one_minus_t * T3[i*m + j]:real(64)
                    + t_d         * Tp[pi*m + j]:real(64);
            T3[i*m + j] = v:real(32);
          }
        }
      }

      // ---- G = 2*(T2 - T1 - T3) + C ----
      for idx in 0..<cells {
        G[idx] = 2.0:real(32) * (T2[idx] - T1[idx] - T3[idx]) + C[idx];
      }
      g_clobbered = false;

      // ---- LAP(G) ----
      if lap_warm then repair_lap_duals(G, m, u_lap, v_lap);
      solve_lap_double(G, m, perm, rowD, colD,
                       u_lap, v_lap, p_lap, way_lap, minv_lap, used_lap,
                       lap_warm);
      lap_warm = true;

      // ---- f(X), FW gap, z ----
      var GdotX: real(64) = 0.0;
      var CdotX: real(64) = 0.0;
      for idx in 0..<cells {
        GdotX += G[idx]:real(64) * X[idx]:real(64);
        CdotX += C[idx]:real(64) * X[idx]:real(64);
      }
      var GdotY: real(64) = 0.0;
      for i in 0..<m do GdotY += G[i*m + perm[i]:int(32)]:real(64);
      const fX     = 0.5 * (GdotX + CdotX) + gamma_s;
      const fw_gap = GdotY - GdotX;
      const z      = fX + fw_gap;

      if (z > z_best) then z_best = z;
      z_last = z;

      // ---- Early termination ----
      if (ceil(z_best - 1.0e-6) >= upperBound) then break;
      if (-fw_gap < tol * max(1.0, abs(fX)))  then break;

      // ---- d = perm_matrix - X, plus inverse permutation ----
      // Both the d += 1 update and the perm_inv write iterate i in 0..<m
      // and read perm[i]. Fuse them to halve the loop overhead and read
      // perm[i] only once per i.
      for idx in 0..<cells do d[idx] = -X[idx];
      for i in 0..<m {
        const pi = perm[i]:int(32);
        d[i*m + pi] += 1.0:real(32);
        perm_inv[pi] = i:int(8);
      }

      // ---- Qdd via sparse-direction trick ----
      // qdd2 = <d, (Sp P - Sp X)>
      var qdd2: real(64) = 0.0;
      for i in 0..<m {
        for j in 0..<m {
          const piv = perm_inv[j]:int(32);
          const dij    = d[i*m + j]:real(64);
          const spp_ij = Sp[i*m + piv]:real(64);
          const spx_ij = T1[i*m + j]:real(64);
          qdd2 += dij * (spp_ij - spx_ij);
        }
      }
      // qdd3 = <d, (P Tp - X Tp)>
      var qdd3: real(64) = 0.0;
      for i in 0..<m {
        const pi = perm[i]:int(32);
        for j in 0..<m {
          const dij    = d[i*m + j]:real(64);
          const ptp_ij = Tp[pi*m + j]:real(64);
          const xtp_ij = T3[i*m + j]:real(64);
          qdd3 += dij * (ptp_ij - xtp_ij);
        }
      }
      // qdd1 = <d, (A P) B - A X B>. Fused A-col-perm + matmul, output to G.
      for i in 0..<m {
        for j in 0..<m {
          var sum: real(64) = 0.0;
          for k in 0..<m {
            const piv_k = perm_inv[k]:int(32);
            sum += A[i*m + piv_k]:real(64) * B[k*m + j]:real(64);
          }
          G[i*m + j] = sum:real(32);
        }
      }
      var qdd1: real(64) = 0.0;
      for idx in 0..<cells {
        const dij    = d[idx]:real(64);
        const apb_ij = G[idx]:real(64);
        const axb_ij = T2[idx]:real(64);
        qdd1 += dij * (apb_ij - axb_ij);
      }
      g_clobbered = true;

      const Qdd = qdd1 - qdd2 - qdd3;
      var t_step: real(64);
      if (Qdd > Q_CURV_EPS) {
        t_step = -fw_gap / (2.0 * Qdd);
        if (t_step < 0.0) then t_step = 0.0;
        if (t_step > 1.0) then t_step = 1.0;
      } else {
        t_step = 1.0;
      }

      // ---- X += t * d ----
      const t_step_f = t_step:real(32);
      for idx in 0..<cells do X[idx] += t_step_f * d[idx];

      t_prev    = t_step;
      have_prev = true;
    }

    // ---- G recovery on maxFW exit ----
    if g_clobbered {
      for idx in 0..<cells {
        G[idx] = 2.0:real(32) * (T2[idx] - T1[idx] - T3[idx]) + C[idx];
      }
    }
  }

  /*****************************************************************************
    Top-level per-child QPB kernel.

    One iteration of the @assertOnGpu foreach calls this once per child. The
    big per-thread tuples (A/B/C/X/Sp/Tp/S1..S5 + LAP scratch + small
    vectors) live in this proc's frame and end up in GPU thread-local memory.

    Notable side-effects:
      * writes `boundOut` (the integer lower bound of this child)
      * writes `output` (Node_QPB_Out: qpbReducedCostsRow + qpbBoundContLast +
        qpbFixedCost)
      * writes the child's primal X into parentXPool_d at the slot named by
        `input.outSlot` (no DtoH for X; the next batch reads it directly as
        warm-start)
  *****************************************************************************/
  proc bound_QPB_kernel(const ref input: Node_QPB_In,
                        ref output: Node_QPB_Out,
                        ref boundOut: int,
                        ref parentXPool_d:    [] real(32),
                        const ref parentW_d:        [] real(32),
                        const ref parentSigma_d:    [] real(64),
                        const ref parentVW_d:       [] real(32),
                        const ref parentFnormFa_d:  [] real(64),
                        const ref parentSingFa_d:   [] real(64),
                        const ref parent_C_d:       [] real(32),
                        const ref F: [] int(32),
                        const ref D: [] int(32),
                        const ref queue_fac: [] int(32),
                        n: int(32), N: int(32),
                        upperBound: int,
                        param maxFW: int, param tol: real(64), param sinkIter: int)
  {
    // Per-thread working set.
    var A : (sizeMax**2)*real(32);
    var B : (sizeMax**2)*real(32);
    var C : (sizeMax**2)*real(32);
    var X : (sizeMax**2)*real(32);
    var Sp: (sizeMax**2)*real(32);
    var Tp: (sizeMax**2)*real(32);
    var S1: (sizeMax**2)*real(32);
    var S2: (sizeMax**2)*real(32);
    var S3: (sizeMax**2)*real(32);
    var S4: (sizeMax**2)*real(32);
    var S5: (sizeMax**2)*real(32);

    var sigma: sizeMax*real(64);
    var lam:   sizeMax*real(64);
    var sbar:  sizeMax*real(64);
    var tbar:  sizeMax*real(64);
    var evF:   sizeMax*real(64);
    var evD:   sizeMax*real(64);
    var rowD:  sizeMax*real(64);
    var colD:  sizeMax*real(64);
    var perm:  sizeMax*int(32);

    var u_lap:    (sizeMax+1)*real(64);
    var v_lap:    (sizeMax+1)*real(64);
    var minv_lap: (sizeMax+1)*real(64);
    var p_lap:    (sizeMax+1)*int(32);
    var way_lap:  (sizeMax+1)*int(32);
    var used_lap: (sizeMax+1)*int(8);

    var unassigned_fac: sizeMax*int(32);
    var unassigned_loc: sizeMax*int(32);
    var assigned_fac:   sizeMax*int(32);

    // ---- Identify assigned / unassigned ----
    var nAss: int(32) = 0;
    var m:    int(32) = 0;
    var pCnt: int(32) = 0;
    for i in 0..<n {
      if (input.mapping[i] != -1:int(8)) {
        assigned_fac[nAss] = i;
        nAss += 1;
      } else {
        unassigned_fac[m] = i;
        m += 1;
      }
    }
    for k in 0..<N {
      if input.available[k] {
        unassigned_loc[pCnt] = k;
        pCnt += 1;
      }
    }

    // ---- Fixed cost ----
    var fc: real(64) = 0.0;
    for i in 0..<n {
      const mi = input.mapping[i]:int(32);
      if (mi == -1) then continue;
      for j in 0..<n {
        const mj = input.mapping[j]:int(32);
        if (mj == -1) then continue;
        fc += F[i*N + j]:real(64) * D[mi*N + mj]:real(64);
      }
    }
    const fixed_cost = fc:int;

    // ---- Trivial cases (m == 0 leaf-like, or shape mismatch) ----
    if (m == 0 || m != pCnt) {
      output.qpbFixedCost     = fixed_cost;
      output.qpbBoundContLast = 0.0;
      boundOut                = fixed_cost;
      return;
    }

    // ---- Build A_sub (raw), B_sub (raw) ----
    for i in 0..<m {
      const fi = unassigned_fac[i];
      for j in 0..<m {
        const fj = unassigned_fac[j];
        A[i*m + j] = F[fi*N + fj]:real(32);
      }
    }
    for i in 0..<m {
      const li = unassigned_loc[i];
      for j in 0..<m {
        const lj = unassigned_loc[j];
        B[i*m + j] = D[li*N + lj]:real(32);
      }
    }

    // ---- Build C: cache path or fallback ----
    const cacheIdx = input.parentEigenIdx;
    if (cacheIdx >= 0) {
      const parentIdxFac = input.parentIdxFac:int(32);
      const parentIdxLoc = input.parentIdxLoc:int(32);
      const branching_fac = queue_fac[input.depth:int(32) - 1];
      const branching_loc = input.mapping[branching_fac]:int(32);
      const parent_C_base = cacheIdx * sizeMaxSq;
      for i in 0..<m {
        const pi = i + (if i >= parentIdxFac then 1:int(32) else 0:int(32));
        const fi = unassigned_fac[i];
        for k in 0..<m {
          const pk = k + (if k >= parentIdxLoc then 1:int(32) else 0:int(32));
          const lk = unassigned_loc[k];
          const base = parent_C_d[parent_C_base + pi*sizeMax + pk]:real(64);
          const new_term =
              F[fi*N + branching_fac]:real(64) * D[lk*N + branching_loc]:real(64)
            + F[branching_fac*N + fi]:real(64) * D[branching_loc*N + lk]:real(64);
          C[i*m + k] = (base + new_term):real(32);
        }
      }
    } else {
      // Same hoist as in parent_eigen_setup_proc: precompute the assigned-
      // facility -> location map once so the inner (i, k) double loop only
      // hits thread-local memory.
      var assigned_locs: sizeMax*int(32);
      for a in 0..<nAss do
        assigned_locs[a] = input.mapping[assigned_fac[a]]:int(32);

      for i in 0..<m {
        const fi = unassigned_fac[i];
        for k in 0..<m {
          const lk = unassigned_loc[k];
          var cross: real(64) = 0.0;
          for a in 0..<nAss {
            const fa = assigned_fac[a];
            const lb = assigned_locs[a];
            cross += F[fi*N + fa]:real(64) * D[lk*N + lb]:real(64);
            cross += F[fa*N + fi]:real(64) * D[lb*N + lk]:real(64);
          }
          C[i*m + k] = cross:real(32);
        }
      }
    }

    // ---- Asymmetry correction (uses S1..S4 + X + Sp as 6 scratch slots) ----
    const has_cached_A = (cacheIdx >= 0 && m >= 3);
    var cached_fnormFa: real(64) = 0.0;
    var cached_singFa: sizeMax*real(64);
    if has_cached_A {
      cached_fnormFa = parentFnormFa_d[cacheIdx];
      const base = cacheIdx * sizeMax;
      for k in 0..<sizeMax do cached_singFa[k] = parentSingFa_d[base + k];
    }
    const asym_correction =
        compute_asym_correction(A, B, m, S1, S2, S3, S4, X, Sp,
                                evF, evD,
                                has_cached_A, cached_fnormFa, cached_singFa);

    // ---- Symmetrize A and B in place ----
    for i in 0..<m {
      for j in (i+1)..<m {
        const sA = 0.5:real(32) * (A[i*m + j] + A[j*m + i]);
        A[i*m + j] = sA; A[j*m + i] = sA;
        const sB = 0.5:real(32) * (B[i*m + j] + B[j*m + i]);
        B[i*m + j] = sB; B[j*m + i] = sB;
      }
    }

    // ---- QPB setup (Sp, Tp, gamma) ----
    var W_cache_local:     (sizeMax**2)*real(32);
    var sigma_cache_local: sizeMax*real(64);
    var VW_cache_local:    (sizeMax**2)*real(32);
    if has_cached_A {
      const wbase  = cacheIdx * sizeMaxSq;
      const sgbase = cacheIdx * sizeMax;
      const vwbase = cacheIdx * sizeMaxSq;
      for idx in 0..<sizeMaxSq {
        W_cache_local[idx]  = parentW_d[wbase + idx];
        VW_cache_local[idx] = parentVW_d[vwbase + idx];
      }
      for idx in 0..<sizeMax do sigma_cache_local[idx] = parentSigma_d[sgbase + idx];
    }
    var gamma_s: real(64) = 0.0;
    qpb_setup(A, B, m, Sp, Tp, gamma_s,
              S2, S3, S4,
              sigma, lam, sbar, tbar,
              has_cached_A, W_cache_local, sigma_cache_local, VW_cache_local);

    // ---- Frank-Wolfe ----
    const pxs   = input.parentXSlot;
    const hasW  = (pxs >= 0);
    const warmSliceStart: int(32) = if hasW then pxs * sizeMaxSq else 0:int(32);
    var z_best: real(64) = 0.0;
    var z_last: real(64) = 0.0;
    qpb_fw(A, B, C, Sp, Tp, gamma_s, m,
           X,
           hasW, parentXPool_d, warmSliceStart,
           input.parentIdxFac:int(32), input.parentIdxLoc:int(32), m + 1,
           (upperBound - fixed_cost):real(64),
           maxFW, tol, sinkIter,
           z_best, z_last,
           S1, S2, S3, S5, S4,
           perm, rowD, colD,
           u_lap, v_lap, p_lap, way_lap, minv_lap, used_lap);

    // ---- Integer bound + outputs ----
    const corrected      = z_best + asym_correction;
    const correctedLast  = z_last + asym_correction;
    const lb_tol         = max(QPB_VF_TOL_BASE, QPB_VF_TOL_REL * abs(corrected));
    const remaining_lb   = ceil(corrected - lb_tol):int;
    output.qpbFixedCost     = fixed_cost;
    output.qpbBoundContLast = correctedLast;
    boundOut                = fixed_cost + remaining_lb;

    // ---- Write X into the assigned pool slot ----
    if (input.outSlot >= 0) {
      const dst = input.outSlot * sizeMaxSq;
      for i in 0..<m {
        for j in 0..<m {
          parentXPool_d[dst + i*m + j] = X[i*m + j];
        }
      }
    }

    // ---- Write the row of dual reduced costs (only the row VF will read) ----
    const row_i = input.nextBranchFacIdx:int(32);
    if (row_i >= 0 && row_i < m) {
      const ru = rowD[row_i];
      for j in 0..<m {
        output.qpbReducedCostsRow[j] =
          (S1[row_i*m + j]:real(64) - ru - colD[j]):real(32);
      }
    }
  }


  /*****************************************************************************
    Per-parent eigen + C-matrix setup. One invocation per parent slot.

    Reads the parent's mapping from the kernel-input record of the parent's
    first surviving child (parent_first_child_idx names the index of that
    record in inputs). Writes:
      * parentW_d / parentSigma_d / parentVW_d  (A-side eigendecomp)
      * parentFnormFa_d / parentSingFa_d        (asym A-side cache)
      * parent_C_d                              (cross-cost matrix)

    For each slot the caches are at stride sizeMax (W/VW/C are sizeMaxSq per
    slot, sigma/singFa are sizeMax per slot, fnormFa is one double per slot).
  *****************************************************************************/
  proc parent_eigen_setup_proc(const ref input: Node_QPB_In,
                               const ref F: [] int(32),
                               const ref D: [] int(32),
                               const ref queue_fac: [] int(32),
                               n: int(32), N: int(32),
                               slot: int(32),
                               ref parentW_d:       [] real(32),
                               ref parentSigma_d:   [] real(64),
                               ref parentVW_d:      [] real(32),
                               ref parentFnormFa_d: [] real(64),
                               ref parentSingFa_d:  [] real(64),
                               ref parent_C_d:      [] real(32))
  {
    var unassigned_fac: sizeMax*int(32);
    var m: int(32) = 0;
    for i in 0..<n {
      if (input.mapping[i] == -1:int(8)) {
        unassigned_fac[m] = i;
        m += 1;
      }
    }

    // ---- Parent C-matrix cache ----
    // Recover the parent's view by undoing the child's branching:
    //   branching_fac = queue_fac[depth_child - 1]
    //   branching_loc = input.mapping[branching_fac]
    //   parent_uf     = child_uf + {branching_fac}    (sorted)
    //   parent_ul     = child_ul + {branching_loc}    (sorted)
    //   parent_assigned = child_assigned - {branching_fac}
    {
      const depth_child   = input.depth:int(32);
      const branching_fac: int(32) = if depth_child > 0 then queue_fac[depth_child - 1] else (-1):int(32);
      const branching_loc: int(32) = if branching_fac >= 0 then input.mapping[branching_fac]:int(32) else (-1):int(32);

      var parent_uf:       sizeMax*int(32);
      var parent_ul:       sizeMax*int(32);
      var parent_assigned: sizeMax*int(32);
      var mp:   int(32) = 0;
      var pp:   int(32) = 0;
      var nass: int(32) = 0;

      for f in 0..<n {
        if (input.mapping[f] == -1:int(8) || f == branching_fac) {
          parent_uf[mp] = f;
          mp += 1;
        } else {
          parent_assigned[nass] = f;
          nass += 1;
        }
      }
      for l in 0..<N {
        if (input.available[l] || l == branching_loc) {
          parent_ul[pp] = l;
          pp += 1;
        }
      }

      // Hoist the per-assigned-facility location lookup out of the (ii, kk)
      // double loop — input.mapping[fa] is invariant in (ii, kk) and lives
      // in the kernel-input record (off-thread memory). Materializing the
      // length-nass int array once saves ~mp*pp redundant loads per parent.
      var assigned_locs: sizeMax*int(32);
      for a in 0..<nass do
        assigned_locs[a] = input.mapping[parent_assigned[a]]:int(32);

      const cbase = slot * sizeMaxSq;
      for ii in 0..<mp {
        const fi = parent_uf[ii];
        for kk in 0..<pp {
          const lk = parent_ul[kk];
          var cross: real(64) = 0.0;
          for a in 0..<nass {
            const fa = parent_assigned[a];
            const lb = assigned_locs[a];
            cross += F[fi*N + fa]:real(64) * D[lk*N + lb]:real(64);
            cross += F[fa*N + fi]:real(64) * D[lb*N + lk]:real(64);
          }
          parent_C_d[cbase + ii*sizeMax + kk] = cross:real(32);
        }
      }
    }

    // ---- Below m >= 3 the main kernel skips lookup; zero defensively. ----
    if (m < 3) {
      const sgbase_small = slot * sizeMax;
      const wbase_small  = slot * sizeMaxSq;
      const vwbase_small = slot * sizeMaxSq;
      for k in 0..<sizeMax  do parentSigma_d[sgbase_small + k] = 0.0;
      for k in 0..<sizeMaxSq do parentW_d[wbase_small + k] = 0.0:real(32);
      for k in 0..<sizeMaxSq do parentVW_d[vwbase_small + k] = 0.0:real(32);
      parentFnormFa_d[slot] = 0.0;
      for k in 0..<sizeMax do parentSingFa_d[sgbase_small + k] = 0.0;
      return;
    }

    // ---- Build A (raw) ----
    var A: (sizeMax**2)*real(32);
    for i in 0..<m {
      const fi = unassigned_fac[i];
      for j in 0..<m {
        const fj = unassigned_fac[j];
        A[i*m + j] = F[fi*N + fj]:real(32);
      }
    }

    // ---- A-side asym cache: fnorm_Fa + singFa[] ----
    var fnorm_Fa: real(64) = 0.0;
    for i in 0..<m {
      for j in (i+1)..<m {
        const dF = A[i*m + j]:real(64) - A[j*m + i]:real(64);
        fnorm_Fa += dF * dF;
      }
    }
    parentFnormFa_d[slot] = fnorm_Fa;
    const sfbase = slot * sizeMax;
    if (fnorm_Fa >= ASYM_EPS) {
      var Fa:     (sizeMax**2)*real(32);
      var FtF:    (sizeMax**2)*real(32);
      var evec_F: (sizeMax**2)*real(32);
      var evF:    sizeMax*real(64);
      for i in 0..<m {
        for j in 0..<m {
          Fa[i*m + j] = 0.5:real(32) * (A[i*m + j] - A[j*m + i]);
        }
      }
      matmul_TN(Fa, Fa, FtF, m, m, m, m, m, m);
      for i in 0..<m {
        for j in (i+1)..<m {
          const sF = 0.5:real(32) * (FtF[i*m + j] + FtF[j*m + i]);
          FtF[i*m + j] = sF; FtF[j*m + i] = sF;
        }
      }
      jacobi_eig_sym(FtF, evec_F, evF, m);
      for k in 0..<m {
        var e = evF[k]; if (e < 0.0) then e = 0.0;
        parentSingFa_d[sfbase + k] = sqrt(e);
      }
      for k in m..<sizeMax do parentSingFa_d[sfbase + k] = 0.0;
    } else {
      for k in 0..<sizeMax do parentSingFa_d[sfbase + k] = 0.0;
    }

    // ---- Symmetrize A in place ----
    for i in 0..<m {
      for j in (i+1)..<m {
        const s = 0.5:real(32) * (A[i*m + j] + A[j*m + i]);
        A[i*m + j] = s; A[j*m + i] = s;
      }
    }

    // ---- A_hat = V^T A V, symmetrize, eigendecompose ----
    const mm = m - 1;
    var tmpM:  (sizeMax**2)*real(32);
    var A_hat: (sizeMax**2)*real(32);
    helmert_VTAV_dev(A, m, m, A_hat, mm);
    for i in 0..<mm {
      for j in (i+1)..<mm {
        const s = 0.5:real(32) * (A_hat[i*mm + j] + A_hat[j*mm + i]);
        A_hat[i*mm + j] = s; A_hat[j*mm + i] = s;
      }
    }
    var sigma_asc: sizeMax*real(64);
    jacobi_eig_sym(A_hat, tmpM, sigma_asc, mm);

    // ---- Write descending eigvals / eigvecs into the slot (stride sizeMax) ----
    const sgbase = slot * sizeMax;
    const wbase  = slot * sizeMaxSq;
    for i in 0..<mm do parentSigma_d[sgbase + i] = sigma_asc[mm - 1 - i];
    for i in 0..<mm {
      for k in 0..<mm {
        parentW_d[wbase + i*sizeMax + k] = tmpM[i*mm + (mm - 1 - k)];
      }
    }

    // ---- VW = V * W at stride sizeMax ----
    // We need helmert_VA_dev to read W at stride sizeMax and write VW at
    // stride sizeMax. helmert_VA_dev takes its input/output as (sizeMax**2)
    // tuples and indexes them with explicit stride, so we stage W into a
    // tuple at stride sizeMax.
    var W_stage:  (sizeMax**2)*real(32);
    var VW_stage: (sizeMax**2)*real(32);
    for i in 0..<sizeMaxSq do W_stage[i] = parentW_d[wbase + i];
    helmert_VA_dev(W_stage, m, sizeMax, VW_stage, sizeMax);
    const vwbase = slot * sizeMaxSq;
    for i in 0..<sizeMaxSq do parentVW_d[vwbase + i] = VW_stage[i];
  }
}
