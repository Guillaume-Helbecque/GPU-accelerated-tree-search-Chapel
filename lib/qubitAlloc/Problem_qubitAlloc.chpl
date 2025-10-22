module Problem_qubitAlloc
{
  use CTypes;

  use Util_qubitAlloc;

  config param sizeMax: int(32) = 27;

  proc Prioritization(ref priority, const ref F, n: int(32), N: int(32))
  {
    var sF: [0..<N] int(32);

    for i in 0..<n do
      for j in 0..<n do
        sF[i] += F[i * n + j];

    var min_inter, min_inter_index: int(32);

    for i in 0..<N {
      min_inter = sF[0];
      min_inter_index = 0;

      for j in 1..<N {
        if (sF[j] < min_inter) {
          min_inter = sF[j];
          min_inter_index = j;
        }
      }

      priority[N-1-i] = min_inter_index;

      sF[min_inter_index] = INF;

      for j in 0..<N {
        if (sF[j] != INF) then
          sF[j] -= F[j * n + min_inter_index];
      }
    }
  }

  proc GreedyAllocation(const ref D, const ref F, const ref priority, n, N)
  {
    var route_cost = INF;

    var l_min, k, i: int(32);
    var route_cost_temp, cost_incre, min_cost_incre: int(32);

    for j in 0..<N {
      var alloc_temp: [0..<sizeMax] int(32) = -1;
      var available: [0..<N] bool = true;

      alloc_temp[priority[0]] = j;
      available[j] = false;

      // for each logical qubit (after the first one)
      for p in 1..<n {
        k = priority[p];

        min_cost_incre = INF;

        // find physical qubit with least increasing route cost
        for l in 0..<N {
          if available[l] {
            cost_incre = 0;
            for q in 0..<p {
              i = priority[q];
              cost_incre += F[i * n + k] * D[alloc_temp[i] * N + l];
            }

            if (cost_incre < min_cost_incre) {
              l_min = l;
              min_cost_incre = cost_incre;
            }
          }
        }

        alloc_temp[k] = l_min;
        available[l_min] = false;
      }

      route_cost_temp = ObjectiveFunction(alloc_temp, D, F, n, N);

      if (route_cost_temp < route_cost) then
        route_cost = route_cost_temp;
    }

    return route_cost;
  }

  proc ObjectiveFunction(const mapping, const ref D, const ref F, n, N)
  {
    var route_cost: int(32);

    for i in 0..<n {
      if (mapping[i] == -1) then
        continue;

      for j in i..<n {
        if (mapping[j] == -1) then
          continue;

        route_cost += F[i * n + j] * D[mapping[i] * N + mapping[j]];
      }
    }

    return 2*route_cost;
  }

  /*******************************************************
                      HIGHTOWER-HAHN BOUND
    *******************************************************/

  proc Hungarian_HHB(ref C, i0, j0, n)
  {
    var w, j_cur, j_next: int(32);

    // job[j] = worker assigned to job j, or -1 if unassigned
    var job: sizeMax*int(32);//allocate(int(32), n+1);
    for i in 0..n do job[i] = -1;

    // yw[w] is the potential for worker w
    // yj[j] is the potential for job j
    var yw: (sizeMax+1)*int(32);//allocate(int(32), n);
    for i in 0..<n do yw[i] = 0;
    var yj: (sizeMax+1)*int(32);//allocate(int(32), n+1);
    for i in 0..n do yj[i] = 0;

    // main Hungarian algorithm
    for w_cur in 0..<n {
      j_cur = n;
      job[j_cur] = w_cur;

      var min_to: (sizeMax+1)*int(32);//allocate(int(32), n+1);
      for i in 0..n do min_to[i] = INFD2;
      var prv: (sizeMax+1)*int(32);//allocate(int(32), n+1);
      for i in 0..n do prv[i] = -1;
      var in_Z: (sizeMax+1)*bool;//allocate(bool, n+1);
      for i in 0..n do in_Z[i] = false;

      while (job[j_cur] != -1) {
        in_Z[j_cur] = true;
        w = job[j_cur];
        var delta = INFD2;
        j_next = 0;

        for j in 0..<n {
          if !in_Z[j] {
            // reduced cost = C[w][j] - yw[w] - yj[j]
            var cur_cost = C[idx4D(i0, j0, w, j, n)] - yw[w] - yj[j];

            if ckmin(min_to[j], cur_cost) then
              prv[j] = j_cur;
            if ckmin(delta, min_to[j]) then
              j_next = j;
          }
        }

        // update potentials
        for j in 0..n {
          if in_Z[j] {
            yw[job[j]] += delta;
            yj[j] -= delta;
          }
          else {
            min_to[j] -= delta;
          }
        }

        j_cur = j_next;
      }

      // update worker assignment along the found augmenting path
      while (j_cur != n) {
        var j = prv[j_cur];
        job[j_cur] = job[j];
        j_cur = j;
      }

      /* deallocate(min_to);
      deallocate(prv);
      deallocate(in_Z); */
    }

    // compute total cost
    var total_cost: int(32);

    // for j in [0..n-1], job[j] is the worker assigned to job j
    for j in 0..<n {
      if (job[j] != -1) then
        total_cost += C[idx4D(i0, j0, job[j], j, n)];
    }

    // OPTIONAL: Reflecting the "reduced costs" after the Hungarian
    // algorithm by applying the final potentials:
    for w in 0..<n {
      for j in 0..<n {
        if (C[idx4D(i0, j0, w, j, n)] < INFD2) {
          // subtract the final potentials from the original cost
          C[idx4D(i0, j0, w, j, n)] = C[idx4D(i0, j0, w, j, n)] - yw[w] - yj[j];
        }
      }
    }

    /* deallocate(job);
    deallocate(yw);
    deallocate(yj); */

    return total_cost;
  }

  proc distributeLeader(ref C, ref L, n)
  {
    var leader_cost, leader_cost_div, leader_cost_rem, val: int(32);

    if (n == 1) {
      C[0] = 0;
      L[0] = 0;

      return;
    }

    for i in 0..<n {
      for j in 0..<n {
        leader_cost = L[i*n + j];

        C[idx4D(i, j, i, j, n)] = 0;
        L[i*n + j] = 0;

        if (leader_cost == 0) {
          continue;
        }

        leader_cost_div = leader_cost / (n - 1);
        leader_cost_rem = leader_cost % (n - 1);

        for k in 0..<n {
          if (k == i) then
            continue;

          val = leader_cost_div + (k < leader_cost_rem || (k == leader_cost_rem && i < k));

          for l in 0..<n {
            if (l != j) then
              C[idx4D(i, j, k, l, n)] += val;
          }
        }
      }
    }
  }

  proc halveComplementary(ref C, n)
  {
    var cost_sum: int(32);

    for i in 0..<n {
      for j in 0..<n {
        for k in i..<n {
          for l in 0..<n {
            if ((k != i) && (l != j)) {
              cost_sum = C[idx4D(i, j, k, l, n)] + C[idx4D(k, l, i, j, n)];
              C[idx4D(i, j, k, l, n)] = cost_sum / 2;
              C[idx4D(k, l, i, j, n)] = cost_sum / 2;

              if (cost_sum % 2 == 1) {
                if ((i + j + k + l) % 2 == 0) then // total index parity for balance
                  C[idx4D(i, j, k, l, n)] += 1;
                else
                  C[idx4D(k, l, i, j, n)] += 1;
              }
            }
          }
        }
      }
    }
  }

  proc bound_HHB(ref node, best, it_max)
  {
    ref lb = node.lower_bound;
    ref C = node.costs;
    ref L = node.leader;
    const m = node.size;

    var cost, incre: int(32);

    var it = 0;

    while (it < it_max && lb <= best) {
      it += 1;

      distributeLeader(C, L, m);
      halveComplementary(C, m);

      // apply Hungarian algorithm to each sub-matrix
      for i in 0..<m {
        for j in 0..<m {
          cost = Hungarian_HHB(C, i, j, m);

          L[i*m + j] += cost;
        }
      }

      // apply Hungarian algorithm to the leader matrix
      incre = Hungarian_HHB(L, 0, 0, m);

      if (incre == 0) then
        break;

      lb += incre;
    }

    return lb;
  }

  proc reduceNode(type Node, parent, i, j, k, l, lb_new)
  {
    var child = new Node();
    child.mapping = parent.mapping;
    child.lower_bound = parent.lower_bound;
    child.depth = parent.depth + 1;
    child.available = parent.available;

    child.domCost = parent.domCost;
    child.costs = parent.costs;
    child.domLeader = parent.domLeader;
    child.leader = parent.leader;
    child.size = parent.size - 1;

    // assign q_i to P_j
    child.mapping[i] = j:int(8);

    const n = parent.size;
    const m = n - 1;

    /* assert(n > 0 && "Cannot reduce problem of size 0.");
    assert(std::min(i, j) >= 0 && std::max(i, j) < n && "Invalid reduction indices."); */

    var L_copy = parent.leader;

    child.domCost = {0..<m**4};
    child.domLeader = {0..<m**2};

    var x2, y2, p2, q2: int(32);

    // Updating the leader
    for x in 0..<n {
      if (x == k) then
        continue;

      for y in 0..<n {
        if (y != l) {
          L_copy[x*n + y] += (parent.costs[idx4D(x, y, k, l, n)] + parent.costs[idx4D(k, l, x, y, n)]);
        }
      }
    }

    // reducing the matrix
    x2 = 0;
    for x in 0..<n {
      if (x == k) then
        continue;

      y2 = 0;
      for y in 0..<n {
        if (y == l) then
          continue;

        // copy C_xy into C_x2y2
        p2 = 0;
        for p in 0..<n {
          if (p == k) then
            continue;

          q2 = 0;
          for q in 0..<n {
            if (q == l) then
              continue;

            child.costs[idx4D(x2, y2, p2, q2, m)] = parent.costs[idx4D(x, y, p, q, n)];
            q2 += 1;
          }
          p2 += 1;
        }

        child.leader[x2*m + y2] = L_copy[x*n + y];
        y2 += 1;
      }
      x2 += 1;
    }

    child.available[j] = false;

    child.lower_bound = lb_new;

    return child;
  }

  /*******************************************************
                       GILMORE-LAWLER
  *******************************************************/

  proc Hungarian_GLB(const ref C, n, m)
  {
    var w, j_cur, j_next: int(32);

    // job[j] = worker assigned to job j, or -1 if unassigned
    var job: (sizeMax+1)*int(32);// = allocate(int(32), m+1);
    for i in 0..m do job[i] = -1;

    // yw[w] is the potential for worker w
    // yj[j] is the potential for job j
    var yw: sizeMax*int(32);// = allocate(int(32), n);
    for i in 0..<n do yw[i] = 0;
    var yj: (sizeMax+1)*int(32);// = allocate(int(32), m+1);
    for i in 0..m do yj[i] = 0;

    // main Hungarian algorithm
    for w_cur in 0..<n {
      j_cur = m;                       // dummy job index
      job[j_cur] = w_cur;

      var min_to: (sizeMax+1)*int(32);// = allocate(int(32), m+1);
      for i in 0..m do min_to[i] = INFD2;
      var prv: (sizeMax+1)*int(32);// = allocate(int(32), m+1);
      for i in 0..m do prv[i] = -1;
      var in_Z: (sizeMax+1)*int(32);// = allocate(bool, m+1);
      for i in 0..m do in_Z[i] = false;

      while (job[j_cur] != -1) {
        in_Z[j_cur] = true;
        w = job[j_cur];
        var delta = INFD2;
        j_next = 0;

        for j in 0..<m {
          if !in_Z[j] {
            // reduced cost = C[w][j] - yw[w] - yj[j]
            var cur_cost = C[w*m + j] - yw[w] - yj[j];

            if ckmin(min_to[j], cur_cost) then
              prv[j] = j_cur;
            if ckmin(delta, min_to[j]) then
              j_next = j;
          }
        }

        // update potentials
        for j in 0..m {
          if in_Z[j] {
            yw[job[j]] += delta;
            yj[j] -= delta;
          }
          else {
            min_to[j] -= delta;
          }
        }

        j_cur = j_next;
      }

      // update worker assignment along the found augmenting path
      while (j_cur != m) {
        var j = prv[j_cur];
        job[j_cur] = job[j];
        j_cur = j;
      }

      /* deallocate(min_to);
      deallocate(prv);
      deallocate(in_Z); */
    }

    // compute total cost
    var total_cost: int(32) = 0;

    // for j in [0..m-1], job[j] is the worker assigned to job j
    for j in 0..<m {
      if (job[j] != -1) then
        total_cost += C[job[j]*m + j];
    }

    /* deallocate(job);
    deallocate(yw);
    deallocate(yj); */

    return total_cost;
  }

  record MinPair {
    var min1, min2, idx1: int(32);
  }

  proc Assemble_LAP(const dp, const partial_mapping, const ref av, const ref D,
    const ref F, const n, const N)
  {
    /* var assigned_fac = allocate(int(32), dp);
    var unassigned_fac = allocate(int(32), n-dp);
    var assigned_loc = allocate(int(32), dp);
    var unassigned_loc = allocate(int(32), N-dp); */

    var assigned_fac: sizeMax*int(32);
    var unassigned_fac: sizeMax*int(32);
    var assigned_loc: sizeMax*int(32);
    var unassigned_loc: sizeMax*int(32);

    var c1, c2, c3, c4: int(32) = 0;

    for i in 0..<n {
      if (partial_mapping[i] != -1) {
        assigned_fac[c1] = i;
        c1 += 1;
        assigned_loc[c3] = partial_mapping[i];
        c3 += 1;
      }
      else {
        unassigned_fac[c2] = i;
        c2 += 1;
      }
    }

    for i in 0..<N {
      if av[i] {
        unassigned_loc[c4] = i;
        c4 += 1;
      }
    }

    var u = n - dp;
    var r = N - dp;

    var L: (sizeMax**2)*int(32);
    /* var L: [0..<(u*r)] int(32) = 0; */

    /* record MinPair {
      var min1, min2, idx1: int(32);
    } */

    var best: sizeMax*MinPair; //allocate(MinPair, r);

    for k_idx in 0..<r {
      var k = unassigned_loc[k_idx];
      var min1 = INF;
      var idx1: int(32) = -1;
      var min2 = INF;

      for l_idx in 0..<r {
        if (k_idx == l_idx) then
          continue;

        var l = unassigned_loc[l_idx];
        var dist = D[k * N + l];

        if (dist < min1) {
          min2 = min1;
          min1 = dist;
          idx1 = l_idx;
        }
        else if (dist < min2) {
          min2 = dist;
        }
      }
      best[k_idx] = new MinPair(min1, min2, idx1);
    }

    // Build reduced L-matrix
    for i_idx in 0..<u {
      var i = unassigned_fac[i_idx];

      for k_idx in 0..<r {
        var k = unassigned_loc[k_idx];
        var cost: int(32) = 0;

        // Interaction with other unassigned facilities
        for j_idx in 0..<u {
          var j = unassigned_fac[j_idx];

          if (i == j) then
            continue;

          // Pick best or second-best distance if best is disallowed
          var d = if (best[k_idx].idx1 == k_idx) then best[k_idx].min2 else best[k_idx].min1;

          cost += F[i * n + j] * d;
        }

        // Interaction with assigned facilities
        for a_idx in 0..<dp {
          var j = assigned_fac[a_idx];
          var l = partial_mapping[j];

          cost += F[i * n + j] * D[k * N + l];
        }

        L[i_idx * r + k_idx] = cost;
      }
    }

    /* deallocate(best); */
    /* deallocate(assigned_fac);
    deallocate(unassigned_fac);
    deallocate(assigned_loc);
    deallocate(unassigned_loc); */

    return L;
  }

  proc bound_GLB(const ref node, const ref D, const ref F, const n, const N)
  {
    use CTypes only c_ptrToConst;
    const D_ = c_ptrToConst(D[0]);

    const partial_mapping = node.mapping;
    const av = node.available;
    const dp = node.depth;

    var fixed_cost, remaining_lb: int(32);

    /* local { */
      var L = Assemble_LAP(dp, partial_mapping, av, D_, F, n, N);

      fixed_cost = ObjectiveFunction(partial_mapping, D_, F, n, N);

      remaining_lb = Hungarian_GLB(L, n - dp, N - dp);
    /* } */

    return fixed_cost + remaining_lb;
  }
}
