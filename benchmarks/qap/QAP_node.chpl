/*******************************************************************************
Implementation of QAP Nodes.
*******************************************************************************/

module QAP_node
{
  use Util_qap;

  config param sizeMax: int(32) = 32;

  // ISSUE: cannot use `noinit` with this node type.
  record Node_RLT1
  {
    var mapping: sizeMax*int(8);
    var lower_bound: int;
    var depth: uint(8);
    var available: sizeMax*bool;

    var domCost: domain(1, idxType = int(32));
    var costs: [domCost] int;
    var domLeader: domain(1, idxType = int(32));
    var leader: [domLeader] int;
    var size: int(32);

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(const n, const N, const ref D, const ref F)
    {
      init this;
      for i in 0..<n do this.mapping[i] = -1:int(8);
      for i in 0..<sizeMax do this.available[i] = true;
      /* this.available = true; */

      this.domCost = {0..<(N**4)};
      this.domLeader = {0..<(N**2)};
      this.size = N;
      Assemble(this.costs, this.leader, D, F, n, N);
    }

    // copy-initializer
    /* proc init(other: Node_RLT1)
    {
      this.mapping = other.mapping;
      this.lower_bound = other.lower_bound;
      this.depth = other.depth;
      this.available = other.available;

      this.domCost = other.domCost;
      this.costs = other.costs;
      this.domLeader = other.domLeader;
      this.leader = other.leader;
      this.size = other.size;
    } */
  }

  record Node_GLB
  {
    var mapping: sizeMax*int(8);
    var depth: uint(8);
    var available: sizeMax*bool;

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(const n)
    {
      init this;
      for i in 0..<n do this.mapping[i] = -1:int(8);
      for i in 0..<sizeMax do this.available[i] = true;
      /* this.available = true; */
    }

    // copy-initializer
    /* proc init(other: Node_GLB)
    {
      this.mapping = other.mapping;
      this.depth = other.depth;
      this.available = other.available;
    } */
  }

  /*
    Node_QPB: per-node state for the batched GPU QPB B&B.

    Fields beyond the standard B&B triple (mapping / available / depth)
    hold the post-bound state that the GPU kernel writes and the host
    re-reads once the node becomes a parent: variable fixing reads
    `qpbReducedCostsRow`, warm-start reads the primal X at `qpbXSlot`
    inside the persistent device-side qpbX pool.
  */
  record Node_QPB
  {
    // Standard B&B fields.
    var mapping: sizeMax*int(8);
    var available: sizeMax*bool;
    var depth: uint(8);

    // Lifecycle flag: true once this node has been bounded by QPB and the
    // *post-bound state below is valid.
    var hasBoundData: bool;

    // Index into the persistent device-side qpbX pool. -1 means "no slot",
    // i.e. root, never-bounded, or already returned to the free list.
    var qpbXSlot: int(32);

    // One row of the dual reduced-costs matrix (the row indexed by this
    // node's own next-branching facility once it becomes a parent). Stored
    // in float; host casts to double for the VF check.
    var qpbReducedCostsRow: sizeMax*real(32);

    // Continuous QPB bound at the last FW iter (already corrected for
    // asymmetry); fixed integer cost at this node.
    var qpbBoundContLast: real(64);
    var qpbFixedCost: int;

    // Push-time lower bound. Re-checked against UB before this node is
    // dispatched as a parent (UB may have tightened since enqueue).
    var lb: int;

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(const n)
    {
      init this;
      for i in 0..<n do this.mapping[i] = -1:int(8);
      for i in 0..<sizeMax do this.available[i] = true;
      this.depth = 0;
      this.hasBoundData = false;
      this.qpbXSlot = -1;
      this.qpbBoundContLast = 0.0;
      this.qpbFixedCost = 0;
      this.lb = 0;
    }
  }

  proc Assemble(ref costs, ref leader, const ref D, const ref F, const n, const N)
  {
    for i in 0..<N {
      for j in 0..<N {
        for k in 0..<N {
          for l in 0..<N {
            if ((k == i) ^ (l == j)) then
              costs[idx4D(i, j, k, l, N)] = INFD2;
            else if (k < n) then
              costs[idx4D(i, j, k, l, N)] = F[i * n + k] * D[j * N + l];
            else
              costs[idx4D(i, j, k, l, N)] = 0;
          }
        }
        leader[i*N + j] = costs[idx4D(i, j, i, j, N)];
      }
    }
  }
}
