/*******************************************************************************
Implementation of Qubit Allocation Nodes.
*******************************************************************************/

module QubitAlloc_node
{
  use Util_qubitAlloc;

  config param sizeMax: int(32) = 27;

  // ISSUE: cannot use `noinit` with this node type.
  record Node_HHB
  {
    var mapping: sizeMax*int(8);
    var lower_bound: int(32);
    var depth: uint(8);
    var available: sizeMax*bool;

    var domCost: domain(1, idxType = int(32));
    var costs: [domCost] int(32);
    var domLeader: domain(1, idxType = int(32));
    var leader: [domLeader] int(32);
    var size: int(32);

    // default-initializer
    proc init()
    {}

    // root-initializer
    proc init(const n, const N, const ref D, const ref F)
    {
      init this;
      for i in 0..<n do this.mapping[i] = -1:int(32);
      for i in 0..<sizeMax do this.available[i] = true;
      /* this.available = true; */

      this.domCost = {0..<(N**4)};
      this.domLeader = {0..<(N**2)};
      this.size = N;
      Assemble(this.costs, this.leader, D, F, n, N);
    }

    // copy-initializer
    /* proc init(other: Node_HHB)
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

  proc Assemble(ref costs, ref leader, const ref D, const ref F, const n, const N)
  {
    for i in 0..<n {
      for j in 0..<N {
        for k in 0..<n {
          for l in 0..<N {
            if ((k == i) ^ (l == j)) then
              costs[idx4D(i, j, k, l, N)] = INFD2;
            else
              costs[idx4D(i, j, k, l, N)] = F[i * n + k] * D[j * N + l];
          }
        }
        leader[i*N + j] = costs[idx4D(i, j, i, j, N)];
      }
    }
  }
}
