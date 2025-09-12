/*******************************************************************************
Implementation of Qubit Allocation Nodes.
*******************************************************************************/

module QubitAlloc_node
{
  use Util_qubitAlloc;

  config param sizeMax: int(32) = 27;

  // ISSUE: cannot use `noinit` with this node type.
  record Node
  {
    var mapping: sizeMax*int(32);
    var lower_bound: int(32);
    var depth: int(32);
    var available: [0..<sizeMax] bool;

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
      this.available = true;

      this.domCost = {0..<(N**4)};
      this.domLeader = {0..<(N**2)};
      this.size = N;
      Assemble(D, F, N);
    }

    // copy-initializer
    /* proc init(other: Node)
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

    proc deinit()
    {}

    proc ref Assemble(D, F, N)
    {
      for i in 0..<N {
        for j in 0..<N {
          for k in 0..<N {
            for l in 0..<N {
              if ((k == i) ^ (l == j)) then
                this.costs[idx4D(i, j, k, l, N)] = INFD2;
              else
                this.costs[idx4D(i, j, k, l, N)] = F[i, k] * D[j, l];
            }
          }
          this.leader[i*N + j] = this.costs[idx4D(i, j, i, j, N)];
        }
      }
    }
  }
}
