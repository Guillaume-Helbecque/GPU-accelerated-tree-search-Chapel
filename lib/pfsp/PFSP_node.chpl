/*******************************************************************************
Implementation of PFSP Nodes.
*******************************************************************************/

module PFSP_node
{
  config param MAX_JOBS = 20;

  record Node {
    var depth: int;
    var limit1: int; // left limit
    var prmu: MAX_JOBS*int;

    // default-initializer
    proc init() {}

    // root-initializer
    proc init(jobs)
    {
      this.limit1 = -1;
      init this;
      for i in 0..#jobs do this.prmu[i] = i;
    }

    /*
      NOTE: This copy-initializer makes the Node type "non-trivial" for `noinit`.
      Perform manual copy in the code instead.
    */
    // copy-initializer
    /* proc init(other: Node)
    {
      this.depth  = other.depth;
      this.limit1 = other.limit1;
      this.prmu   = other.prmu;
    } */
  }
}
