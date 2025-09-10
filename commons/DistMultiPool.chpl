module DistMultiPool
{
  private use IO;

  use Math;

  config param INITIAL_CAPACITY = 1024;

  /*
    Reference counter for DistributedBag_DFS.
  */
  class DistributedBagRC
  {
    type eltType;
    var _pid: int;

    proc deinit()
    {
      coforall loc in Locales do on loc {
        delete chpl_getPrivatizedCopy(unmanaged DistributedBagImpl(eltType), _pid);
      }
    }
  }

  /*
    A parallel-safe distributed multi-pool implementation specialized for depth-first
    search (DFS), that scales in terms of nodes, processors per node (PPN), and workload;
    the more PPN, the more segments we allocate to increase raw parallelism, and the
    larger the workload the better locality (see :const:`distributedBagInitialSegmentCap`).
    This data structure is locally DFS ordered, encapsulates a dynamic work stealing
    mechanism to balance work across nodes, and provides a means to obtain a privatized
    instance of the data structure for maximized performance.
  */
  pragma "always RVF"
  record distBag_DFS
  {
    /*
      The type of the elements contained in this distBag_DFS.
    */
    type eltType;

    // Privatized id
    var _pid: int = -1;

    // Reference Counting
    var _rc: shared DistributedBagRC(eltType);

    proc init(type eltType, targetLocales = Locales)
    {
      this.eltType = eltType;
      this._pid = (new unmanaged DistributedBagImpl(eltType, targetLocales = targetLocales)).pid;
      this._rc = new shared DistributedBagRC(eltType, _pid = _pid);
    }

    inline proc _value
    {
      if (_pid == -1) then halt("distBag_DFS is uninitialized.");
      return chpl_getPrivatizedCopy(unmanaged DistributedBagImpl(eltType), _pid);
    }

    forwarding _value;
  } // end 'distBag_DFS' record

  class DistributedBagImpl
  {
    /*
      The type of element that this Collection holds.
    */
    type eltType;

    var targetLocDom: domain(1);

    /*
      The locales to allocate bags for and load balance across. ``targetLocDom``
      represents the corresponding range of locales.
    */
    var targetLocales: [targetLocDom] locale;

    var pid: int = -1;

    // Node-local fields below. These fields are specific to the privatized bag
    // instance. To access them from another node, make sure you use
    // 'getPrivatizedThis'.
    var bag: unmanaged Bag(eltType)?;

    /*
      Initialize an empty distBag_DFS.
    */
    proc init(type eltType, targetLocales: [?targetLocDom] locale = Locales)
    {
      this.eltType = eltType;
      this.targetLocDom  = targetLocDom;
      this.targetLocales = targetLocales;

      init this;

      this.pid = _newPrivatizedClass(this);
      this.bag = new unmanaged Bag(eltType, this);
    }

    proc init(other, pid, type eltType = other.eltType)
    {
      this.eltType = eltType;
      this.targetLocDom  = other.targetLocDom;
      this.targetLocales = other.targetLocales;
      this.pid           = pid;

      init this;

      this.bag = new unmanaged Bag(eltType, this);
    }

    proc deinit()
    {
      delete bag;
    }

    proc dsiPrivatize(pid)
    {
      return new unmanaged DistributedBagImpl(this, pid);
    }

    proc dsiGetPrivatizeData()
    {
      return pid;
    }

    inline proc getPrivatizedThis
    {
      return chpl_getPrivatizedCopy(this.type, pid);
    }

    proc pushBack(elt: eltType, taskId: int)
    {
      bag!.pushBack(elt, taskId);
    }

    proc pushBackFree(elt: eltType, taskId: int)
    {
      bag!.pushBackFree(elt, taskId);
    }

    proc pushBackBulk(elts, taskId: int)
    {
      bag!.pushBackBulk(elts, taskId);
    }

    proc popBack(ref hasWork, taskId: int)
    {
      return bag!.popBack(hasWork, taskId);
    }

    proc popBackFree(hasWork, taskId: int)
    {
      return bag!.popBackFree(hasWork, taskId);
    }

    proc popBackBulk(const m: int, const M: int, ref parents, taskId: int)
    {
      return bag!.popBackBulk(m, M, parents, taskId);
    }

    proc popBackBulkFree(const m: int, const M: int, taskId: int)
    {
      return bag!.popBackBulkFree(m, M, taskId);
    }
  } // end 'DistributedBagImpl' class

  /*
    We maintain a multi-pool 'bag' per locale. Each bag keeps a handle to its parent,
    which is required for work stealing.
  */
  class Bag
  {
    type eltType;

    // A handle to our parent 'distributed' bag, which is needed for work stealing.
    var parentHandle: borrowed DistributedBagImpl(eltType);

    var segments: [0..#here.maxTaskPar] Segment(eltType);

    var globalStealInProgress: atomic bool = false;

    proc init(type eltType, parentHandle)
    {
      this.eltType = eltType;
      this.parentHandle = parentHandle;
      // KNOWN ISSUE: 'init this' produces an error when 'eltType' is a Chapel
      // array (see Github issue #19859).
    }

    proc pushBack(elt: eltType, taskId: int)
    {
      segments[taskId].pushBack(elt);
    }

    proc pushBackFree(elt: eltType, taskId: int)
    {
      segments[taskId].pushBackFree(elt);
    }

    proc pushBackBulk(elts, taskId: int)
    {
      segments[taskId].pushBackBulk(elts);
    }

    proc popBack(ref hasWork, taskId: int)
    {
      return segments[taskId].popBack(hasWork);
    }

    proc popBackFree(hasWork, taskId: int)
    {
      return segments[taskId].popBackFree(hasWork);
    }

    proc popBackBulk(const m: int, const M: int, ref parents, taskId: int)
    {
      return segments[taskId].popBackBulk(m, M, parents);
    }

    proc popBackBulkFree(const m: int, const M: int, taskId: int)
    {
      return segments[taskId].popBackBulkFree(m, M);
    }
  } // end 'Bag' class

  /*
    A Segment is a parallel-safe pool, implemented as a non-blocking split deque
    (see header). In few words, it is a buffer of memory, called Block, along with
    some logic to ensure parallel-safety.
  */
  record Segment
  {
    type eltType;
    var dom: domain(1);
    var elements: [dom] eltType;
    var capacity: int;
    var front: int;
    var size: int;
    var lock: atomic bool;

    proc init(type eltType) {
      this.eltType = eltType;
      this.dom = {0..#INITIAL_CAPACITY};
      this.capacity = INITIAL_CAPACITY;
      this.lock = false;
    }

    proc ref acquireLock() {
      while true {
        if this.lock.compareAndSwap(false, true) {
          return;
        }

        currentTask.yieldExecution();
      }
    }

    proc ref releaseLock() {
      this.lock.write(false);
    }

    // Parallel-safe insertion to the end of the deque.
    proc ref pushBack(node: eltType) {
      while true {
        if this.lock.compareAndSwap(false, true) {
          if (this.front + this.size >= this.capacity) {
            this.capacity *= 2;
            this.dom = {0..#this.capacity};
          }

          this.elements[this.front + this.size] = node;
          this.size += 1;
          this.lock.write(false);
          return;
        }

        currentTask.yieldExecution();
      }
    }

    // Insertion to the end of the deque. Parallel-safety is not guaranteed.
    proc ref pushBackFree(node: eltType) {
      if (this.front + this.size >= this.capacity) {
        this.capacity *= 2;
        this.dom = {0..#this.capacity};
      }

      this.elements[this.front + this.size] = node;
      this.size += 1;
    }

    // Parallel-safe bulk insertion to the end of the deque.
    proc ref pushBackBulk(nodes: [] eltType) {
      const s = nodes.size;

      while true {
        if this.lock.compareAndSwap(false, true) {
          if (this.front + this.size + s >= this.capacity) {
            this.capacity *= 2**ceil(log2((this.front + this.size + s) / this.capacity:real)):int;
            this.dom = {0..#this.capacity};
          }

          this.elements[(this.front + this.size)..#s] = nodes;
          this.size += s;
          this.lock.write(false);
          return;
        }

        currentTask.yieldExecution();
      }
    }

    // Parallel-safe removal from the end of the deque.
    proc ref popBack(ref hasWork: int) {
      while true {
        if this.lock.compareAndSwap(false, true) {
          if (this.size > 0) {
            hasWork = 1;
            this.size -= 1;
            var elt = this.elements[this.front + this.size];
            this.lock.write(false);
            return elt;
          }
          else {
            this.lock.write(false);
            break;
          }
        }

        currentTask.yieldExecution();
      }

      var default: eltType;
      return default;
    }

    // Removal from the end of the deque. Parallel-safety is not guaranteed.
    proc ref popBackFree(ref hasWork: int) {
      if (this.size > 0) {
        hasWork = 1;
        this.size -= 1;
        return this.elements[this.front + this.size];
      }

      var default: eltType;
      return default;
    }

    // Parallel-safe bulk removal from the end of the deque.
    proc ref popBackBulk(const m: int, const M: int, ref parents) {
      while true {
        if this.lock.compareAndSwap(false, true) {
          if (this.size < m) {
            this.lock.write(false);
            break;
          }
          else {
            const poolSize = min(this.size, M);
            this.size -= poolSize;
            parents[0..#poolSize] = this.elements[(this.front + this.size)..#poolSize];
            this.lock.write(false);
            return poolSize;
          }
        }

        currentTask.yieldExecution();
      }

      return 0;
    }

    // Bulk removal from the end of the deque. Parallel-safety is not guaranteed.
    proc ref popBackBulkFree(const m: int, const M: int) {
      if (this.size >= 2*m) {
        const poolSize = this.size/2; //min(this.size, M);
        this.size -= poolSize;
        var parents: [0..#poolSize] eltType = this.elements[(this.front + this.size)..#poolSize];
        return (poolSize, parents);
      }

      var parents: [0..-1] eltType = noinit;
      return (0, parents);
    }

    // Removal from the front of the deque. Parallel-safety is not guaranteed.
    proc ref popFrontFree(ref hasWork: int) {
      if (this.size > 0) {
        hasWork = 1;
        const elt = this.elements[this.front];
        this.front += 1;
        this.size -= 1;
        return elt;
      }

      var default: eltType;
      return default;
    }

    // Bulk removal from the front of the deque. Parallel-safety is not guaranteed.
    proc ref popFrontBulkFree(const m: int, const M: int) {
      if (this.size >= 2*m) {
        const poolSize = this.size/2; //min(this.size, M);
        this.size -= poolSize;
        var parents: [0..#poolSize] eltType = this.elements[this.front..#poolSize];
        this.front += poolSize;
        return (poolSize, parents);
      }

      var parents: [0..-1] eltType = noinit;
      return (0, parents);
    }
  } // end 'Segment' record
} // end module
