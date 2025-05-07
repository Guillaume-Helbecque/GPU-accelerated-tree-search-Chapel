module Pool_par
{
  use Math;

  /*******************************************************************************
  Extension of the "Pool" data structure ensuring parallel-safety and supporting
  bulk operations.
  *******************************************************************************/

  config param INITIAL_CAPACITY = 1024;

  record SinglePool_par {
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
  }
}
