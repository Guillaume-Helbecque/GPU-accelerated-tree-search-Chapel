module Pool_ext
{
  /*******************************************************************************
  Implementation of a dynamic-sized single pool data structure.
  Its initial capacity is 1024, and we reallocate a new container with double
  the capacity when it is full. Since we perform only DFS, it only supports
  'pushBack' and 'popBack' operations.
  *******************************************************************************/

  config param CAPACITY = 1024000;

  record SinglePool_ext {
    type eltType;
    var dom: domain(1);
    var elements: [dom] eltType;
    var capacity: int;
    var front: int;
    var size: int;
    var lock: atomic bool;

    proc init(type eltType) {
      this.eltType = eltType;
      this.dom = {0..#CAPACITY};
      this.capacity = CAPACITY;
      this.lock = false;
    }

    proc ref pushBack(node: eltType) {
      while true {
        if this.lock.compareAndSwap(false, true) {
          if (this.front + this.size >= this.capacity) {
            this.capacity *= 2;
            this.dom = 0..#this.capacity;
          }

          this.elements[this.front + this.size] = node;
          this.size += 1;
          this.lock.write(false);
          return;
        }

        currentTask.yieldExecution();
      }
    }

    proc ref pushBackBulk(nodes: [] eltType) {
      const s = nodes.size;

      while true {
        if this.lock.compareAndSwap(false, true) {
          if (this.front + this.size >= this.capacity) {
            this.capacity *= 2*ceil(log2((this.front + this.size + s) / this.capacity));
            this.dom = 0..#this.capacity;
          }

          this.elements[(this.front + this.size)..#s] = nodes;
          this.size += s;
          this.lock.write(false);
          return;
        }

        currentTask.yieldExecution();
      }
    }

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

    proc ref popBackFree(ref hasWork: int) {
      if (this.size > 0) {
        hasWork = 1;
        this.size -= 1;
        return this.elements[this.front + this.size];
      }

      var default: eltType;
      return default;
    }

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

    proc ref popBackBulkFree(const m: int, const M: int) {
      if (this.size >= 2*m) {
        const poolSize = this.size/2; //min(this.size, M);
        this.size -= poolSize;
        var parents: [0..#poolSize] eltType = this.elements[(this.front + this.size)..#poolSize];
        return (poolSize, parents);
      } else {
        halt("DEADCODE");
      }

      var parents: [0..-1] eltType = noinit;
      return (0, parents);
    }

    proc ref popFront(ref hasWork: int) {
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
  }
}
