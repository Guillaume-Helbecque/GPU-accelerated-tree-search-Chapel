module Pool
{
  /*******************************************************************************
  Implementation of a dynamic-sized single pool data structure.
  Its initial capacity is 1024, and we reallocate a new container with double
  the capacity when it is full. Since we perform only DFS, it only supports
  'pushBack' and 'popBack' operations.
  *******************************************************************************/

  config param CAPACITY = 1024;

  record SinglePool {
    type eltType;
    var dom: domain(1);
    var elements: [dom] eltType;
    var capacity: int;
    var front: int;
    var size: int;

    proc init(type eltType) {
      this.eltType = eltType;
      this.dom = 0..#CAPACITY;
      this.capacity = CAPACITY;
    }

    proc ref pushBack(node: eltType) {
      if (this.front + this.size >= this.capacity) {
        this.capacity *= 2;
        this.dom = 0..#this.capacity;
      }

      this.elements[this.front + this.size] = node;
      this.size += 1;
    }

    proc ref popBack(ref hasWork: int) {
      if (this.size > 0) {
        hasWork = 1;
        this.size -= 1;
        return this.elements[this.front + this.size];
      }

      var default: eltType;
      return default;
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
