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
    var size: int;

    proc init(type elemType) {
      this.eltType = elemType;
      this.dom = 0..#CAPACITY;
      this.capacity = CAPACITY;
    }

    proc ref pushBack(node: eltType) {
      if (this.size >= this.capacity) {
        this.capacity *=2;
        this.dom = 0..#this.capacity;
      }

      this.elements[this.size] = node;
      this.size += 1;
    }

    proc ref popBack(ref hasWork: int) {
      if (this.size > 0) {
        hasWork = 1;
        this.size -= 1;
        return this.elements[this.size];
      }

      var default: eltType;
      return default;
    }
  }
}
