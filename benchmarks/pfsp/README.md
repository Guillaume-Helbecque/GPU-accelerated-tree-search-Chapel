# The Permutation Flowshop Scheduling Problem (PFSP)

### Formulation

The problem consists in finding an optimal processing order (a permutation) for $n$ jobs on $m$ machines, such that the completion time of the last job on the last machine (makespan) is minimized. The commonly used Taillard's [1] instances are supported as test-cases.

### Configuration options

```
./main_pfsp.out {...}
```

where the available options are:
- **`--inst`**: Taillard's instance to solve
  - any positive integer between `001` and `120` (`014` by default)

<!-- TODO: give references -->
- **`--lb`**: lower bound function
  - `lb1`: one-machine bound which can be computed in $\mathcal{O}(mn)$ steps per subproblem (default)
  - `lb1_d`: fast implementation of `lb1`, which can be compute in $\mathcal{O}(m)$ steps per subproblem
  - `lb2`: two-machine bound which can be computed in $\mathcal{O}(m^2n)$ steps per subproblem
  <!-- a two-machine bound which relies on the exact resolution of two-machine problems obtained by relaxing capacity constraints on all machines, with the exception of a pair of machines \(M<sub>u</sub>,M<sub>v</sub>\)<sub>1<=u<v<=m</sub>, and taking the maximum over all $\frac{m(m-1)}{2}$ machine-pairs. It can be computed in $\mathcal{O}(m^2n)$ steps per subproblem. -->

- **`--ub`**: initial upper bound (UB)
  - `0`: initialize the UB to $+\infty$, leading to a search from scratch
  - `1`: initialize the UB to the best solution known (default)

### References

1. E. Taillard. (1993) Benchmarks for basic scheduling problems. *European Journal of Operational Research*, 64(2):278-285. DOI: [10.1016/0377-2217(93)90182-M](https://doi.org/10.1016/0377-2217(93)90182-M).
