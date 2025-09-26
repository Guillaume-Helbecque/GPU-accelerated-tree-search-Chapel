[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10786275.svg)](https://doi.org/10.5281/zenodo.10786275)

# GPU-accelerated tree search in Chapel

This repository contains the implementation of a GPU-accelerated tree search algorithm in Chapel.
The latter is instantiated on the backtracking method to solve instances of the N-Queens problem (proof-of-concept) and on the Branch-and-Bound method to solve Taillard's instances of the Permutation Flowshop Scheduling Problem (PFSP).
For comparison purpose, CUDA-based counterpart implementations are also provided.

## Design

The algorithm is based on a general multi-pool approach equipped with a static load balancing mechanism.
Each CPU manages its own pool of work, and we assume that one GPU is assigned per CPU.
The tree exploration starts on the CPU, and each node taken from the work pool is evaluated, potentially pruned, and branched.
In order to exploit GPU-acceleration, we offload a chunk of nodes on the GPU when the pool size is sufficiently large.
When the GPU retrieves the nodes, the latter are evaluated in parallel and the results are sent back to the CPU, which uses them to prune or branch the nodes.
This process is repeated until the pool is empty.

## Implementations

The following Chapel implementations are available:
- `[nqueens/pfsp]_chpl.chpl`: sequential version;
- `[nqueens/pfsp]_gpu_chpl.chpl`: single-GPU version;
- `[nqueens/pfsp]_multigpu_chpl.chpl`: multi-GPU version;
- `[nqueens/pfsp]_dist_multigpu_chpl.chpl`: distributed multi-GPU version.

In addition, the [baselines](./baselines/) directory contains the CUDA-based counterparts:
- `[nqueens/pfsp]_c.c`: sequential version (C);
- `[nqueens/pfsp]_gpu_cuda.cu`: single-GPU version (C+CUDA);
- `[nqueens/pfsp]_multigpu_cuda.cu`: multi-GPU version (C+OpenMP+CUDA) (unstable).

In order to compile and execute the CUDA-based code on AMD GPU architectures, we use the `hipify-perl` tool which translates it into portable HIP C++ automatically.

## Getting started

### Setting the environment configuration

The [chpl_config](./chpl_config/) directory contains several Chapel environment configuration scripts.
The latter can serve as templates and can be (and should be) adapted to the target system.

**Note:** The code is implemented using Chapel 2.6.0 and might not compile and run with older or newer versions.
By default, the target architecture for CUDA code generation is set to `sm_70`, and to `gfx906` for AMD.

### Compilation & execution

All the code is compiled using the provided makefiles.

Common command-line options:
- **`--m`**: minimum number of elements to offload on a GPU device
  - any positive integer (`25` by default)

- **`--M`**: maximum number of elements to offload on a GPU device
  - any positive integer greater than `--m` (`50,000` by default)

- **`--D`**: number of GPU device(s) (only in multi-GPU setting)
  - any positive integer, typically the number of GPU devices (`1` by default)

- **`-nl`**: number of Chapel's locales (only in distributed setting)
  - any positive integer, typically the number of compute nodes

- **`--help`** or **`-h`**: help message

Problem-specific command-line options:
- N-Queens:
  - **`--N`**: number of queens
    - any positive integer (`14` by default)

  - **`--g`**: number of safety check(s) per evaluation
    - any positive integer (`1` by default)

- PFSP:
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

Unstable command-line options:
- **`--perc`**: percentage of the total size of the victim's pool to steal in WS (only in CUDA-based multi-GPU implementation)
  - any real number between `0.0` and `1.0` (`0.5` by default)

### Examples

- Chapel single-GPU launch to solve the 15-Queens instance:
```
./nqueens_gpu_chpl.out --N 15
```

- CUDA multi-GPU launch to solve the 17-Queens instance using 4 GPU devices:
```
./nqueens_multigpu_cuda.out -N 17 -D 4
```

## Related publications

1. G. Helbecque. *PGAS-based Parallel Branch-and-Bound for Ultra-Scale GPU-powered Supercomputers*. Ph.D. thesis. Université de Lille, Université du Luxembourg. 2025. URL: https://theses.fr/2025ULILB003.
2. G. Helbecque, E. Krishnasamy, T. Carneiro, N. Melab, and P. Bouvry. A Chapel-Based Multi-GPU Branch-and-Bound Algorithm. *Euro-Par 2024: Parallel Processing Workshops*, Madrid, Spain, 2025, pp. 463-474. DOI: [10.1007/978-3-031-90200-0_37](https://doi.org/10.1007/978-3-031-90200-0_37).
3. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU-Accelerated Tree-Search in Chapel versus CUDA and HIP. *2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, San Francisco, USA, 2024, pp. 872-879. DOI: [10.1109/IPDPSW63119.2024.00156](https://doi.org/10.1109/IPDPSW63119.2024.00156).
4. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU Computing in Chapel: Application to Tree-Search Algorithms. *International Conference in Optimization and Learning (OLA 2024)*, Dubrovnik, Croatia, 2024.
