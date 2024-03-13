# GPU-accelerated tree-search in Chapel

This repository contains the implementation of a GPU-accelerated tree-search algorithm in Chapel.
The latter is instantiated on the backtracking method to solve instances of the N-Queens problem (proof-of-concept) and on the Branch-and-Bound method to solve Taillard's instances of the permutation flowshop scheduling problem (PFSP).
For comparison purpose, CUDA-based counterpart implementations are also provided.

## Design

The algorithm is based on a general multi-pool approach equipped with a static load balancing mechanism.
Each CPU manages its own pool of work, and we assume that one GPU is assigned per CPU.
The tree exploration starts on the CPU, and each node taken from the work pool is evaluated, potentially pruned, and branched.
In order to exploit GPU-acceleration, we offload a chunk of nodes on the GPU when the pool size is sufficiently large.
When the GPU retrieves the nodes, the latter are evaluated in parallel and the non promising nodes are labeled.
Then, the array of labels is sent back to the CPU, which uses it to prune and branch.
This process is repeated until the pool is empty.

## Implementation

The following Chapel implementations are available from the main directory:
- `[nqueens/pfsp]_chpl.chpl`: sequential version;
- `[nqueens/pfsp]_gpu_chpl.chpl`: single-GPU version;
- `[nqueens/pfsp]_multigpu_chpl.chpl`: multi-GPU version.

In addition, the [baselines](./baselines/) directory contains the CUDA-based counterparts:
- `[nqueens/pfsp]_c.c`: sequential version (C);
- `nqueens_gpu_cuda.cu`: single-GPU version (C+CUDA);
- `nqueens_multigpu_cuda.cu`: multi-GPU version (C+OpenMP+CUDA).

In order to compile and execute the CUDA-based code on AMD GPU architectures, we use the `hipify-perl` tool which translates it into portable HIP C++ automatically.

**Note**: The PFSP instantiation for the CUDA-based counterparts is not yet supported.

## Getting started

### Setting the environment configuration

The [chpl_config](./chpl_config/) directory contains several Chapel environment configuration scripts.
The latter can serve as templates and can be (and should be) adapted to the target system.

**Note:** The code is implemented using Chapel 1.33.0 and might not compile and run with older or newer versions.
By default, the target architecture for CUDA code generation is set to `sm_70`, and to `gfx906` for AMD.

### Compilation & execution

All the code is compiled using the provided makefiles.

Common command-line options:
- `m`: minimum number of elements to offload on a GPU device (default: 25);
- `M`: maximum number of elements to offload on a GPU device (default: 50,000);
- `D`: number of GPU device(s) (only in multi-GPU setting - default: 1).

Problem-specific command-line options:
- N-Queens:
    - `N`: number of queens (default: 14);
    - `g`: number of safety check(s) per evaluation (default: 1);
- PFSP:
    - `inst`: Taillard's instance index (1 to 120 - default: 14);
    - `lb`: lower bound function (0 (lb1_d), 1 (lb1), or 2 (lb2) - default: 1);
    - `ub`: upper bound initialization (0 (inf) or 1 (opt) - default: 1).

### Examples

- Chapel single-GPU launch to solve the 15-Queens instance:
```
./nqueens_gpu_chpl.o --N 15
```

- CUDA multi-GPU launch to solve the 17-Queens instance using 4 GPU devices:
```
./nqueens_multigpu_cuda.o -N 17 -D 4
```
