# GPU-accelerated tree-search in Chapel: Comparing against CUDA and HIP on Nvidia and AMD GPUs

This repository contains the implementation of a GPU-accelerated tree-search
algorithm in Chapel.
The latter supports any tree-based problems, but is first instantiated on the
backtracking method to solve instances of the N-Queens problem.
This well-known problem serves as a proof-of-concept that motivates further
improvements in solving combinatorial optimization problems.

## Design

The algorithm is based on a general multi-pool approach equipped with a static load
balancing mechanism.
The tree exploration starts on the CPU, and each node taken from the work pool is
evaluated, potentially pruned, and branched.
This process is repeated until the pool is empty.
In order to exploit GPU-acceleration, we offload a chunk of nodes on the GPU when the
pool size is sufficiently large. When the GPU retrieves the nodes, the latter are
then evaluated in parallel, and the non promising nodes are labelled.
Finally, the array of labels is sent back to the CPU, which uses it to prune and
branch.

## Implementation

The following Chapel implementations are available from the main directory:
- `nqueens_chpl.chpl`: sequential version of the algorithm;
- `nqueens_gpu_chpl.chpl`: single-GPU version of the algorithm;
- `nqueens_multigpu_chpl.chpl`: multi-GPU version of the algorithm.

In addition, the [baselines](./baselines/) directory contains the CUDA-based counterparts:
- `nqueens_c.chpl`: sequential C version of the algorithm;
- `nqueens_gpu_cuda.chpl`: single-GPU version of the algorithm;
- `nqueens_multigpu_cuda.chpl`: multi-GPU version of the algorithm.

In order to compile and execute these codes on both Nvidia and AMD GPU architectures,
we use the `hipify-perl` tool, which translates our CUDA-based source code into
portable HIP C++ automatically.

## Getting started

### Setting the Chapel environment

The [chpl_config](./chpl_config/) directory contains several Chapel environment
configuration scripts.
The latter can be adapted to different systems.

### Compilation & execution
The `makefile`s produce all the executables in the following format:
```
nqueens_[gpu/multigpu]_[chpl/cuda/hip].o
```

The translation from CUDA to HIP is automatically managed.

**Note:** By default, the target architecture for CUDA code generation is set to
`sm_70`, and to `gfx906` for AMD.
Take care to adjust these settings in the `makefile`s if needed.

### Command-line options
The following command-line options are supported:
- `N`: number of queens (default: 14);
- `g`: number of safety check(s) per evaluation (default: 1);
- `m`: minimum number of elements to offload on GPU devices (default: 25);
- `M`: maximum number of elements to offload on GPU devices (default: 50,000);
- `D`: number of GPU devices (only in multi-GPU setting - default: 1).

All these values must be positive integers.

### Examples

- Chapel single-GPU launch to solve the 15-Queens instances:
```
./nqueens_gpu_chpl.o --N 15
```

- CUDA multi-GPU launch to solve the 17-Queens instances using 4 GPU devices:
```
./nqueens_multigpu_cuda.o -N 17 -D 4
```
