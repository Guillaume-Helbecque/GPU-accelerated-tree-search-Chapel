[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10786275.svg)](https://doi.org/10.5281/zenodo.10786275)

# GPU-accelerated tree search in Chapel

This repository presents a generic, problem-independent GPU-accelerated tree search framework implemented in Chapel.
The approach relies on a general multi-pool design with static load balancing, where each CPU manages its own work pool and is associated with one GPU.
Tree exploration begins on the CPU: nodes are taken from the pool, evaluated, potentially pruned, and branched.
When the pool grows large enough, a chunk of nodes is offloaded to the GPU, where they are evaluated in parallel and the results are returned to the CPU to guide pruning and branching.
The framework supports multiple execution modes, including sequential, single-GPU, multi-GPU, and distributed multi-GPU configurations.

### Prerequisites

[Chapel](https://chapel-lang.org/) 2.4.0

The [chpl_config](./chpl_config/) directory contains predefined shell scripts for downloading, configuring, and building the Chapel compiler from source.

**Note:** The code might not compile and run with older or newer versions of Chapel.

### Compilation and configuration options

Compile with `make` and execute with:

```
./main.out {...}
```

where the available options are:
- **`--mode`**: parallel execution mode
  - `sequential`: single-core execution, without parallel feature
  - `gpu`: single-node single-GPU execution
  - `multigpu`: single-node multi-GPU execution (default)
  - `distributed`: multi-node multi-GPU execution

- **`--m`**: minimum number of elements to offload on a GPU device
  - any positive integer (`25` by default)

- **`--M`**: maximum number of elements to offload on a GPU device
  - any positive integer greater than `--m` (`50,000` by default)

- **`--D`**: number of GPU device(s) (only in multi-GPU settings)
  - any positive integer, typically the number of GPU devices (`1` by default)

- **`-nl`**: number of Chapel's locales (only in distributed setting)
  - any positive integer, typically the number of compute nodes

- **`--help`** or **`-h`**: help message

Unstable command-line options:
- **`--perc`**: percentage of the total size of the victim's pool to steal in WS (only in CUDA-based multi-GPU implementation)
  - any real number between `0.0` and `1.0` (`0.5` by default)

Other problem-specific options are supported; see next section.

### Supported problems

The B&B skeletons have already been tested on the following benchmark problems:
- [The Permutation Flowshop Scheduling problem](./benchmarks/pfsp) (PFSP)
- [The N-Queens problem](./benchmarks/nqueens)

Supported execution modes:

| benchmark | sequential         | single-GPU         | multi-GPU          | distributed multi-GPU |
|-----------|--------------------|--------------------|--------------------|-----------------------|
| PFSP      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:    |
| N-Queens  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:    |

In addition, the [baselines](./baselines/) directory contains some CUDA-based counterpart implementations for comparison purposes.

**Note:** In order to compile and execute the CUDA-based codes on AMD GPU architectures, we use the `hipify-perl` tool which translates it into portable HIP C++ automatically.

## Related publications

1. G. Helbecque. *PGAS-based Parallel Branch-and-Bound for Ultra-Scale GPU-powered Supercomputers*. Ph.D. thesis. Université de Lille, Université du Luxembourg. 2025. URL: https://theses.fr/2025ULILB003.
2. G. Helbecque, E. Krishnasamy, T. Carneiro, N. Melab, and P. Bouvry. A Chapel-Based Multi-GPU Branch-and-Bound Algorithm. *Euro-Par 2024: Parallel Processing Workshops*, Madrid, Spain, 2025, pp. 463-474. DOI: [10.1007/978-3-031-90200-0_37](https://doi.org/10.1007/978-3-031-90200-0_37).
3. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU-Accelerated Tree-Search in Chapel versus CUDA and HIP. *2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, San Francisco, USA, 2024, pp. 872-879. DOI: [10.1109/IPDPSW63119.2024.00156](https://doi.org/10.1109/IPDPSW63119.2024.00156).
4. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU Computing in Chapel: Application to Tree-Search Algorithms. *International Conference in Optimization and Learning (OLA 2024)*, Dubrovnik, Croatia, 2024.
