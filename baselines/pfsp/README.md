# Building code

The provided `makefile` builds all CUDA-based codes as well as HIP-based ones (using the ROCm/HIP `hipify-perl` tool).

The `SYSTEM` command-line option can be set to `{g5k, lumi}` to handle manually the system specific library paths etc. It defaults to `g5k`. For the moment, the following systems are supported:
- the [Grid5000](https://www.grid5000.fr/w/Grid5000:Home) large-scale testbed;
- the [LUMI](https://docs.lumi-supercomputer.eu/) pre-exascale supercomputer.
