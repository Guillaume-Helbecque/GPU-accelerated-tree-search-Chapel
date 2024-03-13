#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH -p gpu
#SBATCH -G 4

# Configuration of Chapel for (multi-)GPU-accelerated experiments on the Iris
# cluster of the Université du Luxembourg (https://hpc-docs.uni.lu/systems/iris/).

# Load the foss toolchain to get access to gcc, mpi, etc...
module load toolchain/foss/2020b
module load system/CUDA/11.1
module load devel/CMake

export CHPL_VERSION=$(cat CHPL_VERSION)
export CHPL_HOME="$PWD/chapel-${CHPL_VERSION}MCG"

# Download Chapel if not found
if [ ! -d "$CHPL_HOME" ]; then
    wget -c https://github.com/chapel-lang/chapel/releases/download/$CHPL_VERSION/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
    mv chapel-$CHPL_VERSION $CHPL_HOME
fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHPL_HOST_PLATFORM="linux64"
export CHPL_HOST_COMPILER="gnu"
export CHPL_LLVM="bundled"

export CHPL_RT_NUM_THREADS_PER_LOCALE=2
export CHPL_RT_NUM_GPUS_PER_LOCALE=2
export CHPL_LOCALE_MODEL="gpu"
export CHPL_GPU="nvidia"
export CHPL_GPU_ARCH="sm_70"
export CHPL_GPU_MEM_STRATEGY="array_on_device"

export GASNET_PHYSMEM_MAX='64 GB'

cd $CHPL_HOME
make -j $SLURM_CPUS_PER_TASK
cd ../..
