#!/bin/bash -l

# Configuration of Chapel for (multi-)GPU-accelerated experiments on the Iris
# cluster of the Université du Luxembourg (https://hpc-docs.uni.lu/systems/iris/).

# Load modules
module load toolchain/foss/2020b
module load system/CUDA/11.1
module load devel/CMake

export HERE=$(pwd)

export CHPL_VERSION=$(cat CHPL_VERSION)
export CHPL_HOME="$HERE/chapel-${CHPL_VERSION}MCG_nvidia"

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

cd $CHPL_HOME
make -j $SLURM_CPUS_PER_TASK
cd $HERE/..
