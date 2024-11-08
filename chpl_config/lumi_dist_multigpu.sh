#!/usr/bin/env bash

# Configuration of Chapel for distributed AMD (multi-)GPU-accelerated experiments
# on the LUMI pre-exascale supercomputer (https://docs.lumi-supercomputer.eu/).

# Load modules
module load LUMI/23.09
module load partition/G
module load rocm/5.4.6
module load cray-pmi/6.1.13

export HERE=$(pwd)

export CHPL_VERSION=$(cat CHPL_VERSION)
export CHPL_HOME=~/chapel-${CHPL_VERSION}_DG_amd

# Download Chapel if not found
if [ ! -d "$CHPL_HOME" ]; then
  cd ~
  wget -c https://github.com/chapel-lang/chapel/releases/download/${CHPL_VERSION}/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
  mv chapel-$CHPL_VERSION $CHPL_HOME
fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
export CHPL_TARGET_COMPILER="llvm"
export CHPL_LLVM="system" # required for AMD GPU architectures

# Network-related environment variables
export CHPL_COMM="ofi" # Cray-specific native communication layer
# export CHPL_COMM_SUBSTRATE='udp' # for Ethernet network
# export CHPL_LAUNCHER="slurm-srun"

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
export CHPL_RT_NUM_THREADS_PER_LOCALE=8

# export GASNET_PHYSMEM_MAX='64 GB'

# GPU-related environment variables
export CHPL_LOCALE_MODEL="gpu"
export CHPL_GPU="amd"
export CHPL_GPU_ARCH="gfx90a"
export CHPL_RT_NUM_GPUS_PER_LOCALE=8
export CHPL_GPU_MEM_STRATEGY="array_on_device" # default

# Setting ROCm/LLVM path manually as the Chapel compiler targets a more recent
# ROCm version (6.0.3) that is not supported by Chapel 2.1.0 (see Chapel issue #25952 on GitHub)
export CHPL_ROCM_PATH="/appl/lumi/SW/LUMI-23.09/G/EB/rocm/5.4.6"

# Install Chapel
cd $CHPL_HOME
patch -N -p1 < $HERE/perf_patch.patch # see Chapel PR #24970 on Github (remove it when Chapel 2.2 is released)
make -j $NUM_T_LOCALE
cd $HERE/..
