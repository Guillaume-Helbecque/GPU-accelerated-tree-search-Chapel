#!/usr/bin/env bash

# Configuration of Chapel for distributed Nvidia (multi-)GPU-accelerated experiments
# on the French national Grid5000 testbed (https://www.grid5000.fr/w/Grid5000:Home).

# Load modules
module load gcc/13.2.0_gcc-10.4.0
module load cmake/3.23.3_gcc-10.4.0
module load cuda/12.2.1_gcc-10.4.0
module load libfabric/1.15.1_gcc-10.4.0

export HERE=$(pwd)

export CHPL_VERSION=$(cat CHPL_VERSION)
export CHPL_HOME=~/chapel-${CHPL_VERSION}DG_nvidia

# Download Chapel if not found
if [ ! -d "$CHPL_HOME" ]; then
  cd ~
  wget -c https://github.com/chapel-lang/chapel/releases/download/${CHPL_VERSION}/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
  mv chapel-$CHPL_VERSION $CHPL_HOME
fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
# export CHPL_HOST_COMPILER=gnu
export CHPL_LLVM=bundled

# Network-related environment variables
export CHPL_COMM='gasnet'
export CHPL_COMM_SUBSTRATE='udp' # for Ethernet network
# export CHPL_COMM_SUBSTRATE='ibv' # for InfiniBand network
export CHPL_TARGET_CPU='native'
# NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
export CHPL_RT_NUM_THREADS_PER_LOCALE=2

export GASNET_QUIET=1
export GASNET_IBV_SPAWNER='ssh'
export GASNET_SSH_SERVERS=$(uniq $OAR_NODEFILE | tr '\n' ' ')
export GASNET_PHYSMEM_MAX='64 GB'

# GPU-related environment variables
export CHPL_LOCALE_MODEL="gpu"
export CHPL_GPU="nvidia"
export CHPL_GPU_ARCH="sm_70"
export CHPL_RT_NUM_GPUS_PER_LOCALE=2
export CHPL_GPU_MEM_STRATEGY="array_on_device"

# Install Chapel
cd $CHPL_HOME
make -j $NUM_T_LOCALE
cd $HERE/..
