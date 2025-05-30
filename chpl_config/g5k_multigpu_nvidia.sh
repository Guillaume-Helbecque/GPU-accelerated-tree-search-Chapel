#!/usr/bin/env bash

# Configuration of Chapel for Nvidia (multi-)GPU-accelerated experiments on the
# French national Grid5000 testbed (https://www.grid5000.fr/w/Grid5000:Home).

# Load modules
module load gcc/13.2.0_gcc-10.4.0
module load cmake/3.23.3_gcc-10.4.0
module load cuda/12.2.1_gcc-10.4.0

export HERE=$(pwd)

export CHPL_VERSION=$(cat CHPL_VERSION)
export CHPL_HOME=~/chapel-${CHPL_VERSION}MCG_nvidia

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

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
export CHPL_RT_NUM_THREADS_PER_LOCALE=12
export CHPL_RT_NUM_GPUS_PER_LOCALE=8
export CHPL_LOCALE_MODEL="gpu"
export CHPL_GPU="nvidia"
export CHPL_GPU_ARCH="sm_80"
export CHPL_GPU_MEM_STRATEGY="array_on_device"

cd $CHPL_HOME
make -j $NUM_T_LOCALE
cd $HERE/..
