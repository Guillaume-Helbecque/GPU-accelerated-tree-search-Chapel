#!/usr/bin/env bash

# Configuration of Chapel for AMD (multi-)GPU-accelerated experiments on the
# LUMI pre-exascale supercomputer (https://docs.lumi-supercomputer.eu/).

# Load modules
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

export HERE=$(pwd)

export CHPL_VERSION=$(cat CHPL_VERSION)
export CHPL_HOME=~/chapel-${CHPL_VERSION}_MCG_amd

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
export CHPL_LLVM="bundled" # required for AMD GPU architectures
export CHPL_COMM="none"
export CHPL_LAUNCHER="none"

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
export CHPL_LOCALE_MODEL="gpu"
export CHPL_GPU="amd"
export CHPL_GPU_ARCH="gfx90a"
export CHPL_GPU_MEM_STRATEGY="array_on_device" # default
export CHPL_RT_NUM_THREADS_PER_LOCALE=8
export CHPL_RT_NUM_GPUS_PER_LOCALE=8

cd $CHPL_HOME
make -j $NUM_T_LOCALE
cd $HERE/..
