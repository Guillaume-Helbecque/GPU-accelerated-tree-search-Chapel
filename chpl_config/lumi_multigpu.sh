#!/usr/bin/env bash

# Configuration of Chapel for AMD (multi-)GPU-accelerated experiments on the
# LUMI pre-exascale supercomputer (https://docs.lumi-supercomputer.eu/).

# Load modules
module load LUMI/23.09
module load partition/G
module load PrgEnv-cray/8.4.0
module load cce/16.0.1
module load ncurses/6.4-cpeCray-23.09
module load buildtools/23.09 # contains CMake

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
export CHPL_LLVM="system" # required for AMD arch
export CHPL_COMM="none"
export CHPL_LAUNCHER="none"

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
export CHPL_LOCALE_MODEL="gpu"
export CHPL_GPU="amd"
export CHPL_GPU_ARCH="gfx90a"
export CHPL_GPU_MEM_STRATEGY="array_on_device" # default
export CHPL_RT_NUM_THREADS_PER_LOCALE=4
export CHPL_RT_NUM_GPUS_PER_LOCALE=4

export GASNET_PHYSMEM_MAX='64 GB'

cd $CHPL_HOME
patch -N -p1 < $HERE/perf_patch.patch # see Chapel PR #24970 on Github (remove it when Chapel 2.1 is released)
make -j $NUM_T_LOCALE
cd $HERE/..
