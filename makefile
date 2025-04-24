SHELL := /bin/bash

# Common settings
CHPL_COMPILER := chpl
CHPL_COMMON_OPTS := --fast -M lib/commons

# Source files
CHPL_NQUEENS_SOURCES := nqueens_chpl.chpl nqueens_gpu_chpl.chpl nqueens_multigpu_chpl.chpl nqueens_dist_multigpu_chpl.chpl
CHPL_PFSP_SOURCES := pfsp_chpl.chpl pfsp_gpu_chpl.chpl pfsp_multigpu_chpl.chpl pfsp_dist_multigpu_chpl.chpl

# Executable files
CHPL_NQUEENS_EXECUTABLES := $(CHPL_NQUEENS_SOURCES:.chpl=.out)
CHPL_PFSP_EXECUTABLES := $(CHPL_PFSP_SOURCES:.chpl=.out)

# Library paths
CHPL_NQUEENS_LIBPATH := -M lib/nqueens
CHPL_PFSP_LIBPATH := -M lib/pfsp

# Build codes
all: $(CHPL_NQUEENS_EXECUTABLES) $(CHPL_PFSP_EXECUTABLES)

# N-Queens

nqueens_chpl.out: nqueens_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_gpu_chpl.out: nqueens_gpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_multigpu_chpl.out: nqueens_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_dist_multigpu_chpl.out: nqueens_dist_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

# PFSP

pfsp_chpl.out: pfsp_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_gpu_chpl.out: pfsp_gpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_multigpu_chpl.out: pfsp_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_dist_multigpu_chpl.out: pfsp_dist_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_NQUEENS_EXECUTABLES) $(CHPL_PFSP_EXECUTABLES) *_real
