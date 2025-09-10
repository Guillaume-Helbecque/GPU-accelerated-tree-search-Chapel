SHELL := /bin/bash

# Common settings
CHPL_COMPILER := chpl
CHPL_COMMON_OPTS := --fast -M commons

# Source files
CHPL_NQUEENS_SOURCES := main_nqueens.chpl
CHPL_PFSP_SOURCES := main_pfsp.chpl

# Executable files
CHPL_NQUEENS_EXECUTABLES := $(CHPL_NQUEENS_SOURCES:.chpl=.out)
CHPL_PFSP_EXECUTABLES := $(CHPL_PFSP_SOURCES:.chpl=.out)

# Library paths
CHPL_NQUEENS_LIBPATH := -M benchmarks/nqueens
CHPL_PFSP_LIBPATH := -M benchmarks/pfsp

# Build codes
all: $(CHPL_NQUEENS_EXECUTABLES) $(CHPL_PFSP_EXECUTABLES)

# N-Queens

main_nqueens.out: main_nqueens.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

# PFSP

main_pfsp.out: main_pfsp.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_NQUEENS_EXECUTABLES) $(CHPL_PFSP_EXECUTABLES) *_real
