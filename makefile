SHELL := /bin/bash

# Common settings
CHPL_COMPILER := chpl
CHPL_COMMON_OPTS := --fast -M lib/common

# Source files
CHPL_NQUEENS_SOURCES := nqueens_chpl.chpl nqueens_gpu_chpl.chpl nqueens_multigpu_chpl.chpl
CHPL_PFSP_SOURCES := pfsp_chpl.chpl #pfsp_gpu_chpl.chpl pfsp_multigpu_chpl.chpl

# Object files
CHPL_NQUEENS_OBJECTS := $(CHPL_NQUEENS_SOURCES:.chpl=.o)
CHPL_PFSP_OBJECTS := $(CHPL_PFSP_SOURCES:.chpl=.o)

# Library paths
CHPL_NQUEENS_LIBPATH :=
CHPL_PFSP_LIBPATH := -M lib/pfsp

# Build codes
all: $(CHPL_NQUEENS_OBJECTS) $(CHPL_PFSP_OBJECTS)

# Pattern rule for N-Queens
nqueens_%.o: nqueens_%.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

# Pattern rule for PFSP
pfsp_%.o: pfsp_%.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_NQUEENS_OBJECTS) $(CHPL_PFSP_OBJECTS)
