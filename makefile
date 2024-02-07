SHELL := /bin/bash

# Compilers & common options
CHPL_COMPILER := chpl
CHPL_COMMON_OPTS := --fast
CHPL_PFSP_LIBPATH := -M lib/pfsp

# Source files
CHPL_NQUEENS_SOURCES := nqueens_chpl.chpl nqueens_gpu_chpl.chpl nqueens_gpu_unified_mem_chpl.chpl nqueens_multigpu_chpl.chpl
CHPL_PFSP_SOURCES := pfsp_chpl.chpl #pfsp_gpu_chpl.chpl pfsp_gpu_unified_mem_chpl.chpl pfsp_multigpu_chpl.chpl

# Object files
CHPL_NQUEENS_OBJECTS := $(CHPL_NQUEENS_SOURCES:.chpl=.o)
CHPL_PFSP_OBJECTS := $(CHPL_PFSP_SOURCES:.chpl=.o)

# Build codes
all: $(CHPL_NQUEENS_OBJECTS) $(CHPL_PFSP_OBJECTS)

# Pattern rule for CHPL NQueens
%.o: %.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $< -o $@

# Pattern rule for CHPL PFSP
pfsp_chpl.o: pfsp_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_OBJECTS)
