SHELL := /bin/bash

# Compilers & common options
CHPL_COMPILER := chpl
CHPL_COMMON_OPTS := --fast

# Source & object files
CHPL_SOURCES := nqueens_chpl.chpl nqueens_gpu_chpl.chpl nqueens_gpu_unified_mem_chpl.chpl nqueens_multigpu_chpl.chpl
CHPL_OBJECTS := $(CHPL_SOURCES:.chpl=.o)

# Build codes
all: $(CHPL_OBJECTS)

# Pattern rule for CHPL source files
%.o: %.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $< -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_OBJECTS)
