SHELL := /bin/bash

# ==========================
# Compiler & common options
# ==========================

CHPL_COMPILER := chpl
CHPL_COMMON_OPTS := --fast -M lib/commons

# ==========================
# Build Chapel codes
# ==========================

MAIN_FILES = $(wildcard *_chpl.chpl)
EXECUTABLES = $(MAIN_FILES:.chpl=.out)

all: $(EXECUTABLES)

# ==================
# NQueens
# ==================

CHPL_NQUEENS_LIBPATH := -M lib/nqueens

nqueens_chpl.out: nqueens_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_gpu_chpl.out: nqueens_gpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_multigpu_chpl.out: nqueens_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

nqueens_dist_multigpu_chpl.out: nqueens_dist_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_LIBPATH) $< -o $@

# ==================
# PFSP
# ==================

CHPL_PFSP_LIBPATH := -M lib/pfsp

pfsp_chpl.out: pfsp_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_gpu_chpl.out: pfsp_gpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_multigpu_chpl.out: pfsp_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

pfsp_dist_multigpu_chpl.out: pfsp_dist_multigpu_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_LIBPATH) $< -o $@

# ==================
# Qubit allocation
# ==================

CHPL_QUBIT_ALLOC_LIBPATH := -M lib/qubitAlloc

qubitAlloc_glb_chpl.out: qubitAlloc_glb_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QUBIT_ALLOC_LIBPATH) -snewRangeLiteralType $< -o $@

qubitAlloc_hhb_chpl.out: qubitAlloc_hhb_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QUBIT_ALLOC_LIBPATH) -snewRangeLiteralType $< -o $@

qubitAlloc_gpu_glb_chpl.out: qubitAlloc_gpu_glb_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QUBIT_ALLOC_LIBPATH) -snewRangeLiteralType $< -o $@

qubitAlloc_gpu_hhb_chpl.out: qubitAlloc_gpu_hhb_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QUBIT_ALLOC_LIBPATH) -snewRangeLiteralType $< -o $@

qubitAlloc_multigpu_glb_chpl.out: qubitAlloc_multigpu_glb_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QUBIT_ALLOC_LIBPATH) -snewRangeLiteralType $< -o $@
#
# qubitAlloc_dist_multigpu_chpl.out: qubitAlloc_dist_multigpu_chpl.chpl
# 	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QUBIT_ALLOC_LIBPATH) -snewRangeLiteralType $< -o $@

# ==========================
# Utilities
# ==========================

.PHONY: clean

clean:
	rm -f $(EXECUTABLES)
	rm -f $(EXECUTABLES:=_real)
