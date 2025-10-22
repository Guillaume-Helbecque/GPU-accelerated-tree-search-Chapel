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
# QAP
# ==================

CHPL_QAP_LIBPATH := -M lib/qap

qap_sequential_glb.out: qap_sequential_glb.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QAP_LIBPATH) -snewRangeLiteralType $< -o $@

qap_sequential_hhb.out: qap_sequential_hhb.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QAP_LIBPATH) -snewRangeLiteralType $< -o $@

qap_gpu_glb.out: qap_gpu_glb.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QAP_LIBPATH) -snewRangeLiteralType $< -o $@

qap_gpu_hhb.out: qap_gpu_hhb.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QAP_LIBPATH) -snewRangeLiteralType $< -o $@

qap_multigpu_glb.out: qap_multigpu_glb.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_QAP_LIBPATH) -snewRangeLiteralType $< -o $@
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
