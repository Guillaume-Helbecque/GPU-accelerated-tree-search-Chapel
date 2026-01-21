SHELL := /bin/bash

# ==========================
# Compiler & common options
# ==========================

CHPL_COMPILER := chpl
CHPL_COMMONS_DIR = ./commons

CHPL_COMMON_OPTS := --fast -M $(CHPL_COMMONS_DIR)

# ==========================
# Build Chapel codes
# ==========================

MAIN_FILES = $(wildcard main_*.chpl)
EXECUTABLES = $(MAIN_FILES:.chpl=.out)

all: $(EXECUTABLES)

# ==================
# PFSP
# ==================

CHPL_PFSP_MODULES_DIR = ./benchmarks/pfsp
CHPL_PFSP_OPTS = -M $(CHPL_PFSP_MODULES_DIR)

main_pfsp.out: main_pfsp.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_OPTS) $< -o $@

# ==================
# NQueens
# ==================

CHPL_NQUEENS_MODULES_DIR = ./benchmarks/nqueens
CHPL_NQUEENS_OPTS = -M $(CHPL_NQUEENS_MODULES_DIR)

main_nqueens.out: main_nqueens.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_OPTS) $< -o $@

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
