# Makefile for OpenVX Pipelining Multicore Samples
# Usage: make OPENVX_INCLUDE=/path/to/headers OPENVX_LIB=/path/to/libs

CC       ?= gcc
CFLAGS   ?= -O3 -Wall -Wextra -std=c99 -D_POSIX_C_SOURCE=200809L
LDFLAGS  ?=

OPENVX_INCLUDE ?= /usr/local/include
OPENVX_LIB     ?= /usr/local/lib

TARGETS  = pipelining_multicore pipelining_vs_nonpipelining multiscale_feature_extraction

.PHONY: all clean run run-all help

all: $(TARGETS)

pipelining_multicore: pipelining_multicore.c
	$(CC) $(CFLAGS) -o $@ $< \
		-I$(OPENVX_INCLUDE) \
		-L$(OPENVX_LIB) \
		-lopenvx \
		-Wl,-rpath,$(OPENVX_LIB) \
		$(LDFLAGS)

pipelining_vs_nonpipelining: pipelining_vs_nonpipelining.c
	$(CC) $(CFLAGS) -o $@ $< \
		-I$(OPENVX_INCLUDE) \
		-L$(OPENVX_LIB) \
		-lopenvx \
		-Wl,-rpath,$(OPENVX_LIB) \
		$(LDFLAGS)

multiscale_feature_extraction: multiscale_feature_extraction.c
	$(CC) $(CFLAGS) -o $@ $< \
		-I$(OPENVX_INCLUDE) \
		-L$(OPENVX_LIB) \
		-lopenvx \
		-Wl,-rpath,$(OPENVX_LIB) \
		$(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

run: pipelining_multicore
	@echo "=== pipelining_multicore (auto threads) ==="
	LD_LIBRARY_PATH=$(OPENVX_LIB):$(LD_LIBRARY_PATH) ./pipelining_multicore
	@echo ""
	@echo "=== pipelining_multicore (1 thread) ==="
	OPENVX_PIPELINING_THREADS=1 LD_LIBRARY_PATH=$(OPENVX_LIB):$(LD_LIBRARY_PATH) ./pipelining_multicore

run-all: all
	@echo "========================================"
	@echo "  pipelining_vs_nonpipelining"
	@echo "========================================"
	LD_LIBRARY_PATH=$(OPENVX_LIB):$(LD_LIBRARY_PATH) ./pipelining_vs_nonpipelining
	@echo ""
	@echo "========================================"
	@echo "  multiscale_feature_extraction"
	@echo "========================================"
	LD_LIBRARY_PATH=$(OPENVX_LIB):$(LD_LIBRARY_PATH) ./multiscale_feature_extraction

help:
	@echo "OpenVX Pipelining Multicore Samples"
	@echo ""
	@echo "Build:"
	@echo "  make OPENVX_INCLUDE=../../include OPENVX_LIB=../../target/release"
	@echo ""
	@echo "Samples:"
	@echo "  pipelining_multicore           - basic wave-parallel demo"
	@echo "  pipelining_vs_nonpipelining    - throughput comparison"
	@echo "  multiscale_feature_extraction  - real-world CV pipeline"
	@echo ""
	@echo "Run:"
	@echo "  make run            - run pipelining_multicore"
	@echo "  make run-all        - run comparison + real-world samples"
	@echo ""
	@echo "Environment:"
	@echo "  OPENVX_PIPELINING_THREADS=N   - set thread pool size (0=auto)"
