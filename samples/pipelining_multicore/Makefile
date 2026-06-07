# Makefile for OpenVX Pipelining Multicore Sample
# Usage: make OPENVX_INCLUDE=/path/to/headers OPENVX_LIB=/path/to/libs

CC       ?= gcc
CFLAGS   ?= -O3 -Wall -Wextra -std=c99
LDFLAGS  ?=

OPENVX_INCLUDE ?= /usr/local/include
OPENVX_LIB     ?= /usr/local/lib

TARGET   = pipelining_multicore
SRCS     = pipelining_multicore.c
OBJS     = $(SRCS:.c=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $< \
		-I$(OPENVX_INCLUDE) \
		-L$(OPENVX_LIB) \
		-lopenvx \
		-Wl,-rpath,$(OPENVX_LIB) \
		$(LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)

run: $(TARGET)
	@echo "=== Running with auto-detected threads ==="
	LD_LIBRARY_PATH=$(OPENVX_LIB):$(LD_LIBRARY_PATH) ./$(TARGET)
	@echo ""
	@echo "=== Running with 1 thread (sequential) ==="
	OPENVX_PIPELINING_THREADS=1 LD_LIBRARY_PATH=$(OPENVX_LIB):$(LD_LIBRARY_PATH) ./$(TARGET)

help:
	@echo "Usage: make OPENVX_INCLUDE=/path/to/openvx/include OPENVX_LIB=/path/to/openvx/lib"
	@echo ""
	@echo "Examples:"
	@echo "  # Using rustVX build artifacts:"
	@echo "  make OPENVX_INCLUDE=../../include OPENVX_LIB=../../target/release"
	@echo ""
	@echo "  # Run with custom thread count:"
	@echo "  OPENVX_PIPELINING_THREADS=4 make run"
