SHELL := /bin/bash
.DEFAULT_GOAL := graph

PLATFORM ?= $(XILINX_VITIS)/base_platforms/xilinx_vek280_base_202410_1/xilinx_vek280_base_202410_1.xpfm
AIECC ?= aiecompiler
AIESIM ?= aiesimulator
AIE_FLAGS ?=
# Let the graph finish all iterations; slow FP32 graphs can exceed a fixed limit.
AIESIM_CYCLE_TIMEOUT ?= 0
AIE_SOURCES := $(wildcard src/aie/*.cc src/aie/*.cpp src/aie/*.h)

.PHONY: graph aiesim clean
graph: libadf.a

libadf.a: $(AIE_SOURCES) Makefile
	$(AIECC) --platform="$(PLATFORM)" --target=hw \
		-include="$(XILINX_VITIS)/aietools/include" -include="src/aie" \
		-include="data" -workdir=Work $(AIE_FLAGS) src/aie/graph.cpp

aiesim: libadf.a
	$(AIESIM) --pkg-dir=Work --simulation-cycle-timeout=$(AIESIM_CYCLE_TIMEOUT)

clean:
	rm -rf Work aiesimulator_output .Xil libadf.a
