# Primitive cost microbenchmarks. No torch dependency: plain CUDA binaries so
# they build against whatever toolkit is installed, independent of the version
# PyTorch was compiled with. NVML is linked for the shared-GPU guard, and the
# driver API for TMA tensor maps.
#
#   make                      # build for SM 100a (B200)
#   make run                  # build and print the tables
#   make run JSON=b200        # also record b200_sync.json and b200_tma.json
#
# Override SM for a different target, e.g. `make SM=90a` for Hopper.

NVCC ?= nvcc
SM   ?= 100a
JSON ?=

NVCCFLAGS := -O3 -std=c++17 -gencode=arch=compute_$(SM),code=sm_$(SM)
LIBS      := -lnvidia-ml

BINS := sync_primitives tma_bandwidth

all: $(BINS)

# Rebuild when the flags change (e.g. a different SM), not only the sources.
.build_flags: FORCE
	@echo '$(NVCC) $(NVCCFLAGS) $(LIBS)' | cmp -s - $@ || \
		echo '$(NVCC) $(NVCCFLAGS) $(LIBS)' > $@

sync_primitives: sync_primitives.cu microbench_common.cuh .build_flags
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS)

tma_bandwidth: tma_bandwidth.cu microbench_common.cuh .build_flags
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS) -lcuda

run: $(BINS)
	./sync_primitives $(if $(JSON),$(JSON)_sync.json)
	./tma_bandwidth $(if $(JSON),$(JSON)_tma.json)

clean:
	rm -f $(BINS) .build_flags

.PHONY: all run clean FORCE
