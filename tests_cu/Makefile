# Compiler
NVCC = nvcc

# Optimization toggle: OPT=on (default, -O3) or OPT=off (-O0)
OPT ?= on
ifeq ($(OPT),on)
  OPT_FLAG := -O3
else
  OPT_FLAG := -G
endif

# Compiler flags
NVCC_FLAGS = -std=c++17 $(OPT_FLAG) --expt-relaxed-constexpr -use_fast_math -Xcompiler -Wno-psabi -arch=native -lineinfo
#   -gencode arch=compute_80,code=sm_80 \
#   -gencode arch=compute_86,code=sm_86 \
#   -gencode arch=compute_89,code=sm_89 \
#   -gencode arch=compute_90,code=sm_90 \
#   -gencode arch=compute_90a,code=sm_90a \
#   -gencode arch=compute_80,code=compute_80

# Directories
INCLUDE_DIR = -I../include -I../include/mirage/persistent_kernel/tasks
# INCLUDE_DIR = -I../../include

# Targets
TARGETS = test_prompt_lookup test_linear

all: $(TARGETS)

test_prompt_lookup: ./test_prompt_lookup.cu Makefile
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIR) -o $(if $(OUT),$(OUT),$@) $<

test_linear: ./test_linear.cu Makefile ../include/mirage/persistent_kernel/tasks/linear.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIR) -o $(if $(OUT),$(OUT),$@) $<

run_linear: test_linear
	./test_linear

run_prompt_lookup: test_prompt_lookup
	./test_prompt_lookup

clean:
	rm -f $(TARGETS)