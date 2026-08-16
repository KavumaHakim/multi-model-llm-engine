# Kimi K3 inference engine.
#
#   make                build the engine (bin/k3)
#   make test           run every test that needs no model weights
#   make bench          kernel microbenchmarks
#   make portable       build without -march/-mcpu=native (for distribution)
#   make debug          -O0 -g with assertions
#   make asan / ubsan   sanitizer builds
#   make format         clang-format the tree
#   make clean
#
# Nothing here requires a checkpoint. `make test` is the gate that must stay green.
#
# PLATFORMS. Linux/x86-64 is the reference. macOS/arm64 builds with plain `make` too,
# but needs Homebrew's libomp for OpenMP (`brew install libomp`) because Apple Clang
# ships no OpenMP runtime; the platform block below detects and wires it up.

# ---------------------------------------------------------------------------- config --
CC       ?= cc
PYTHON   ?= python3
BUILD    ?= build
BIN      ?= bin
PREFIX   ?= /usr/local

# ---------------------------------------------------------------------- platform --
# Two things differ on macOS/arm64 and both are build failures, not warnings:
#
#   -march=native   is x86 spelling. -mcpu= is the arm64 one, and it is what sets
#                   tuning as well as architecture; -march= on aarch64 selects an
#                   architecture level and leaves tuning alone even where it is
#                   accepted, and it is rejected outright by Apple Clang before
#                   roughly Xcode 13. `native` rather than a named core, so the same
#                   line works on M1 through M5 and beyond.
#   -fopenmp        Apple Clang ships no OpenMP runtime. Homebrew's libomp supplies it,
#                   but the flag must be passed through the preprocessor
#                   (-Xpreprocessor -fopenmp) and the library linked explicitly.
#
# UNAME_S/UNAME_M are the detection; everything below keys off them. Any of these can
# still be overridden on the command line, which is how `make portable` and the
# sanitizer targets work.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  ifeq ($(UNAME_M),arm64)
    ARCH ?= -mcpu=native
  else
    ARCH ?= -march=native
  endif
  # Locate libomp without hardcoding a prefix: Homebrew is /opt/homebrew on Apple
  # Silicon and /usr/local on Intel, and MacPorts is elsewhere again. Fall back to the
  # Apple Silicon default if brew is not on PATH, so the error message names a real
  # path rather than an empty one.
  # `:=`, not `?=`. CFLAGS and LDFLAGS are recursive, so a recursive OMP_PREFIX re-runs
  # `brew --prefix` once per expansion -- about twenty forks for `make test`, each a
  # process spawn. A command-line `make OMP_PREFIX=...` still wins either way, because
  # command-line assignments override makefile assignments regardless of flavour.
  OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMP_CFLAGS ?= -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LDFLAGS ?= -L$(OMP_PREFIX)/lib -lomp
  # Say what is wrong and how to fix it, rather than letting the compiler report a
  # missing omp.h fifty lines into the build.
  ifeq ($(wildcard $(OMP_PREFIX)/include/omp.h),)
    $(warning libomp not found at $(OMP_PREFIX). Install it with `brew install libomp`,)
    $(warning or point the build at another copy with `make OMP_PREFIX=/path/to/libomp`.)
  endif
else
  # -march=native is a real win on the expert matmuls but produces a binary that will
  # not run on an older CPU. `make portable` drops it.
  ARCH ?= -march=native
  OMP_CFLAGS ?= -fopenmp
  OMP_LDFLAGS ?= -fopenmp
endif

# -Wpointer-arith is not cosmetic: weight pointers are `const void *`, and arithmetic on
# a void pointer is a silent GNU extension that strides by ONE BYTE. Without this flag
# that mistake compiles clean under -Wall -Wextra and returns the wrong tensor.
#
# -ffp-contract=off keeps floating-point results reproducible across compilers by
# disabling automatic FMA contraction. The test-suite compares against a reference to a
# fixed tolerance; letting the compiler fuse changes results by more than that.
WARN     := -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter
# -pthread is for the trunk's asynchronous reader, which is a plain pthread rather than
# an OpenMP construct. It composes with the platform OpenMP flags above rather than
# replacing them: Apple Clang needs -Xpreprocessor -fopenmp AND -pthread.
CFLAGS   ?= -O3 -std=gnu99 $(WARN) $(ARCH) $(OMP_CFLAGS) -pthread -ffp-contract=off
LDFLAGS  ?= -lm $(OMP_LDFLAGS) -pthread

# Flat include search across the module dirs: sources use "k3.h", "k3_cache.h" etc
# rather than path-qualified includes, which keeps them relocatable.
INCLUDES := -Iinclude -Iinclude/k3 -Ithird_party \
            -Isrc/core -Isrc/io -Isrc/cache -Isrc/model -Isrc/tokenizer \
            -Isrc/tensor -Isrc/storage -Isrc/formats -Isrc/runtime -Isrc/kernels \
            -Isrc/quant -Isrc/models -Isrc/models/kimi -Isrc/models/qwen3

# ----------------------------------------------------------------------------- files --
# The generic runtime layer. Model-independent by construction: nothing under src/tensor,
# src/storage or src/formats includes a k3_* header except formats/safetensors.c, which
# is a documented migration adapter over the existing reader (see its header comment).
GENERIC_SRC := src/tensor/dtype.c src/tensor/tensor.c \
               src/storage/file.c src/storage/cache.c src/storage/streamer.c \
               src/formats/safetensors.c \
               src/runtime/hwinfo.c src/runtime/memory.c src/runtime/planner.c \
               src/kernels/kernel.c src/kernels/kernel_avx2.c \
               src/quant/quant.c src/quant/mxfp4.c src/quant/q8_0.c \
               src/quant/q4_k.c src/quant/q6_k.c \
               src/formats/gguf.c \
               src/tokenizer/bpe.c
GENERIC_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(GENERIC_SRC))

# Model backends. NOT part of GENERIC_OBJ, and the distinction is the architecture
# rather than bookkeeping: everything in GENERIC_SRC is model-independent and must link
# without any backend present, which is what stops the runtime growing an `if (arch ==
# ...)`. Folding these in there made every generic test drag in kimi.o and, through it,
# k3_ops -- a link error that was really a layering violation.
MODEL_SRC := src/models/registry.c src/models/kimi/kimi.c src/models/qwen3/qwen3.c
MODEL_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(MODEL_SRC))

ENGINE_SRC := src/core/k3_ops.c \
              src/io/k3_st.c src/io/k3_load.c src/io/k3_trunk.c \
              src/cache/k3_cache.c \
              src/model/k3_bind.c
ENGINE_OBJ := $(patsubst %.c,$(BUILD)/%.o,$(ENGINE_SRC)) $(GENERIC_OBJ) $(MODEL_OBJ)

# What k3_ops.o now needs to link. Since M6 the MXFP4 kernels live in src/quant, so
# every test binary that links k3_ops.o needs the quant layer behind it: quant.c for the
# registry, mxfp4.c for the kernels themselves, dtype.c because the registry names
# dtypes, and kernel_avx2.o because mxfp4.c asks the CPU at runtime whether it has AVX2.
# Named once rather than repeated across the seven targets that link k3_ops.o.
QUANT_OBJ := $(BUILD)/src/quant/quant.o $(BUILD)/src/quant/mxfp4.o \
             $(BUILD)/src/quant/q8_0.o $(BUILD)/src/quant/q4_k.o \
             $(BUILD)/src/quant/q6_k.o

K3_OPS_OBJ := $(BUILD)/src/core/k3_ops.o $(QUANT_OBJ) \
              $(BUILD)/src/tensor/dtype.o $(BUILD)/src/kernels/kernel_avx2.o

CLI_SRC    := src/cli/k3_run.c
CLI_BIN    := $(BIN)/k3

# The multi-model CLI. bin/k3 stays exactly as it was -- it is still the way to run K3,
# whose forward pass has not yet migrated onto the backend interface.
ENGINE_CLI_SRC := src/app/cli.c
ENGINE_CLI_BIN := $(BIN)/engine

# Tests that need no checkpoint. These run in CI on every push.
UNIT_TESTS := test_ops test_cache test_st test_cfg test_tok scale_test k3_model \
              test_tensor test_cache_generic test_streamer test_planner test_kernels \
              test_quant test_model test_gguf
# Tests that need real shards. Built and run by `make test-all` with SHARD_DIR set;
# see the weights-test target below.
WEIGHT_TESTS := test_expert test_real_layer

TEST_BINS  := $(addprefix $(BIN)/,$(UNIT_TESTS))
WEIGHT_BINS := $(addprefix $(BIN)/,$(WEIGHT_TESTS))

FIXTURES   ?= tests/fixtures
TOK_FILES  ?= $(HOME)/k3model

# The safetensors test rebuilds an index and writes it to $(BUILD) rather than /tmp, so
# two concurrent `make test` runs cannot race on one filename and `make clean` removes it.

# ---------------------------------------------------------------------------- targets --
.PHONY: all test test-all bench portable debug asan ubsan format clean install help \
        tok cfg ops cache st oracle weights-test

all: $(CLI_BIN) $(ENGINE_CLI_BIN)

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(CLI_BIN): $(CLI_SRC) $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $(CLI_SRC) $(ENGINE_OBJ) -o $@ $(LDFLAGS)

$(BIN):
	@mkdir -p $(BIN)

# Each test links only what it needs, so a failure points at one subsystem.
$(BIN)/test_ops: tests/unit/test_ops.c $(K3_OPS_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_cache: tests/unit/test_cache.c $(BUILD)/src/cache/k3_cache.o \
                   $(BUILD)/src/io/k3_load.o $(BUILD)/src/io/k3_st.o \
                   $(K3_OPS_OBJ) $(BUILD)/src/storage/cache.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# The generic cache links NOTHING from K3. A link error here means a model-specific
# dependency crept into src/storage, which is the layering rule this target enforces.
$(BIN)/test_cache_generic: tests/unit/test_cache_generic.c $(BUILD)/src/storage/cache.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_streamer: tests/unit/test_streamer.c $(BUILD)/src/storage/streamer.o \
                      $(BUILD)/src/storage/file.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# The planner links only runtime/. It must not need storage, formats or any model: it is
# arithmetic over facts already gathered, which is what lets `inspect` run it on a
# machine that could not host the model.
$(BIN)/test_planner: tests/unit/test_planner.c $(BUILD)/src/runtime/hwinfo.o \
                     $(BUILD)/src/runtime/memory.o $(BUILD)/src/runtime/planner.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_kernels: tests/unit/test_kernels.c $(BUILD)/src/kernels/kernel.o \
                     $(BUILD)/src/kernels/kernel_avx2.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# K3_OPS_OBJ already carries kernel_avx2.o, so it is not listed again here: naming an
# object twice on one link line is a duplicate-symbol error, not a no-op. The kimi
# backend needs k3_ops for k3_layer_scratch/k3_moe_scratch, which is deliberate --
# inspect reports the same scratch figures the run path actually allocates rather than
# an independent estimate of them.
# $(MODEL_OBJ), not the individual backends: the registry references every built-in by
# symbol, so a backend added there must be linked here too or the link fails. Listing
# them one by one meant adding qwen3 broke this target. $^ removes duplicates, so the
# overlap between GENERIC_OBJ and K3_OPS_OBJ is harmless.
$(BIN)/test_model: tests/unit/test_model.c $(MODEL_OBJ) $(GENERIC_OBJ) \
                   $(BUILD)/src/io/k3_st.o $(K3_OPS_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Needs the 5 GB container, so it is NOT in UNIT_TESTS: `make test` must stay runnable
# on a machine without the model. `make qwen3` runs it.
$(BIN)/test_qwen3: tests/unit/test_qwen3.c $(MODEL_OBJ) $(GENERIC_OBJ) \
                   $(BUILD)/src/io/k3_st.o $(K3_OPS_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(ENGINE_CLI_BIN): $(ENGINE_CLI_SRC) $(ENGINE_OBJ) $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_tokenizer: tests/unit/test_tokenizer.c $(BUILD)/src/tokenizer/bpe.o \
                       $(BUILD)/src/formats/gguf.o $(BUILD)/src/storage/file.o \
                       $(BUILD)/src/tensor/dtype.o $(BUILD)/src/tensor/tensor.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# Both need the container, so neither is in UNIT_TESTS: `make test` must stay runnable
# without the 5 GB model.
.PHONY: qwen3 tokenizer
qwen3: $(BIN)/test_qwen3
	./$(BIN)/test_qwen3 tests/fixtures/gguf/qwen3_golden.bin

tokenizer: $(BIN)/test_tokenizer
	./$(BIN)/test_tokenizer tests/fixtures/gguf/tokenizer_golden.txt

$(BIN)/test_gguf: tests/unit/test_gguf.c $(BUILD)/src/formats/gguf.o \
                  $(BUILD)/src/storage/file.o $(BUILD)/src/tensor/dtype.o \
                  $(BUILD)/src/tensor/tensor.o $(QUANT_OBJ) \
                  $(BUILD)/src/kernels/kernel_avx2.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_quant: tests/unit/test_quant.c $(QUANT_OBJ) \
                   $(BUILD)/src/tensor/dtype.o $(BUILD)/src/kernels/kernel_avx2.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# The bit-identity gate is only meaningful if the AVX2 code is present WITHOUT -mavx2 on
# the command line -- that is the whole point of the per-function target attributes, and
# it is what lets one binary serve CPUs that differ. Building this target portably and
# running it proves the dispatcher is doing the work, not the compiler flags.
$(BIN)/test_kernels_portable: tests/unit/test_kernels.c src/kernels/kernel.c \
                              src/kernels/kernel_avx2.c | $(BIN)
	$(CC) -O2 -std=gnu99 $(WARN) $(INCLUDES) $^ -o $@ -lm

$(BIN)/test_st: tests/unit/test_st.c $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# The tokenizer and config reader are portable C99 with no OpenMP and no platform calls,
# so they build and are verifiable on any machine, including one with no checkpoint.
$(BIN)/test_tok: tests/unit/test_tok.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $< -o $@

# Compiles from source rather than linking objects, so the quant sources are listed
# individually. The registry in quant.c references every built-in format by symbol, so
# adding a format means adding it here too or the link fails.
$(BIN)/test_cfg: tests/unit/test_cfg.c src/core/k3_ops.c src/quant/quant.c \
                 src/quant/mxfp4.c src/quant/q8_0.c src/quant/q4_k.c src/quant/q6_k.c \
                 src/tensor/dtype.c src/kernels/kernel_avx2.c | $(BIN)
	$(CC) -O2 -std=c99 $(WARN) -Wno-unused-function $(INCLUDES) $^ -o $@ -lm

# Allocates at REAL model widths (a ~1.8 GB KDA layer), so it needs the optimised build
# rather than the portable C99 one the tokenizer and config tests use.
$(BIN)/scale_test: tests/unit/scale_test.c $(K3_OPS_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/k3_model: tests/unit/k3_model.c $(K3_OPS_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

# The generic layer links the safetensors reader because its adapter wraps it. It does
# NOT link k3_ops.o: nothing under src/tensor or src/storage may depend on a kernel, and
# a link error here is the enforcement of that rule.
$(BIN)/test_tensor: tests/unit/test_tensor.c $(GENERIC_OBJ) $(BUILD)/src/io/k3_st.o | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/bench_kernels: benchmarks/bench_kernels.c $(K3_OPS_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

## test: everything that needs no model weights
test: $(TEST_BINS)
	@echo "== op kernels ==";        ./$(BIN)/test_ops $(FIXTURES)/ops
	@echo "== streaming cache ==";   ./$(BIN)/test_cache $(FIXTURES)/cache
	@echo "== safetensors ==";       ./$(BIN)/test_st $(FIXTURES)/st $(BUILD)/st_index.json \
	    plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32
	@echo "== config reader ==";     ./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
	@echo "== config refusals =="; \
	  for f in no_layermap bad_layer_index bad_topk; do \
	      ./$(BIN)/test_cfg reject $(FIXTURES)/cfg/$$f.json || exit 1; \
	  done
	@echo "== tokenizer =="; \
	  if [ -f "$(TOK_FILES)/tiktoken.model" ]; then \
	      ./$(BIN)/test_tok $(TOK_FILES) roundtrip src/core/k3_ops.c; \
	  else \
	      echo "  NOT RUN: no tiktoken.model at $(TOK_FILES)"; \
	      echo "           the vocabulary ships with the checkpoint, not with this"; \
	      echo "           repository. Run: make tok TOK_FILES=/path/to/k3model"; \
	  fi
	@echo "== generic layer ==";     ./$(BIN)/test_tensor $(FIXTURES)/st $(BUILD)
	@echo "== generic cache ==";     ./$(BIN)/test_cache_generic
	@echo "== block streamer ==";    ./$(BIN)/test_streamer $(BUILD)
	@echo "== planner ==";           ./$(BIN)/test_planner
	@echo "== cpu kernels ==";       ./$(BIN)/test_kernels
	@echo "== quantization ==";      ./$(BIN)/test_quant
	@echo "== model backends ==";    ./$(BIN)/test_model
	@echo "== gguf + k-quants ==";   ./$(BIN)/test_gguf tests/fixtures/gguf/golden.bin $(BUILD)
	@echo "== cpu kernels (no -mavx2, runtime dispatch only) =="; \
	  $(MAKE) --no-print-directory $(BIN)/test_kernels_portable && ./$(BIN)/test_kernels_portable | tail -3
	@echo "== real dimensions ==";   ./$(BIN)/scale_test
	@echo "== full-model oracle =="; ./$(BIN)/k3_model $(FIXTURES)
	@echo
	@if [ ! -f "$(TOK_FILES)/tiktoken.model" ]; then \
	     echo "NOTE: tokenizer parity did NOT run; see above. Everything else did."; \
	 fi
	@echo "ALL WEIGHTLESS TESTS PASSED"

## test-all: adds tests that need a real checkpoint (set SHARD_DIR)
test-all: test
	@test -n "$(SHARD_DIR)" || { echo "set SHARD_DIR=/path/to/shards"; exit 2; }
	$(MAKE) weights-test SHARD_DIR=$(SHARD_DIR)

weights-test: $(WEIGHT_BINS)
	./$(BIN)/test_expert $(SHARD_DIR) 1 64
	./$(BIN)/test_real_layer $(SHARD_DIR) 1 4 8

$(BIN)/test_expert: tests/unit/test_expert.c $(BUILD)/src/io/k3_load.o \
                    $(BUILD)/src/io/k3_st.o $(K3_OPS_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN)/test_real_layer: tests/unit/test_real_layer.c $(ENGINE_OBJ) | $(BIN)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

## tok: tokenizer parity against the reference implementation
tok: $(BIN)/test_tok
	@$(PYTHON) tools/tok_parity.py ./$(BIN)/test_tok

## cfg: config reader against both supported config layouts
cfg: $(BIN)/test_cfg
	@./$(BIN)/test_cfg fixture $(FIXTURES)/ref_k3.json
	@test -f "$(TOK_FILES)/config.json" && ./$(BIN)/test_cfg real $(TOK_FILES)/config.json \
	    || echo "  (skipped real config: none at $(TOK_FILES))"

## bench: kernel microbenchmarks, no weights required
bench: $(BIN)/bench_kernels
	./$(BIN)/bench_kernels

## portable: drop the -march/-mcpu=native tuning, for a distributable binary
# On x86-64 that means a generic AVX2 + FMA baseline. On arm64 there is no equivalent
# sub-baseline worth naming -- the ISA is the baseline -- so tuning is simply omitted.
portable:
ifeq ($(UNAME_S)/$(UNAME_M),Darwin/arm64)
	$(MAKE) ARCH= all
else
	$(MAKE) ARCH="-mavx2 -mfma" all
endif

## debug: -O0 -g, assertions on
debug:
	$(MAKE) CFLAGS="-O0 -g3 -std=gnu99 $(WARN) $(OMP_CFLAGS) -ffp-contract=off" all

# The sanitizer builds drop OpenMP deliberately: ASan's interceptors and the OpenMP
# runtime's thread pool produce false positives together, and a serial build is the
# point of a sanitizer run. OMP_CFLAGS is still omitted rather than replaced, so the
# #pragma omp lines compile to nothing on every platform alike.
asan:
	$(MAKE) CFLAGS="-O1 -g -std=gnu99 $(WARN) -fsanitize=address,undefined -fno-omit-frame-pointer" \
	        LDFLAGS="-lm -fsanitize=address,undefined" ARCH= all

ubsan:
	$(MAKE) CFLAGS="-O1 -g -std=gnu99 $(WARN) -fsanitize=undefined" \
	        LDFLAGS="-lm -fsanitize=undefined" ARCH= all

format:
	@command -v clang-format >/dev/null || { echo "clang-format not installed"; exit 1; }
	clang-format -i $(shell find src include tests benchmarks -name '*.c' -o -name '*.h' 2>/dev/null)

# Installs the binary AND the public headers, matching CMake's install rules exactly
# the two build systems are documented as interchangeable, so they must stay so.
# third_party/json.h goes with them: k3_cfg.h includes it and exposes jval in its
# signatures, so an installed k3_cfg.h without it does not compile.
install: $(CLI_BIN)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(CLI_BIN) $(DESTDIR)$(PREFIX)/bin/k3
	install -d $(DESTDIR)$(PREFIX)/include/k3
	install -m 644 include/k3/*.h $(DESTDIR)$(PREFIX)/include/k3/
	install -m 644 third_party/json.h $(DESTDIR)$(PREFIX)/include/k3/

clean:
	rm -rf $(BUILD) $(BIN)

## help: list targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
