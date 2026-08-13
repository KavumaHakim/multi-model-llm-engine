#!/usr/bin/env bash
# baseline.sh - capture the known-good K3 baseline before any refactoring.
#
# This is the M0 gate from docs/architecture-report.md §11. It records what the
# upstream engine does on THIS machine so that every later milestone can be checked
# against it rather than against a claim in a README.
#
# Everything here runs WITHOUT the 1.56 TB checkpoint. That is a property of the
# upstream test design, not a compromise: `make test` is explicitly built to gate
# correctness with no weights. The checkpoint-dependent tests (test_expert,
# test_real_layer, conform_all.py) cannot run on this machine and are recorded as
# NOT RUN rather than silently skipped.
#
# Usage:  scripts/baseline.sh [outdir]
set -uo pipefail

OUT="${1:-baseline}"
mkdir -p "$OUT"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------- environment --
log "environment"
{
    echo "date            : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uname           : $(uname -a)"
    echo "git commit      : $(git rev-parse HEAD 2>/dev/null || echo n/a)"
    echo
    echo "--- cpu ---"
    lscpu 2>/dev/null | grep -Ei 'model name|^cpu\(s\)|thread|core|socket|mhz|cache|flags' \
        | sed 's/^/  /' | cut -c1-400
    echo
    echo "--- avx2/fma present? ---"
    grep -o -m1 -E 'avx2|fma|avx512[a-z]*' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' '
    echo
    echo
    echo "--- memory ---"
    free -h 2>/dev/null
    grep -E 'MemTotal|MemAvailable' /proc/meminfo 2>/dev/null
    echo
    echo "--- disk ---"
    df -h . 2>/dev/null
    echo
    echo "--- toolchain ---"
    ${CC:-cc} --version 2>&1 | head -2
    make --version 2>&1 | head -1
    python3 --version 2>&1
} | tee "$OUT/environment.txt"

# ---------------------------------------------------------------------- build --
# The portable baseline (-mavx2 -mfma), which is what upstream CI builds and what a
# distributable binary would use. -march=native is measured separately below.
log "build (portable: -mavx2 -mfma)"
make clean >/dev/null 2>&1
if ! make ARCH="-mavx2 -mfma" -j"$(nproc)" 2>&1 | tee "$OUT/build.log"; then
    echo "BUILD FAILED" | tee -a "$OUT/build.log"
    exit 1
fi
ls -la bin/ 2>/dev/null | tee -a "$OUT/build.log"

# ----------------------------------------------------------------------- test --
# `make test` and not the individual binaries: it is the command upstream documents
# as the gate, and running it whole is what catches a test dropping out of the list.
log "make test (weightless gates)"
/usr/bin/time -v make ARCH="-mavx2 -mfma" test 2>&1 | tee "$OUT/test.log"
TEST_RC=${PIPESTATUS[0]}
echo "make test exit code: $TEST_RC" | tee -a "$OUT/test.log"

# Guard against a silently empty suite, the same assertion upstream CI makes.
for gate in "op kernels" "streaming cache" "safetensors" "config reader" \
            "real dimensions" "full-model oracle"; do
    if grep -q "$gate" "$OUT/test.log"; then
        echo "  present: $gate"
    else
        echo "  MISSING GATE: $gate" | tee -a "$OUT/test.log"
    fi
done | tee -a "$OUT/test.log"

# The three end-to-end gates must be exact. Record them explicitly.
log "end-to-end oracle gates"
grep -E 'GATE [123]' "$OUT/test.log" | tee "$OUT/oracle-gates.txt" || \
    echo "no GATE lines found" | tee "$OUT/oracle-gates.txt"

# ------------------------------------------------------------------ benchmark --
log "kernel microbenchmarks"
make ARCH="-mavx2 -mfma" bin/bench_kernels >/dev/null 2>&1
./bin/bench_kernels 2>&1 | tee "$OUT/bench-portable.txt"

# -march=native, to quantify what the CPU-specific build buys on this Skylake part.
log "kernel microbenchmarks (-march=native)"
make clean >/dev/null 2>&1
make bin/bench_kernels -j"$(nproc)" >/dev/null 2>&1
./bin/bench_kernels 2>&1 | tee "$OUT/bench-native.txt"

# ------------------------------------------------------------- thread scaling --
# The brief calls for 1/2/3/4 threads explicitly, and warns against assuming more
# threads is faster. This machine is 2 physical / 4 logical, so 3 and 4 are
# hyperthread pairs and may well regress. Measure rather than assume.
log "thread scaling sweep"
{
    echo "threads,notes"
    for t in 1 2 3 4; do
        echo "--- OMP_NUM_THREADS=$t ---"
        OMP_NUM_THREADS=$t ./bin/bench_kernels 2>&1
        echo
    done
} | tee "$OUT/thread-sweep.txt"

# ---------------------------------------------------------------- sanitizers --
# ASan+UBSan over the test suite. Slow, but this is the one chance to record that
# the baseline is clean before the refactor starts moving memory ownership around.
log "sanitizer run (ASan + UBSan)"
make clean >/dev/null 2>&1
if make CFLAGS="-O1 -g -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter -fsanitize=address,undefined -fno-omit-frame-pointer -ffp-contract=off" \
        LDFLAGS="-lm -fsanitize=address,undefined" ARCH= \
        bin/test_ops bin/test_st bin/test_cfg bin/k3_model -j"$(nproc)" >/dev/null 2>&1; then
    for t in "test_ops tests/fixtures/ops" "test_st tests/fixtures/st /tmp/idx.json plain.f32.2d plain.bf16.1d tricky.f16.1d packed.u8.2d scalar.f32 second.shard.f32" \
             "test_cfg fixture tests/fixtures/ref_k3.json" "k3_model tests/fixtures"; do
        echo "--- ./bin/$t ---"
        # shellcheck disable=SC2086
        ./bin/$t 2>&1 | tail -20
    done
else
    echo "sanitizer build failed (recorded, not fatal)"
fi 2>&1 | tee "$OUT/sanitizers.txt"

# ------------------------------------------------------------------- not run --
log "recording what could NOT be measured here"
cat > "$OUT/NOT-RUN.md" <<'EOF'
# Baseline gaps on this machine

These parts of the upstream suite need artifacts that do not fit here. They are
recorded as NOT RUN so a later "all green" cannot be mistaken for full coverage.

| test | needs | why not here |
|---|---|---|
| `test_expert` | `SHARD_DIR` (released safetensors shards) | checkpoint is 1.56 TB; 24.6 GB free |
| `test_real_layer` | `SHARD_DIR` | same |
| `tools/conform_all.py` | `SHARD_DIR` + torch | same, plus torch not installed |
| `test_tok` roundtrip | `tiktoken.model` (~2.8 MB, ships with checkpoint) | not downloaded |
| `make tok` parity | `tiktoken.model` + `tokenization_kimi.py` + `pip install tiktoken` | not downloaded |
| full K3 generation | 1.56 TB checkpoint + 109 GB packed trunk | not obtainable here |

The tokenizer gap is the only one that is cheap to close (~2.8 MB from
`moonshotai/Kimi-K3`). The rest are not, and the engine's own design anticipated
this: correctness is gated by weightless tests precisely so it does not depend on
having the checkpoint.
EOF
cat "$OUT/NOT-RUN.md"

log "baseline written to $OUT/"
ls -la "$OUT/"
echo
echo "make test exit code was: $TEST_RC"
exit "$TEST_RC"
