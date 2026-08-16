#!/usr/bin/env bash
# verify.sh - the gates every milestone must pass.
#
# Run after any change to the engine:  scripts/verify.sh
#
# THREADSANITIZER NEEDS ASLR DISABLED ON THIS KERNEL.
#   Under WSL2 (and some hardened Linux kernels) TSan fails to start with
#       FATAL: ThreadSanitizer: unexpected memory mapping 0x...
#   because it cannot claim its shadow-memory region against the process's randomised
#   layout. It is intermittent -- measured here, 3 of 5 runs aborted at startup and the
#   others segfaulted, while a sixth run happened to succeed and looked like a pass.
#
#   That is a tool-startup failure, not a finding, and it is dangerous precisely because
#   the exit code (66, or 139 on the segfault path) is indistinguishable from a real
#   race unless you read the stderr. `setarch -R` disables randomisation for the child
#   and makes the run deterministic: 5 of 5 clean afterwards.
#
#   So: never treat a non-zero TSan exit as a race without checking for "FATAL:".
set -u
cd "$(dirname "$0")/.." || exit 1
rc=0

# The two hashes bench_kernels prints. They are K3's "identical output at any memory
# budget" guarantee reduced to two numbers, and they held across every thread count and
# both build arms at M0. A refactor that changes either has broken K3, whatever the
# tests say.
BF16_HASH=d65cab2d141bb3b8
MXFP4_HASH=a231061237b5579d

section() { printf '\n############ %s ############\n' "$1"; }

# CLEAN FIRST. The sanitizer sections below rebuild objects with -fsanitize=..., and a
# later plain build that reuses them fails to link with "undefined reference to
# __tsan_init". Starting dirty made this script's own first section fail on a tree that
# was perfectly fine.
make clean >/dev/null 2>&1

section "full test suite"
make ARCH="-mavx2 -mfma" test -j"$(nproc)" 2>&1 | tail -4
tr=${PIPESTATUS[0]}
[ "$tr" = "0" ] && echo "  OK   make test" || { echo "  FAIL make test"; rc=1; }

section "determinism invariants"
make ARCH="-mavx2 -mfma" bin/bench_kernels -j"$(nproc)" >/dev/null 2>&1
got=$(./bin/bench_kernels 2>&1 | grep FNV1a)
echo "$got"
bf=$(echo "$got" | grep bf16  | awk '{print $NF}')
mx=$(echo "$got" | grep mxfp4 | awk '{print $NF}')
[ "$bf" = "$BF16_HASH" ]  && echo "  OK   bf16 hash unchanged"  || { echo "  BROKEN bf16 hash"; rc=1; }
[ "$mx" = "$MXFP4_HASH" ] && echo "  OK   mxfp4 hash unchanged" || { echo "  BROKEN mxfp4 hash"; rc=1; }

section "ASan + UBSan"
make clean >/dev/null 2>&1
SAN="-fsanitize=address,undefined -fno-omit-frame-pointer"
CF="-O1 -g -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter $SAN -ffp-contract=off"
make CFLAGS="$CF" LDFLAGS="-lm $SAN" ARCH= \
     bin/test_streamer bin/test_cache_generic bin/test_cache bin/test_tensor \
     bin/test_planner bin/test_kernels bin/test_quant bin/test_ops bin/k3_model \
     -j"$(nproc)" >/dev/null 2>&1
# LEAK POLICY IS PER TEST, and the split is meaningful rather than convenient.
#
#   leaks=1  the tests written for the generic layer. These own every allocation they
#            make and must exit clean, so a leak here is a real defect in new code.
#   leaks=0  the upstream K3 tests. They load fixtures and exit without freeing them by
#            design (test_ops leaks ~65 MB of parsed JSON), so leak detection reports
#            the test harness, not the engine. Memory ERRORS -- overflow, use-after-free,
#            UB -- are still checked in both cases, which is what this gate is for.
for t in "test_streamer|build|1" \
         "test_cache_generic||1" \
         "test_cache|tests/fixtures/cache|1" \
         "test_tensor|tests/fixtures/st build|1" \
         "test_planner||1" \
         "test_kernels||1" \
         "test_quant||1" \
         "test_ops|tests/fixtures/ops|0" \
         "k3_model|tests/fixtures|0"; do
    n=${t%%|*}; rest=${t#*|}; a=${rest%|*}; leaks=${rest##*|}
    [ -x "bin/$n" ] || { echo "  BUILD FAILED: $n"; rc=1; continue; }
    # shellcheck disable=SC2086
    ASAN_OPTIONS="detect_leaks=$leaks" ./bin/$n $a >/dev/null 2>"/tmp/san_$n.err"; ex=$?
    # grep's status, not head's: a pipeline ending in head returns 0 either way, which
    # made an earlier version of this script report findings that did not exist.
    if grep -qE 'ERROR: AddressSanitizer|runtime error' "/tmp/san_$n.err"; then
        echo "  FINDINGS in $n:"
        grep -E 'ERROR: AddressSanitizer|runtime error' "/tmp/san_$n.err" | head -5
        rc=1
    elif [ "$ex" != "0" ]; then
        echo "  FAIL $n exited $ex with no sanitizer finding (test failure)"
        rc=1
    else
        echo "  OK   $n (leak check $([ "$leaks" = 1 ] && echo on || echo off))"
    fi
done

section "ThreadSanitizer (streamer reader thread)"
make clean >/dev/null 2>&1
TS="-fsanitize=thread -fno-omit-frame-pointer"
make CFLAGS="-O1 -g -std=gnu99 -Wall -Wextra -Wno-unused-parameter $TS -ffp-contract=off" \
     LDFLAGS="-lm $TS" ARCH= bin/test_streamer -j"$(nproc)" >/dev/null 2>&1
if [ -x bin/test_streamer ]; then
    # setarch -R: see the header comment. Without it this is a coin flip.
    setarch "$(uname -m)" -R ./bin/test_streamer build >/dev/null 2>/tmp/tsan.err
    ex=$?
    if grep -q "FATAL: ThreadSanitizer" /tmp/tsan.err; then
        echo "  INCONCLUSIVE: TSan could not start (ASLR); not a finding"
        head -2 /tmp/tsan.err
    elif grep -q "WARNING: ThreadSanitizer" /tmp/tsan.err; then
        echo "  RACE DETECTED:"
        grep -A6 "WARNING: ThreadSanitizer" /tmp/tsan.err | head -20
        rc=1
    else
        echo "  OK   no races (exit $ex)"
        [ "$ex" != "0" ] && rc=1
    fi
else
    echo "  TSan build unavailable (skipped)"
fi

section "-Werror, engine + CLI"
make clean >/dev/null 2>&1
if make CFLAGS="-O2 -std=gnu99 -Wall -Wextra -Wpointer-arith -Wshadow -Wvla -Wno-unused-parameter -Werror -fopenmp -ffp-contract=off" \
        ARCH="-mavx2 -mfma" all -j"$(nproc)" >/dev/null 2>&1; then
    echo "  OK   bin/k3 builds clean under -Werror"
else
    echo "  FAIL -Werror build"; rc=1
fi

printf '\nVERIFY EXIT: %d\n' "$rc"
exit $rc
