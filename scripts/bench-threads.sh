#!/usr/bin/env bash
# bench-threads.sh - thread scaling, measured in a way this host can support.
#
# WHY THIS IS NOT A for-LOOP AROUND OMP_NUM_THREADS
#   The same binary, run 15 times back to back on this machine, spanned 19.33-61.32 ms
#   on the bf16 matmul: a 3.17x spread with a 32% standard deviation. This is a 15 W
#   laptop part (i5-6300U) inside a VM, so it throttles and it shares the package with
#   the Windows host. A single-shot benchmark here cannot resolve anything smaller than
#   a 3x effect, which is larger than most of the effects worth measuring.
#
#   So: interleave the arms (drift hits all arms equally), take several reps, and quote
#   the MINIMUM. Minimum is the right statistic for a throttled machine because
#   interference only ever makes a run slower -- the fastest observed run is the one
#   least contaminated, and it is reproducible in a way the mean is not.
#
# Usage: scripts/bench-threads.sh [reps] [outfile]
set -u
REPS="${1:-7}"
OUT="${2:-baseline/thread-scaling.csv}"
BIN=bin/bench_kernels

[ -x "$BIN" ] || { echo "build $BIN first"; exit 1; }
mkdir -p "$(dirname "$OUT")"

echo "reps=$REPS, interleaved, quoting minimum per arm"
echo "rep,threads,bf16_ms,mxfp4_ms" > "$OUT"

for rep in $(seq 1 "$REPS"); do
    for t in 1 2 3 4; do
        out=$(OMP_NUM_THREADS=$t "$BIN" 2>&1)
        bf=$(echo "$out" | grep 'bf16 matmul'  | awk '{print $(NF-3)}')
        mx=$(echo "$out" | grep 'MXFP4 matmul' | awk '{print $(NF-3)}')
        echo "$rep,$t,$bf,$mx" >> "$OUT"
        printf '.'
    done
done
echo
echo

python3 - "$OUT" <<'EOF'
import csv, sys, statistics as st
rows = list(csv.DictReader(open(sys.argv[1])))
by = {}
for r in rows:
    by.setdefault(int(r['threads']), []).append(
        (float(r['bf16_ms']), float(r['mxfp4_ms'])))

print(f"{'thr':>3}  {'bf16 min':>9} {'med':>7} {'max':>7}   "
      f"{'mxfp4 min':>9} {'med':>7} {'max':>7}   {'speedup(bf16,min)':>18}")
base = None
for t in sorted(by):
    bf = sorted(x[0] for x in by[t])
    mx = sorted(x[1] for x in by[t])
    if base is None:
        base = bf[0]
    sp = base / bf[0]
    print(f"{t:>3}  {bf[0]:>9.2f} {st.median(bf):>7.2f} {bf[-1]:>7.2f}   "
          f"{mx[0]:>9.2f} {st.median(mx):>7.2f} {mx[-1]:>7.2f}   {sp:>17.2f}x")

print()
print("Read the min column. The med/max columns are shown so the noise stays visible:")
print("where an arm's min overlaps the next arm's min, the difference is not resolved.")
EOF
