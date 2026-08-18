# M0 — known-good K3 baseline

Captured before any production change, per §3 of the modernization brief. Raw
artifacts are in [`baseline/`](../baseline/); this file is the reading of them.

Reproduce with `scripts/baseline.sh`.

| | |
|---|---|
| commit | `0cfde2f` (upstream v1.0.0 + docs) |
| host | Intel i5-6300U (Skylake, 2 physical / 4 logical, 15 W), AVX2 + FMA, no AVX-512 |
| platform | WSL2, Ubuntu 24.04.4, kernel 6.18.33.2-microsoft-standard-WSL2 |
| toolchain | gcc 13.3.0, GNU make, OpenMP (libgomp) |
| memory visible to build | 3.84 GB (WSL default = 50% of the 7.86 GB host) |
| build | `make ARCH="-mavx2 -mfma"` — the portable baseline upstream CI uses |
| **`make test` exit code** | **0** |

## Correctness — all gates green

**End-to-end oracle** (`tests/unit/k3_model.c`, tiny model with K3's exact tensor
graph). All three must be exact; there is no tolerance on token identity:

```
GATE 1  teacher forcing : 32/32 positions match tf_pred
GATE 2  greedy decode   : 20/20 generated tokens match full_ids
GATE 3  incremental     : 20/20 generated tokens match full_ids   (KV cache + carried KDA state)
```

GATE 3 is the important one for this project: it proves incremental decode with a KV
cache and carried recurrent state produces *the same tokens* as full recompute. That
equivalence is what the streaming and caching work must not break.

**Kernel gates** (`test_ops`, tolerance `atol=1e-5 rtol=1e-4` from `MANIFEST.json`):
22 fixtures, all PASS. Worst case across the whole suite is **0.58× tolerance**
(`layer_kda`), so there is roughly 1.7× headroom before anything goes yellow. Two
results are stronger than a tolerance:

- `mxfp4` — **EXACT** on real released-checkpoint bytes, 64 rows × 3584 elements.
- `matmul_bf16` — **bit-identical** to `k3_matmul`, asserted rather than approximated.

**Other gates:** streaming cache 5/5 (including "prefetch_reads ≤ hits" and byte-exact
serial/batch/mixed reads), safetensors reader self-checks, config reader accepts the
fixture and refuses all three malformed configs, `scale_test` at real 7168-wide
dimensions (1.77 GB allocation) passes.

**Not run:** tokenizer roundtrip and parity — `tiktoken.model` ships with the 1.56 TB
checkpoint, not the repo. Reported as NOT RUN rather than passed quietly. Full list of
gaps in [`baseline/NOT-RUN.md`](../baseline/NOT-RUN.md).

**Sanitizers:** ASan + UBSan over `test_ops`, `test_st`, `test_cfg`, `k3_model` —
**no memory errors and no UB reported**. Only exit-time leaks in test binaries that
never free fixtures, which is not a correctness defect. This matters because the
refactor is about to start moving memory ownership around; the baseline is clean.

## The measurement noise floor — read this before trusting any benchmark

The same binary, 15 consecutive runs, no changes:

| kernel | min | median | max | sd | max/min |
|---|---:|---:|---:|---:|---:|
| bf16 matmul | 19.33 ms | 32.52 ms | 61.32 ms | 10.6 ms (32%) | **3.17×** |
| MXFP4 matmul | 5.42 ms | 9.35 ms | 21.40 ms | 4.0 ms (39%) | **3.95×** |

A 15 W laptop part inside a VM, sharing the package with the Windows host, throttles
hard. Load average sat at 1.06–1.41 throughout with nothing of ours running.

**Consequence: a single-shot benchmark on this host cannot resolve anything smaller
than a ~3× effect.** Every performance number in this project must therefore be
best-of-N, interleaved across arms, quoting the **minimum** — minimum is the right
statistic here because interference only ever makes a run slower, so the fastest
observed run is the least contaminated and the most reproducible.

`scripts/bench-threads.sh` implements this. Raw data:
[`baseline/thread-scaling.csv`](../baseline/thread-scaling.csv).

### A correction this discipline caught immediately

The first sequential pass measured `-march=native` at 62.16 ms against 42.68 ms for
`-mavx2 -mfma` and appeared to show native **1.46× slower**, contradicting upstream's
Makefile comment that native is "a real win on the expert matmuls".

Interleaving the two builds over 5 reps showed the difference is noise. The two
binaries are **byte-identical in size** (84,584 bytes), and `gcc -Q --help=target`
confirms `-march=native` on this Skylake part enables exactly the same feature set as
`-mavx2 -mfma` — there is no AVX-512 to unlock, so only `-mtune` differs and it changes
nothing measurable here. Output FNV1a hashes match across both arms.

Neither claim — "native is faster" or "native is slower" — is supportable on this
machine. Recorded so it is not re-litigated later.

## Thread scaling — 7 reps, interleaved, minimum

```
thr   bf16 min     med     max   mxfp4 min     med     max   speedup(bf16)
  1      26.12   29.85   50.83        9.05    9.51   19.32          1.00x
  2      15.30   16.74   23.86        4.69    5.45   10.26          1.71x
  3      13.88   21.30   33.76        3.59    8.80   13.98          1.88x
  4      14.09   17.35   26.55        3.18    4.79    8.23          1.85x
```

The brief warns against assuming more threads is faster. Measured, on 2 physical cores:

- **bf16 matmul saturates at 2–3 threads.** 1→2 buys 1.71×; 2→3 buys a further 1.10×;
  3→4 is a 1.5% *regression* and the two arms' minima (13.88 vs 14.09) overlap well
  inside noise. Beyond two threads this kernel is **memory-bandwidth-bound**, and the
  hyperthread pairs contend for the same L2 and the same memory port rather than adding
  throughput.
- **MXFP4 matmul keeps scaling to 4 threads**: 1.93× → 2.52× → 2.85×. It reads ~7.5×
  fewer bytes per useful multiply, so it sits further from the bandwidth roof and has
  real work for the hyperthreads to hide.

**The two kernels want different thread counts on the same CPU.** That is a direct
input to the adaptive planner (§15 of the brief): thread count is not a single global
setting to be maxed out, it is per-kernel and depends on whether that kernel is
bandwidth-bound or compute-bound. A planner that just sets `nproc` leaves ~10% on the
table for bf16 and is roughly right only by accident.

Raw data: [`baseline/thread-scaling.csv`](../baseline/thread-scaling.csv).

## Determinism invariants to preserve

These hashes are emitted by `bench_kernels` and were **identical across every thread
count (1/2/3/4) and both build arms**:

```
bf16  OUTPUT FNV1a = d65cab2d141bb3b8
mxfp4 OUTPUT FNV1a = a231061237b5579d
```

This is K3's central guarantee — same output regardless of how much parallelism or
memory it is given — reduced to two 64-bit numbers. **Any refactor that changes either
hash has broken K3, whatever the tests say.** They are cheap to check and belong in
regression runs from here on.

## What this baseline does and does not cover

Covered: all kernels, the cache, the safetensors reader, the config reader, real-width
allocation, and end-to-end token identity across three decode strategies — with no
model weights, which is a deliberate property of the upstream test design.

Not covered here, and not obtainable on this machine: anything requiring the released
checkpoint (1.56 TB) or the packed trunk (109 GB) against 24.6 GB of free disk. So the
*storage-level* behaviours — real trunk streaming, real expert cache hit rates, real
peak RSS at a given budget — are baselined by upstream's published measurements
(`docs/data/`) and not independently reproduced. Qwen3-8B at 4.68 GB **will** exercise
those paths for real, which is why it is the right first backend for this work.
