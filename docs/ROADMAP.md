# Roadmap

Ordered by value, with the reasoning stated so the order can be argued with.

## 1. A numerical policy for the quantization vtable

The k-quant vector paths accumulate in double because quant.h requires every
implementation of a kernel to agree with its reference bit for bit. That costs half the
register width -- 4 doubles against 8 floats -- and measurement puts the resulting AVX2
gain at 1.17x on compute rather than the 3-4x the change would otherwise suggest
(docs/PERFORMANCE.md).

K3 needs that guarantee: its claim is that output does not depend on the memory budget.
Qwen3 declares ENG_NUM_FAST and needs no such thing. The kernels layer already takes
numerical policy as a per-call parameter; the quantization vtable does not, so
`dot_row` cannot express "fp32 is fine here". Threading it through would let Qwen3 use
the full width while K3 keeps its exactness.

This is the largest single lever left on compute, and it is a design change rather than
a tweak, which is why it is written down instead of done quietly.

## 2. docs/adding-a-model.md

Required by the brief and still missing. It is also the only real test of whether the
backend interface is usable by someone who did not write it.

## 3. Migrating K3's forward pass onto the backend interface

K3's storage, cache, streamer and quantization are already generic; its execution still
runs through bin/k3. Deferred rather than forgotten: the 1.56 TB checkpoint cannot run
on the development machine, so the work would be unverifiable here. Worth doing on a
host that can execute it, and not before.

## 4. Serving

No HTTP API. Deliberately last: it is product surface, and everything above changes
what would be served.

## Inherited from kimi-k3-in-c, still open

## 1. Chunked prefill, the highest-value missing piece

Prefill currently runs as a single forward pass over the whole prompt, and attention in
the 24 MLA layers is quadratic in sequence length. The practical effect: the context
ceiling is 32k tokens, but a 21k-token prompt does not complete in reasonable time.

Raising the ceiling was necessary and is done. This is the part that makes it usable.

## 2. Re-run the published campaign under the replicating harness

The measured noise floor is 33%, and almost every published figure is a single sample.
The harness itself is done, `benchmarks/memory-ladder.sh` and `benchmarks/split-sweep.sh`
already default to 3 repeats and report mean, sd and spread. What remains is re-running
the 12 ladder rungs and the 12 splits under it and replacing the single-sample tables in
docs/data/ with replicated ones.

## 3. Thread scaling

`OMP_NUM_THREADS` has never been swept on this engine. The workload is I/O bound at low
memory budgets, so the useful thread count is probably well below the core count, and
on memory-bound workloads throughput often *declines* past a point. Unknown here.

## 4. SIMD in the KDA recurrence

The bf16 trunk matmul and the MXFP4 expert matmul already have hand-written AVX2 paths
(`src/core/k3_ops.c`), each written to reproduce the scalar reduction order exactly. The
KDA recurrence does not: it is still plain scalar C, and it is the largest remaining
un-vectorised kernel on the non-I/O path.

## 5. Sampling

Greedy only today. Adding temperature and top-p is small, but note the trade-off: greedy
decoding is what makes output identical across memory budgets, which is a property the
test-suite depends on. Sampling must be opt-in and off by default.

## 6. Chat template

K3 ships an XTML chat format. Without it the engine produces base-model continuations,
which is why asking a question gets the question completed rather than answered.

## 7. Vision

K3 is natively multimodal. The vision tower is 27 layers and ~0.4B parameters, small,
and its weights are ~0.9 GB, a fraction of a percent of the checkpoint. Self-contained
enough to be tractable.

## 8. Serving

No HTTP API. Deliberately last: it is product surface, and everything above changes what
would be served.

## Explicitly not planned

**A precision dial for the trunk.** The trunk is streamed losslessly rather than
quantised, and that is a design decision, not an omission. Post-hoc int4 measures ~17%
mean relative weight error on K3 attention tensors against ~1% for int8; streaming
costs time, which more memory buys back, while rounding costs accuracy, which nothing
buys back.

**GPU support.** Out of scope for this project.
