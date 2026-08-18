# multi-model-llm-engine

A small, portable, CPU-oriented LLM inference runtime in C99. Storage, memory,
caching, quantization, scheduling and kernels are generic; model architectures are
isolated behind a backend interface.

No BLAS. No framework. No GPU.

**The models do not have to fit in RAM.** Qwen3-8B (5.03 GB on disk) runs in a
2.34 GB memory budget on a 2-core laptop, by streaming layers from storage and
overlapping those reads with compute.

---

## Origin

This began as a fork of [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c),
which runs a 2.78-trillion-parameter Kimi K3 on one CPU in 8.24 GB of RAM. That project
solved the hard problem — executing a model far larger than memory — but its runtime and
its architecture were one piece of code.

The work here extracts the reusable half. The streaming, the expert cache, the low-memory
execution and the packed-weight kernels are upstream's ideas, preserved and generalised;
what is new is that they are now model-independent, with Qwen3 as the first architecture
added through the seam rather than into it.

Upstream's README is kept verbatim at [docs/README-kimi-k3.md](docs/README-kimi-k3.md).
See [NOTICE](NOTICE) for attribution and the list of changes.

---

## What works

| | Qwen3 | Kimi K3 |
|---|---|---|
| container | GGUF | safetensors + packed trunk |
| probe / inspect | yes | yes |
| execution | yes, through the backend interface | yes, via the original `bin/k3` |
| quantization | Q4_K, Q6_K | MXFP4, bf16 |
| attention | GQA 32/8, QK-norm, RoPE θ=1e6 | KDA recurrence + gated MLA |
| MLP | dense SwiGLU | 896-expert MoE, top-16 |
| numerics | fp32 accumulate | bit-identical across memory budgets |

K3's forward pass has not yet been migrated onto the new interface; it is still served by
the original CLI, and its backend declares that rather than pretending otherwise. Nothing
about K3's behaviour changed — the two determinism hashes that pin its output are checked
on every commit.

---

## Quick start

```bash
make                     # builds bin/engine and bin/k3
make test                # the correctness gate; needs no model weights

./bin/engine list                          # registered backends
./bin/engine inspect model.gguf            # architecture, cost, and the plan for this host
./bin/engine run model.gguf -p "Hello" -n 32
./bin/engine run model.gguf --memory 3G --threads 4 --context 2048
./bin/engine benchmark model.gguf --tokens 16
```

`--memory auto` (the default) sizes a budget from what the OS reports as available,
keeping the larger of 1 GB or 20% for the rest of the system. `inspect` explains what it
chose and why, and loads no weights.

---

## Design

```
src/
├── runtime/    hardware detection, memory budget, execution planner
├── tensor/     dtype registry, tensor descriptors (resident or file-backed)
├── kernels/    matmul, rmsnorm, rope, softmax, silu, swiglu; runtime AVX2 dispatch
├── quant/      quantization vtable: mxfp4, q8_0, q4_k, q6_k
├── storage/    file/mmap backends, LRU weight cache, pinned-prefix layer streamer
├── formats/    gguf, safetensors
├── models/     the backend interface and registry
│   ├── qwen3/  GQA + QK-norm + RoPE + SwiGLU
│   └── kimi/   KDA + MLA + latent MoE
├── tokenizer/  byte-level BPE
└── app/        the CLI
```

Four decisions are worth knowing about, because each was measured rather than assumed.

**The cache is LRU; the layer streamer deliberately is not.** A transformer walks its
layers in a fixed cycle, and LRU is the *worst* possible policy for that: with fewer slots
than layers, returning to layer 0 finds it exactly least-recently-used and just evicted,
so the hit rate is zero however much memory you add. The streamer pins a prefix instead,
giving a deterministic K/N hit rate where every extra gigabyte buys its share. Expert
routing is data-dependent and genuinely wants LRU, so both exist and the planner chooses.

**Numerical policy travels with the backend.** K3's claim is that its output is identical
whether it runs in 8 GB or 224 GB, which requires the vector path to be *bit-identical* to
the scalar path — costing half the vector width, since an AVX2 register holds 4 doubles
against 8 floats. Qwen3 makes no such claim and should not pay for it. So it is a
per-backend parameter, and the bit-identity gate compares raw bits rather than values
within a tolerance.

**Quantized weights are never materialised.** The primary operation is a fused dot
product over packed bytes. A matrix-vector product is memory-bound, so reading 7.5× fewer
bytes makes the fused kernel *faster* than dequantising first — it wins on memory and
speed at once.

**Reads overlap compute.** The layer walk order is fixed, so the next read is always known
and a prefetch can never be wrong. One reader thread loads layer L+1 while the CPU works
on layer L.

---

## Correctness

Generating fluent text is not evidence of correctness. A wrong RoPE base, the adjacent
instead of halved pairing, a missing QK-norm, or GQA head mapping done with `%` instead of
`/` all yield a model that runs and sounds fine. So every numerical claim here is checked
against an independent implementation.

**Qwen3, per tensor.** `tools/qwen3_ref.py` is a numpy forward pass written from the
architecture, sharing no code with the C. It dumps the hidden state after every layer plus
seven named intermediates inside layer 0. Measured through the streaming path:

| layer | 0 | 10 | 20 | 30 | 35 |
|---|---|---|---|---|---|
| relative RMS | 2.67e-06 | 3.34e-06 | 3.11e-06 | 3.10e-06 | 3.89e-06 |

49 tensors, worst 3.89e-06. The drift is *flat*, which is what fp32-versus-f64
accumulation looks like; a structural error is order-1 at the first affected tensor.

**Q4_K and Q6_K, bit-exact.** `tools/gguf_ref.py` decodes both formats independently.
Across 18 rows lifted from the real model: worst absolute difference **0.000e+00**. A
deliberate sabotage case — swapping every quant byte's nibbles — still produces finite,
in-range weights, which is exactly the failure this is built to catch.

**The container, structurally.** `data_start + Σ tensor bytes = 5,956,416 + 5,021,827,072
= 5,027,783,488`, the file size exactly. Every block-size constant feeds that sum.

**The same answer at every memory budget.** Every other test fixes one configuration and
checks the arithmetic within it; this one varies the configuration and checks the
arithmetic did not move. Three budgets producing structurally different plans — 13, 7 and
**zero** pinned layers, the last streaming every weight on every token — compared on raw
bits:

```
pinned 13 / 7 / 0        0 of 151,936 logits differ
```

Not the greedy token: an argmax is stable under small perturbations, so comparing tokens
would pass on a run whose distribution had visibly moved.

**K3, unchanged.** Two FNV1a hashes pin its kernel output and are checked on every commit.

`scripts/verify.sh` runs the whole suite plus ASan, UBSan, ThreadSanitizer and a `-Werror`
build.

---

## Measured performance

Reference host: Intel i5-6300U (2 cores / 4 threads, AVX2+FMA, no AVX-512), 7.86 GB RAM,
model on ext4. Qwen3-8B-Q4_K_M, 5.03 GB.

At a 2400 MB budget, 6 of 36 layers pinned: **33.2 s per generated token**, with the split
roughly 60% waiting on storage and 40% computing.

This is an I/O-bound configuration by construction — 30 of 36 layers are read from disk on
every token.

Vectorising the Q4_K and Q6_K dot products cut compute by **1.17×** (97.8 s → 83.7 s over
8 tokens). The gain is smaller than a scalar-to-AVX2 change suggests because the engine
requires implementations of a kernel to agree bit for bit, so the vector path accumulates
in double — 4 doubles per register against 8 floats. Buying that width back for Qwen3,
which makes no bit-identity claim, is the largest remaining compute lever and is item 1
in [the roadmap](docs/ROADMAP.md).

Two results worth recording because they were counter-intuitive:

- **Holding the 511 MB LM head resident is 2.57× slower**, not faster. It costs about four
  pinned layers, and a pinned layer is read on every step while the LM head is read only
  on steps that produce logits. An earlier A/B suggested the opposite; that comparison had
  varied the memory budget as well, and the faster arm simply had more memory.
- **More threads is not always better.** bf16 matmul saturates at 2–3 threads on this CPU
  and regresses at 4, while MXFP4 keeps scaling to 4 — because one moves 2 bytes per
  multiply-add and the other about 0.5. The planner picks from the model's bytes-per-weight
  rather than from the core count.

---

## Documentation

| | |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | how the pieces fit |
| [docs/adding-a-model.md](docs/adding-a-model.md) | **adding an architecture**, written from what Qwen3 cost |
| [docs/extending.md](docs/extending.md) | adding a quantization format, or a CPU kernel |
| [docs/PERFORMANCE.md](docs/PERFORMANCE.md) | measured figures, two machines, kept apart |
| [docs/architecture-report.md](docs/architecture-report.md) | the audit of upstream that shaped this design |
| [docs/qwen3-model-facts.md](docs/qwen3-model-facts.md) | Qwen3-8B as measured from the GGUF, not from documentation |
| [docs/baseline-m0.md](docs/baseline-m0.md) | the pre-refactor K3 baseline |
| [docs/ROADMAP.md](docs/ROADMAP.md) | what is next, with the reasoning |

Full index: [docs/README.md](docs/README.md).

---

## Requirements

Linux x86-64 with AVX2 and FMA for the vector paths; the portable scalar paths run
anywhere. GCC ≥ 9 or Clang ≥ 10, and OpenMP. Python 3 with numpy is needed only to
regenerate the reference fixtures, not to build or run.

Model weights are not included and none are distributed here.

## Licence

Apache 2.0, as upstream. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
