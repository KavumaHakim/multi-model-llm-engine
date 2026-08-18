# Architecture

A generic runtime with model architectures isolated behind a backend interface. For how
Kimi K3 itself maps onto the codebase see
[architecture-kimi-k3.md](architecture-kimi-k3.md); for the audit of upstream that shaped
this design see [architecture-report.md](architecture-report.md).

## The split

```
                    ┌──────────────────────────────┐
   app/cli.c  ─────▶│  models/registry.c           │  probe → the right backend
                    └──────────────┬───────────────┘
                                   │ EngModelBackend vtable
                    ┌──────────────┴───────────────┐
                    │                              │
              models/qwen3/                   models/kimi/
              GQA · QK-norm                   KDA · MLA
              RoPE · SwiGLU                   latent MoE
                    │                              │
                    └──────────────┬───────────────┘
                                   │  uses, never reimplements
   ┌───────────────────────────────┴────────────────────────────────┐
   │  runtime/   hwinfo · memory budget · planner                   │
   │  tensor/    dtype registry · descriptors (resident or on disk) │
   │  storage/   file · mmap · weight cache · block streamer        │
   │  formats/   gguf · safetensors                                 │
   │  quant/     mxfp4 · q8_0 · q4_k · q6_k  (fused dot products)   │
   │  kernels/   matmul · rmsnorm · rope · softmax · silu · swiglu   │
   └────────────────────────────────────────────────────────────────┘
```

**The rule that keeps it honest:** everything below the line must link and pass its tests
with no backend present. `GENERIC_SRC` and `MODEL_SRC` are separate in the Makefile for
exactly that reason — when `models/` was briefly folded into `GENERIC_SRC`, every generic
test began dragging in the K3 backend, and the link error was really a layering violation.

## What a decode does

```
token ──▶ embedding row (read on demand)
            │
            ▼  for each layer L:
        streamer.get(L) ────────────┐   pinned prefix, or a ring slot
            │                       │
            ├─ streamer.prefetch(L+1)│   reader thread loads the next
            │                        │   while the CPU works on this
            ▼                        │
        backend-specific compute ◀───┘
            │  (attention, MLP — the backend's business entirely)
            ▼
        sequence state updated  ──▶ backend-owned layout
            │
            ▼
     final norm ─▶ LM head ─▶ logits    (only when ENG_DEC_LOGITS is set)
```

## Four decisions

Each was measured. Where a measurement contradicted the obvious argument, that is recorded
rather than quietly reversed.

### The cache is LRU; the streamer deliberately is not

A transformer walks its layers in a fixed cycle. LRU is the *worst possible* policy for
that: with fewer slots than layers, returning to layer 0 finds it exactly
least-recently-used and just evicted, so the hit rate is zero however much memory is added.
Every extra gigabyte buys nothing.

The streamer pins a **prefix** instead, giving a deterministic K/N hit rate. Measured in
`test_streamer.c`: 6 of 12 blocks pinned over 4 passes gives exactly 18 hits — `npin ×
(passes−1)`, with zero from the streamed tail.

Expert routing is data-dependent and genuinely wants LRU, so both exist and the planner
chooses. Neither pretends to cover the other case.

### Numerical policy travels with the backend

K3 claims its output is identical whether it runs in 8 GB or 224 GB. That requires the
vector path to be *bit-identical* to the scalar path — not close — and costs half the
vector width, since an AVX2 register holds 4 doubles against 8 floats.

Qwen3 makes no such claim. Forcing K3's policy on it halves its speed for a guarantee it
does not need; forcing Qwen3's on K3 silently breaks the one claim K3 makes. So it is a
per-backend parameter, and the gate compares raw bits rather than values within a
tolerance — a tolerance would pass an implementation that is merely close.

### Weights are never materialised

The quantization vtable's primary operation is a fused dot product over packed bytes. One
K3 expert is 132 MB as fp32 against 17.55 MB packed, and a token touches 1,472 of them.
Since a matrix-vector product is memory-bound, reading 7.5× fewer bytes makes the fused
kernel *faster* than dequantising first — it wins on both axes at once.

### Reads overlap compute

The layer walk order is fixed, so the next read is always known and a prefetch can never be
wrong. One reader thread loads layer L+1 while the CPU computes layer L.

**With one ring slot the reader is disabled, and that is correctness, not tuning.** It
would otherwise read the next block over the one being computed on: the read succeeds, no
pointer changes, and the model emits fluent wrong output. Upstream measured a prompt
silently changing from `17374 20829 10 …` to `32609 2329 146429 …` with no diagnostic.

## Memory

One budget, partitioned in a fixed priority order:

```
reserve      OS, allocator, and headroom for the plan's own error
resident     weights that cannot stream — taken first, since a model
             that cannot hold these cannot run at any budget
activations  hidden states and per-step intermediates
scratch      kernel working buffers
kv_cache     grows with context; reduced rather than refused if it will not fit
weights      the remainder, split between streamer and cache
```

The order is the point. If the remainder cannot hold one layer, that is a refusal at plan
time with both figures side by side, rather than an OOM kill mid-token.

## What holds it together

| gate | what it catches |
|---|---|
| per-tensor reference comparison | architecture errors, located to a layer |
| bit-identity between kernel implementations | a vector path that is merely close |
| streaming equivalence across budgets | residency changing the answer |
| two FNV1a hashes | any change to K3's kernel output |
| ASan · UBSan · TSan · `-Werror` | memory errors, races, warnings |

The streaming-equivalence gate is the one that covers this design specifically: every other
test fixes a configuration and checks the arithmetic within it. Measured at 13, 7 and
**zero** pinned layers — the last streaming every weight on every token — 0 of 151,936
logits differ.
