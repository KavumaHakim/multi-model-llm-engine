# Architecture & Dependency Report

**Subject:** `FareedKhan-dev/kimi-k3-in-c` @ `ff11dce` (tag v1.0.0)
**Purpose:** audit before refactoring the K3 engine into a generic multi-model CPU
inference runtime whose first new backend is Qwen3-8B.
**Status:** audit complete. No production code has been modified.

This report is the gate described in §2 of the project brief. It answers the twelve
required questions, and adds an environment section because the build environment
turns out to be a hard constraint on the plan.

---

## 0. What was audited

The full repository was cloned and read. 138 files, 24.5 MB checked out
(`docs/images/` and the 2.5 MB tech-report PDF were excluded from checkout — they are
binary assets with no bearing on build, test, or behaviour).

The engine itself is small. That is the single most important finding for planning
purposes:

| area | files | bytes of C |
|---|---:|---:|
| kernels (`src/core/k3_ops.c`) | 1 | 67,801 |
| CLI + driver (`src/cli/k3_run.c`) | 1 | 76,174 |
| storage (`src/io/`) | 5 | 60,281 |
| expert cache (`src/cache/`) | 2 | 25,037 |
| weight binding (`src/model/`) | 2 | 28,356 |
| tokenizer (`src/tokenizer/k3_tok.h`) | 1 | 9,705 |
| public headers (`include/k3/`) | 2 | 42,809 |
| **engine total** | **14** | **~310 KB** |
| tests | 9 | 128,011 |
| Python tooling (`tools/`) | 23 | ~205 KB |

~310 KB of engine C. The refactor is therefore a tractable restructuring job, not a
rewrite of a large system. The documentation-to-code ratio is unusually high (the
README alone is 146 KB) and the header comments carry real engineering rationale —
these are worth mining, not discarding.

Everything below is drawn from reading the source, not from the README's claims.

---

## 1. Current execution pipeline

Entry point is `main()` in `src/cli/k3_run.c:554`. The flow:

```
argv parse ─→ preset/budget resolution
      │
      ├─→ k3_cfg_read()          config.json → K3Cfg. Refuses missing fields.
      ├─→ k3_st_open(dir)        index every tensor across 96 safetensors shards
      │                          into an open-addressed hash (497,220 tensors)
      ├─→ k3_bind_model()        embed, final norm, lm_head, model-level attn-res
      ├─→ k3_trunk_open()        [if --trunk] open packed trunk.bin, size pin+ring
      ├─→ k3_bind_layer()        [resident path] per-layer tensors → K3LayerBind
      ├─→ k3_cache_init()        routed-expert LRU arena
      └─→ k3_tok_load()          [if text prompt] tiktoken vocabulary
      │
      ▼
  decode loop  ── per generated token ──→ forward()      (k3_run.c:470)
```

`forward()` (`k3_run.c:473-552`) is the whole model:

```
for t in 0..T-1:  k3_embed_row(h[t], embed, ids[t])        gather, widen bf16→f32
memset(block_residual)
for L in 0..n_layers-1:
    if streaming:  k3_trunk_bind(tr, L, &lay[L])           make layer L resident
                   k3_trunk_prefetch(tr, L+1)              async read of L+1
    if lay[L].moe: moe.src = &cache->src;  moe.layer = L   attach expert source
    k3_decoder_layer_inc(h, br, &nb, &lay[L].lay, c, L, T, kstate+L*kper,
                         scratch, kvc, ropec, cached, cap)
k3_attn_res(...)                                           model-level aggregator
k3_rmsnorm(nrm, h[T-1], norm)
k3_mmw(logits, nrm, lm_head)                               → argmax → next token
```

And one decoder layer (`k3_ops.c:925` `k3_decoder_layer_inc`) reproduces the
reference's `_forward_attn_residual` statement for statement:

```
prefix_sum = h
if block stack non-empty:  h = attn_res([blocks…, prefix_sum], self_attention_res)
if layer_idx % attn_res_block == 0:  push prefix_sum; prefix_sum = NONE
h = input_layernorm(h)
h = KDA(h)  or  MLA(h)                       69 KDA layers / 24 MLA layers
prefix_sum = (prefix_sum==NONE) ? h : prefix_sum + h
h = attn_res([blocks…, prefix_sum], mlp_res)  unconditional
h = post_attention_layernorm(h)
h = MoE(h)  or  dense_mlp(h)                  layer 0 is dense
prefix_sum = (prefix_sum==NONE) ? h : prefix_sum + h
```

Two decode strategies exist. The default **re-runs the entire prefix every step**
(O(T²)) because that is the path the tiny-model oracle validates. `--incremental`
prefills then decodes one token at a time, carrying the KDA recurrent state and an MLA
KV cache; GATE 3 of `tests/unit/k3_model.c` asserts the two produce *identical tokens*.

There is also a speculative path (`--spec N`, n-gram draft + batched verify) and a
hybrid draft-model path (`--draft-trunk`, an int8 trunk proposes, the exact bf16 model
verifies). Both are exactness-preserving by construction: the draft only proposes.

**Assessment.** The pipeline is a hardcoded K3 graph. There is no model abstraction of
any kind — `forward()` calls `k3_decoder_layer_inc` which branches on
`k3_is_mla(c, L)`. This is precisely the layer that must become a backend interface.

---

## 2. Current memory-management design

There is **no memory manager**. There is a *budget-splitting convention* plus direct
`malloc`/`posix_memalign` at four sites:

| consumer | allocation | sized by |
|---|---|---|
| trunk pin + ring | `K3Trunk.pin[]` (exact per layer) + `arena` (uniform slots) | `--trunk-gb` |
| expert cache | `K3Cache.arena`, `nslot × slot_bytes`, page aligned | `--cache-gb` |
| KV cache | `kvc`, `ropec` in `Weights` | context × 2.37 MB/pos |
| scratch | one `float*` sized by `k3_*_scratch()` helpers | max over layers |

Notable properties that are genuinely good and worth preserving:

- **Every kernel declares its scratch requirement** via a `k3_*_scratch(cfg, T)`
  function rather than allocating internally. `k3_mla_scratch`, `k3_kda_scratch`,
  `k3_moe_scratch`, `k3_layer_scratch`. No kernel mallocs. This is exactly the
  discipline a real memory manager needs, and it is already there.
- **Two storage classes, deliberately chosen** (`k3_bind.h:33-45`): big matrices keep
  the checkpoint's bf16 bytes and are widened *inside* the matmul; small vectors read
  elementwise (norm gains, `A_log`, `dt_bias`, conv kernels, router gate) are widened
  to fp32 at bind time. The rationale — a silent dtype error in an elementwise read is
  not a crash but a different model — is sound and should carry over.
- Trunk slots are **not** uniform: pinned layers get exact-size allocations because
  layer 0's dense MLP is 2.34 GB against 1.27 GB for a normal layer, and uniform
  slots would waste half the budget.

What is missing relative to the brief:

- No partitioning of a *single* budget into reserve / activations / KV / scratch /
  weight cache. The user hands out `--trunk-gb` and `--cache-gb` independently and
  nothing stops their sum exceeding RAM.
- `mem_available_bytes()` reads `/proc/meminfo` `MemAvailable` — Linux only, returns
  0.0 elsewhere, and `--preset auto` degrades silently when it does.
- The pre-run banner is explicitly documented as a *plan, not a measurement*, and is
  known to overstate by 0.13–1.84 GB. Peak RSS comes from `getrusage` afterwards.
- No accounting of where memory actually went beyond that banner.

---

## 3. Current storage / streaming design

Three cooperating pieces.

### 3.1 `k3_st.c` — safetensors reader (hand-written, no dependency)

Format handling is correct and complete: `[8-byte LE header length N][N bytes JSON]
[tensor data]`, `data_offsets` relative to end of header, absolute offset `8+N+start`.
Handles F32/BF16/F16/U8 plus a private `I8R` per-row int8 draft dtype.

The two design decisions that matter:

- **Open-addressed hash index over all 497,220 tensors.** With 6 lookups per expert ×
  16 experts × 92 layers per token, a linear scan is not a slow path, it is a
  non-starter. Lookup must be O(1).
- **`pread`, deliberately not `mmap`** (`k3_st.h:22-25`). Pages read into engine-owned
  buffers never become file-backed mappings counted against the process, so peak RSS
  tracks what is really resident. `mmap` would make the RSS number meaningless against
  a 1.56 TB checkpoint.

`k3_st_read_aligned()` implements O_DIRECT: since an expert run starts wherever the
checkpoint put it, the read is *widened outward* to the enclosing 4096-aligned window
and the caller is told where the payload begins inside the buffer. Falls back to
buffered reads when O_DIRECT is unavailable.

### 3.2 `k3_load.c` — coalesced expert reads

Built on a measurement: the six tensors of one routed expert (`w1/w2/w3` × packed +
scale) form **one contiguous 17,547,264-byte run** in shard order
`w1.packed w1.scale w2.packed w2.scale w3.packed w3.scale`. So an expert fetch is one
`pread`, not six. The loader *verifies* contiguity per expert rather than assuming it,
and falls back to six preads when it does not hold. On seek-bound storage that is 1,472
seeks per token instead of 8,832.

### 3.3 `k3_trunk.c` — trunk streaming

`tools/pack_trunk.py` copies each layer's contiguous trunk run into a single
`trunk.bin` with offsets in `trunk.json`, padded to 4096 for O_DIRECT. Loading a layer
is then one `pread` from a known offset.

The replacement policy here is the interesting part, and the reasoning is documented at
`k3_trunk.h:38-47`: **a cyclic sequential scan is the classic LRU pathology.** With
N < 93 slots, by the time the walk returns to layer 0 it is exactly the least recently
used thing and has just been evicted — hit rate is *zero* no matter how much RAM is
added. So the trunk **pins a prefix** of K layers and streams the rest through a small
ring: hit rate is exactly K/93, deterministically, and every extra gigabyte buys its
fair share. The expert cache keeps LRU because expert reuse is data-dependent — the
opposite situation.

Prefetch: `k3_trunk_prefetch(tr, L+1)` hands layer L+1 to a pthread reader while the
main thread computes layer L. Safe because the walk order is fixed 0..92 every token,
so there is nothing to predict. Measured 71.75 → 42.27 s/token (1.70×) at the laptop
preset. Needs a second ring slot (2.37 GB at the floor), so it is only enabled when the
budget can pay; `k3_trunk_open` says so on stdout when it cannot.

**Assessment.** This subsystem is the most valuable thing in the repository and is
already ~80% generic. The concepts — contiguity verification, coalesced runs, aligned
O_DIRECT with payload offsets, pin-prefix-plus-ring for cyclic access, fixed-order
prefetch — are architecture-independent. What is K3-specific is only the *naming and
grouping* (`k3_expert_ref`, the six-tensor run) and the assumption of a per-layer run.

---

## 4. Current cache design

`src/cache/k3_cache.c`, one cache: routed experts, LRU with pinning.

```c
K3Cache {
    K3ExpertSrc src;          // MUST be first: &cache->src is passed as the interface
    unsigned char *arena;     // nslot × slot_bytes, page aligned
    int32_t *slot_of;         // [n_layers*n_experts] → slot, or -1
    int32_t *key_of;          // [nslot] → layer*n_experts+expert, EMPTY(-1)/INFLIGHT(-2)
    uint64_t *used_at;        // LRU stamps
    unsigned char *pinned;    // never evict while set
    K3ExpertRef *ref;         // geometry of the resident expert
    int32_t *pad;             // payload offset within slot (O_DIRECT widening)
    // stats: hits, misses, evictions, bytes_read, prefetch_reads, load_seconds
    uint32_t *hist;           // [n_layers*n_experts] request counts
    int32_t *trace;           // (layer,expert) in request order
}
```

Design points worth carrying forward verbatim:

- **The cache holds MXFP4, never floats.** Dequantised an expert is 132 MB vs
  17.55 MB packed; caching floats would cut resident experts 7.5× for no benefit,
  since `k3_matmul_mxfp4` consumes the packed form directly and matrix-vector is
  memory-bound. Reading a seventh of the bytes is *faster*.
- **Capacity is enforced > topk at construction.** `k3_moe` fetches an expert and
  immediately uses it, so a handed-out slot cannot be evicted before use *as long as*
  capacity exceeds topk. The constructor enforces rather than trusting.
- **Victim search is a linear scan**, justified explicitly: a few hundred comparisons
  against a 17.55 MB read is not worth a heap.
- **`INFLIGHT` is a distinct sentinel from `EMPTY`** so `pick_victim` cannot hand out a
  slot whose read has not landed.
- **The histogram is not instrumentation** (`k3_cache.h:27-31`) — which experts are hot
  is the *input* to any pinning strategy, and 82,432 counters cost 330 KB.
- **The access trace enables offline policy simulation.** Routing does not depend on
  the cache, so one expensive run yields the entire hit-rate-vs-capacity curve;
  `tools/sim_cache.py` replays it at any capacity. This is a genuinely good idea and is
  how the Belady-vs-LRU comparison in the docs was produced.
- Batch prefetch (`getmany`) issues the top-k reads concurrently under
  `#pragma omp parallel for schedule(dynamic,1)`, taking queue depth from 1 to 16.
  `prefetch_reads` is counted separately *because otherwise the hit rate is a lie* — a
  prefetched expert is resident when `get()` asks, so it records a hit, but the bytes
  still moved. Effective hit rate is `(hits - prefetch_reads) / requests`.

**Assessment.** Structurally this is already a generic cache with an
expert-shaped key. Generalizing it means replacing the `(layer, expert)` key with an
opaque key and the `K3ExpertRef` payload with a generic descriptor. The policy,
statistics, pinning, tracing and inflight handling all transfer unchanged.

`K3ExpertSrc` (`k3.h:332-355`) is already a **vtable**: `get`, optional `getmany`,
optional `resident`, plus `void *ctx`. It is the one abstraction boundary the codebase
already has, and it is the right shape.

---

## 5. Current tensor representation

**There is no tensor type.** This is the largest architectural gap.

What exists instead is three separate, partial representations:

1. **`K3Tensor`** (`k3_st.h:38-46`) — file metadata only: name, shard index, dtype,
   ndim, `shape[4]`, absolute offset, nbytes. No strides, no residency, no data
   pointer. It describes bytes on disk and nothing else.
2. **Tagged raw pointers** — every kernel takes `const void *W` plus an `int wdt`
   (`K3_WF32=0`, `K3_WBF16=1`, `K3_WI8=2`) and shapes are passed as separate `int in,
   int out` arguments. `k3_mmw()` is the dispatch:
   ```c
   if (wdt == K3_WBF16)     k3_matmul_bf16(...);
   else if (wdt == K3_WI8)  k3_matmul_q8(...);
   else                     k3_matmul(...);
   ```
3. **Per-role weight structs** — `K3KdaW`, `K3MlaW`, `K3MoeW`, `K3LayerW` are bundles
   of named `const void*`/`const float*` fields with one shared `wdt`. The header
   carries a shouted warning (`k3.h:305-310`) that these **must be zero-initialised**
   before filling, because `K3MoeW` holds a function pointer and several NULL-ness
   tests select a code path — an uninitialised stack struct jumps to garbage.

So: dtype lives in an `int` beside the pointer, shape lives in the call signature,
strides are implicit (everything is dense row-major), and residency is expressed by
whether the pointer is non-NULL.

**Assessment.** Rewrite required. The brief asks for a tensor that can describe
dtype / shape / strides / byte size / file offset / storage backend / resident state in
one object, so a model can say "I need tensor X" and let storage decide where it comes
from. Nothing here can be extended into that; it must be built and the existing code
migrated onto it. The mitigating factor is that the *conventions* are uniform (dense,
row-major, no bias anywhere in the model), so the migration is mechanical.

---

## 6. Current quantization implementation

Two quantized formats, both K3-driven, both hand-rolled with no abstraction between
them.

### MXFP4 (OCP MX FP4) — routed experts

```
value = E2M1[nibble] · 2^(E8M0_scale − 127)
```

- E2M1: 1 sign, 2 exponent, 1 mantissa → exactly 16 values
  (0, 0.5, 1, 1.5, 2, 3, 4, 6 and negatives).
- E8M0: bare biased exponent; **255 is NaN by spec and is mapped to 0** so one bad byte
  cannot poison a row.
- Group size 32, named as `K3_MXFP4_GROUP` rather than spelled at call sites.
- 0.5 bytes/element + 1/32 bytes/element scale = **0.53125 bytes per weight**, which is
  why one 33,030,144-parameter expert is exactly 17,547,264 bytes.
- **Nibble order is a convention, not a rule**: low nibble = even element. Reversing it
  yields the right values in the wrong places — every statistic looks perfect and the
  model is wrong. There is a fixture (`tests/fixtures/mxfp4.json`) that records the
  swapped result specifically to catch this.

`k3_matmul_mxfp4` (`k3_ops.c:1243`) consumes packed nibbles **directly and never
materialises fp32**. Two precomputed tables make it fast: `K3_E2M1_PAIR[256][2]` maps a
whole byte to its two values (one 8-byte load instead of mask+shift+two lookups), and
`K3_E8M0[256]` maps a scale byte to its power of two. The loop is structured around the
32-element group because the scale is constant within one and factors out of the inner
sum.

It carries an explicit **accuracy contract**: it is deliberately *not* bit-identical to
dequantise-then-matmul, because it sums each group of 32 and applies that group's scale
before accumulating. Every individual product is exact in double (E2M1 carries 3
mantissa bits, x carries 24, so 27 of 53 bits are needed), so only the additions round;
reassociating exact terms moves the result ~1 ULP of double, order 1e-16 relative. The
required agreement is 1e-6, gated by `tests/unit/test_expert.c`. Nine orders of margin.

`k3_mxfp4_dequant()` also exists, used only by tests and the resident fixture path.

### Per-row int8 (`K3_WI8` / `K3_DT_I8R`) — draft model only

Each row is stored inline as `[f32 scale][int8 × in]`, so a matrix stays one tagged
pointer. Written by `tools/int8_trunk.py`. Explicitly carries **no exactness
contract** — it is only ever used by the hybrid draft model whose output is verified by
the exact model, which is what licenses a fast non-deterministic kernel (float
accumulation, fused products, natural AVX2 reduction).

### The negative result worth keeping

`k3_trunk.h:5-15` records a measurement, not an opinion: on 31 real attention tensors,
post-hoc int4 costs **17.4% mean relative weight reconstruction error against 0.96% for
int8** — an ~18× gap, consistent across every tensor sampled. K3's tech report §4.1.4
says experts are MXFP4 with quantisation-aware training while non-expert components
"remain in higher precision". That list *is* the trunk. So 4-bit on the trunk is ruled
out by measurement, and streaming (which costs zero error, the bytes are the
checkpoint's own) was chosen instead.

**Assessment.** The MXFP4 kernel is excellent and directly reusable — MXFP4 is one of
the formats the brief wants supported. But there is no `QuantType` abstraction: format
knowledge is spread across `k3_ops.c` (kernels + tables), `k3_st.h` (the dtype enum),
`k3_load.c` (geometry), and `k3_bind.c` (storage-class decisions). Qwen3 GGUF needs
Q4_K/Q6_K/Q8_0, which share *nothing* with MXFP4's layout beyond "blocked with scales".
A generic quant vtable is required, and MXFP4 becomes its first registered
implementation rather than the only path.

---

## 7. Current CPU kernels

All in `src/core/k3_ops.c` (67.8 KB, 30 functions). Complete inventory:

| kernel | line | notes |
|---|---:|---|
| `k3_rmsnorm` | 91 | accumulates in **double**; eps inside the rsqrt |
| `k3_situ_glu` | 107 | K3's SiTU-GLU, not SwiGLU. `|y| ≤ b1·b2 = 100` |
| `k3_shortconv` | 126 | causal depthwise conv, **SiLU fused**, state updated in place |
| `k3_kda_decay` | 161 | `g = lb·σ(exp(A_log[h])·z)`, A_log **per head** |
| `k3_kda_step` | 179 | the recurrence; order is load-bearing |
| `k3_matmul` | 243 | fp32 weights, **16 double accumulators**, OpenMP `if (out>64)` |
| `k3_mla_cached` | 299 | MLA with optional KV cache |
| `k3_router` | 397 | carries its own inline matmul |
| `k3_attn_res` | 459 | block attention residual aggregation |
| `k3_moe` | 534 | per-token routed + shared expert path |
| `k3_moe_prefill` | 667 | batched: fetches each unique expert once per chunk, 3–4× less prefill I/O, bit-identical per token |
| `k3_kda_layer` | 811 | full KDA layer |
| `k3_decoder_layer(_inc)` | 925/1017 | the layer graph |
| `k3_matmul_bf16` | 1066 | AVX2: 4 × `__m256d` accumulators |
| `k3_matmul_q8` | 1136 | AVX2 fp32, draft only |
| `k3_matmul_mxfp4` | 1243 | AVX2, packed nibbles direct |
| `k3_mxfp4_dequant` | 1331 | tests/fixtures only |

Three properties dominate the design, and all three are consequences of K3's
exactness contract:

1. **Accumulation is in `double`, everywhere on the exact path.** `k3_matmul` uses 16
   `double` accumulators; `k3_matmul_bf16`'s AVX2 path uses four `__m256d`. This is
   ~2× the vector work of fp32 accumulation on AVX2 (4 doubles/vector vs 8 floats).
2. **The AVX2 path is bit-identical to the scalar path, not merely close.** The
   reduction tree in the vector path is lane-for-lane the tree the scalar path's 16
   accumulators use, and `test_ops` asserts it. `k3_matmul_bf16` deliberately uses
   `_mm256_fmadd_pd` where the product is provably exact (bf16 widens exactly; float ×
   float needs 48 mantissa bits, which fits double's 53), while `k3_matmul` uses
   mul-then-add: "the two would agree anyway, but that is a proof about the inputs;
   mul-then-add is a proof about the code."
3. **`-ffp-contract=off` is mandatory.** The build disables automatic FMA contraction
   so results are reproducible across compilers, because the op tests compare against a
   reference at a fixed tolerance and letting the compiler fuse moves results past it.

Threading is OpenMP `parallel for schedule(static)` over output rows, guarded
`if (out > 64)`, plus one `schedule(dynamic, 1)` in the cache's batch prefetch. There
is no thread pool, no affinity control, no `--threads` flag — thread count is whatever
`OMP_NUM_THREADS` says.

SIMD is AVX2 + FMA only, guarded by `#if defined(__AVX2__)` with scalar fallbacks. No
AVX-512, no NEON intrinsics (arm64 builds take the scalar path), no runtime dispatch —
the ISA is chosen at compile time.

**Assessment.** The kernels are correct, well-reasoned and well-tested, but they are
tuned for a contract Qwen3 does not need. Qwen3-8B has no bit-exactness requirement
against a streamed-vs-resident split; fp32 accumulation would roughly double AVX2
throughput. So: keep the double-accumulate kernels as the K3 path *and* as the
reference implementation, and add fp32-accumulate variants selected per backend
through a kernel-capability flag rather than a compile-time switch.

Missing entirely, and needed for Qwen3: **RoPE** (K3's MLA is NoPE and KDA is a
recurrence — there is no rotary embedding anywhere in this codebase), **softmax as a
standalone reusable kernel**, **SiLU/SwiGLU** (K3 uses SiTU-GLU, a different function),
**standard multi-head / grouped-query attention** (K3 has MLA and KDA, neither is GQA),
and a **runtime CPU feature detector**.

---

## 8. K3-specific components

Must move to `models/kimi/` and stay there:

- **KDA** — `k3_kda_layer`, `k3_kda_step`, `k3_kda_decay`, `k3_shortconv`, and the
  `K3KdaW` weight struct. The nine-step order is load-bearing and documented as such.
- **Gated MLA with NoPE** — `k3_mla`, `k3_mla_cached`, `K3MlaW`. The invariant that the
  64 rope dimensions *exist and are cached but never rotated*, and that the softmax
  scale is over the full 192-wide head, not 128.
- **Block Attention Residuals** — `k3_attn_res` and the `block_residual`/`n_blocks`
  machinery threaded through every layer. No other architecture in scope has this.
- **Stable LatentMoE** — `k3_moe`, `k3_moe_prefill`, `k3_router`, `K3MoeW`. Including
  the invariant that routing bias steers *selection only* while combining weights come
  from unbiased sigmoid scores.
- **SiTU-GLU** — `k3_situ_glu`, β₁=4, β₂=25.
- **The 93-layer map** — `k3_is_mla` / `k3_is_kda` / `k3_is_dense` and the one-based
  `full_attn` list, including that layers 91 and 92 are both MLA.
- **`k3_cfg.h`** — reads K3's `config.json` shape, refuses to default a missing field.
- **`k3_bind.c`** — binds the released checkpoint's specific tensor names.
- **`k3_load.c`** — the six-tensor expert run and its geometry.
- **`k3_tok.h`** — tiktoken loader for K3's 163,840-entry vocabulary.
- **`tools/pack_trunk.py`** and the trunk JSON layout.
- The draft-model and speculative-decode paths, which depend on K3's cost structure.

---

## 9. Reusable components

Extract to the generic runtime, in rough order of value:

| component | source | generalization needed |
|---|---|---|
| **Aligned/O_DIRECT reads with payload offsets** | `k3_st.c` | none — already generic |
| **Contiguity verification + coalesced range reads** | `k3_load.c` | key by opaque descriptor instead of `(layer, expert)` |
| **Pin-prefix + streaming ring for cyclic access** | `k3_trunk.c` | drop "layer" naming; key by block id |
| **Fixed-order async prefetch (pthread reader)** | `k3_trunk.c` | none — the pattern transfers directly |
| **LRU cache with pinning, inflight sentinel, stats, histogram, trace** | `k3_cache.c` | opaque key + descriptor payload |
| **Batch prefetch (`getmany`) with concurrent reads** | `k3_cache.c` | none |
| **`K3ExpertSrc` vtable pattern** | `k3.h` | rename/widen — already the right shape |
| **Safetensors reader + hash index** | `k3_st.c` | none — GGUF becomes a sibling |
| **Explicit scratch-sizing discipline** | all `k3_*_scratch` | formalize into an arena allocator |
| **Two-storage-class weight policy** | `k3_bind.c` | becomes a tensor-level property |
| **`k3_matmul_mxfp4`** | `k3_ops.c` | register as one quant implementation |
| **`k3_rmsnorm`** | `k3_ops.c` | already generic; add fp32-accumulate variant |
| **`k3_matmul` / `k3_matmul_bf16` / `k3_matmul_q8`** | `k3_ops.c` | already generic |
| **Fixture-manifest-driven tolerance testing** | `test_ops.c` | none — reuse the pattern for Qwen3 |
| **Offline cache simulation from traces** | `tools/sim_cache.py` | none |

The testing methodology deserves separate mention as reusable technology: fixtures
carry their own tolerance in a manifest (`fp32_abs=1e-5`, `fp32_rel=1e-4`) rather than
hardcoding it, and the stated rule is that *a test that cannot fail is not a test* —
fixtures are made adversarial until a plausible wrong implementation fails them. The
router fixture reorders its top-k on 5 of 6 rows; the SiTU-GLU fixture drives the
activation to its exact analytic cap. This is the standard the Qwen3 validation work
should be held to.

---

## 10. Components that should be rewritten

| component | why |
|---|---|
| **Tensor representation** | does not exist; tagged `void*` + separate `int` shapes cannot express file-backed/not-yet-resident state (§5) |
| **`src/cli/k3_run.c`** | 76 KB monolith mixing arg parsing, presets, memory planning, state serialization, speculative decode, the forward pass and reporting. Must split into `app/cli.c` + `runtime/planner.c` + `runtime/memory.c` + a K3 backend |
| **Quantization dispatch** | `k3_mmw`'s three-way `if` on an `int` tag does not extend to Q4_K/Q5/Q6/Q8/MXFP4; needs a vtable (§6) |
| **Memory budgeting** | independent `--trunk-gb`/`--cache-gb` with no single budget, no partitioning, no exhaustion guard (§2) |
| **Hardware detection** | `/proc/meminfo` only, Linux only, silent 0.0 elsewhere; no CPU feature or core detection at all |
| **Kernel dispatch** | compile-time `#if defined(__AVX2__)` only; needs runtime detection so one binary serves multiple CPUs |
| **Threading** | no `--threads`, no pool, no affinity; OpenMP env var only |
| **KV cache** | inline `float *kvc, *ropec` in `Weights` with MLA-specific layout hardcoded into `forward()`; must become a subsystem with backend-supplied layout |
| **Config reading** | `k3_cfg.h` is K3-shaped; needs a generic metadata layer with per-backend schemas |
| **Build system** | GCC/Clang + POSIX only (see §12) |

---

## 11. Proposed new architecture

Principle: **generic runtime + generic storage + generic kernels + model backends.**
The runtime never names a model; a backend never opens a file.

```
src/
├── app/
│   ├── cli.c              inspect | benchmark | run | serve
│   └── server.c           OpenAI-compatible HTTP (later milestone)
│
├── runtime/
│   ├── runtime.{c,h}      engine lifecycle, orchestration, session state
│   ├── planner.{c,h}      hardware probe → execution plan (--auto)
│   ├── memory.{c,h}       one budget → reserve/activations/KV/scratch/weights
│   ├── scheduler.{c,h}    layer pipeline, prefetch depth, thread pool
│   ├── profile.{c,h}      I/O vs compute vs stall accounting
│   └── hwinfo.{c,h}       RAM, cores, AVX2/FMA/AVX512, storage class
│
├── tensor/
│   ├── tensor.{c,h}       dtype+shape+strides+bytes+offset+backend+residency
│   ├── shape.c            shape/stride algebra
│   └── dtype.{c,h}        dtype registry and conversion
│
├── kernels/
│   ├── kernel.h           dispatch table; runtime ISA selection
│   ├── matmul.c           reference + fp32-accum + double-accum variants
│   ├── matmul_avx2.c
│   ├── rmsnorm.c  rope.c  softmax.c  silu.c  swiglu.c  attention.c
│   └── quant/
│       ├── quant.h        QuantType vtable: dequant, dot, block geometry
│       ├── q4.c  q6.c  q8.c        GGUF K-quants
│       └── mxfp4.c        lifted from k3_ops.c, unchanged numerics
│
├── storage/
│   ├── storage.h          Storage vtable: read, read_aligned, map, stat
│   ├── file.c  mmap.c     backends
│   ├── streamer.c         pin-prefix + ring, from k3_trunk.c
│   ├── prefetch.c         async reader thread, from k3_trunk.c
│   ├── cache.{c,h}        generic keyed LRU: pinning, inflight, stats, trace
│   └── pack.c             optional engine-native repacking
│
├── formats/
│   ├── format.h           Format vtable: open, iterate tensors, find, close
│   ├── safetensors.c      from k3_st.c
│   └── gguf.c             new — Qwen3
│
├── models/
│   ├── model.h            ModelBackend vtable
│   ├── registry.c         arch string → backend
│   ├── qwen3/             qwen3.c, qwen3_config.c
│   ├── kimi/              kda.c, mla.c, attn_res.c, latent_moe.c, situ_glu.c,
│   │                      kimi_bind.c, kimi_cfg.c, kimi_pack.c
│   └── gemma/             (stub; proves the interface)
│
├── tokenizer/
│   ├── tokenizer.h        vtable
│   ├── tiktoken.c         from k3_tok.h (Kimi)
│   └── bpe_gguf.c         new — Qwen3 vocab embedded in GGUF
│
└── kv/
    └── kvcache.{c,h}      generic; backend supplies layout descriptor
```

### The two interfaces that carry the design

**`ModelBackend`** — everything architecture-specific:

```c
typedef struct ModelBackend {
    const char *arch;                       /* "qwen3", "kimi-k3", "gemma" */
    int   (*probe)(const ModelMeta *);      /* claim this file? */
    int   (*load)(ModelCtx *, Runtime *, const ModelMeta *);
    int   (*inspect)(const ModelCtx *, ModelInfo *out);
    /* Weight plan: what this layer needs, so the runtime can prefetch it
       without knowing what any of it means. */
    int   (*layer_tensors)(const ModelCtx *, int layer, TensorReq *out, int max);
    /* Execution. Kernels and scratch come from the runtime. */
    int   (*forward_layer)(ModelCtx *, int layer, Activations *, KVCache *,
                           Scratch *, const Batch *);
    int   (*embed)(ModelCtx *, const int *ids, int n, float *out);
    int   (*logits)(ModelCtx *, const float *h, float *out);
    KVLayout (*kv_layout)(const ModelCtx *);
    uint32_t (*capabilities)(const ModelCtx *);   /* MOE | NEEDS_EXACT_FP | ... */
    void  (*destroy)(ModelCtx *);
} ModelBackend;
```

**`Tensor`** — resident and file-backed in one type:

```c
typedef struct Tensor {
    const char *name;
    DType    dtype;          /* F32 BF16 F16 Q4_K Q6_K Q8_0 MXFP4 I8R … */
    int      ndim;
    int64_t  shape[4];
    int64_t  stride[4];      /* elements; 0 = dense row-major */
    int64_t  nbytes;
    Storage *backend;        /* NULL = pure RAM */
    int64_t  file_off;
    void    *data;           /* NULL until materialised */
    uint32_t flags;          /* RESIDENT | MAPPED | CACHED | PINNED | QUANTIZED */
    QuantInfo quant;         /* block size, scale layout — NULL for dense */
} Tensor;
```

The runtime resolves `tensor_data(rt, t)` from RAM → cache → mmap → disk. The backend
never learns which.

### Migration sequence

Each step compiles and keeps `make test` green:

```
M0  baseline: build upstream, record test + bench + generation outputs   ← blocked, §12
M1  tensor + dtype + storage vtable; safetensors behind Format iface
M2  generic keyed cache (from k3_cache.c); K3 uses it; tests still green
M3  generic streamer + prefetch (from k3_trunk.c); K3 uses it
M4  memory manager + hwinfo + planner; --memory / --threads / --auto
M5  kernel dispatch table + runtime ISA detection; K3 kernels registered
M6  quant vtable; MXFP4 registered; K3 unchanged numerically
M7  ModelBackend iface; K3 becomes models/kimi/; CLI split out
M8  GGUF loader + Qwen3 tokenizer
M9  Qwen3 kernels: RoPE, softmax, SwiGLU, GQA — each validated standalone
M10 Q4_K/Q6_K/Q8_0 dequant + fused quantized matmul
M11 Qwen3 backend; layer-by-layer numerical validation vs reference
M12 streaming/memory-budget tests; AVX2 tuning; fusion; profiling
M13 serve
```

K3 regression tests run at every step. That is the definition of "not destroying it".

---

## 12. Dependency graph of major modules

Current (`→` = includes/calls):

```
             k3_run.c ──────────────────────────────────┐
                │                                       │
     ┌──────────┼──────────┬─────────────┬──────────┐   │
     ▼          ▼          ▼             ▼          ▼   ▼
 k3_cfg.h   k3_bind.c  k3_trunk.c   k3_cache.c  k3_tok.h  k3_ops.c
     │          │          │             │                  ▲
     │          │          │             ▼                  │
     │          │          │         k3_load.c ─────────────┤
     │          │          │             │                  │
     │          ▼          ▼             ▼                  │
     │      ──────────  k3_st.c  ────────                   │
     ▼                     │                                │
  json.h                   ▼                          (everything)
                     k3_portable_io.h                     k3.h
```

Observations: `k3.h` is the universal dependency (types + kernel decls + the three
invariants). `k3_ops.c` depends on *nothing* but `k3.h` and libm — the kernels are
already cleanly separable. `k3_run.c` depends on everything and is depended on by
nothing, which is why splitting it is low-risk. The only cycle-free layering violation
is `k3_trunk.h` including `k3_bind.h`, coupling storage to model binding.

Proposed (strict layering, arrows point downward only):

```
                        app/cli.c   app/server.c
                              │           │
                              ▼           ▼
                          runtime/runtime.c
                     ┌────────┼────────┬──────────┐
                     ▼        ▼        ▼          ▼
                 planner  memory   scheduler   profile
                     │                 │
                     ▼                 ▼
                  hwinfo         models/registry.c
                                       │
                     ┌─────────────────┼─────────────────┐
                     ▼                 ▼                 ▼
              models/qwen3/     models/kimi/      models/gemma/
                     └─────────────────┼─────────────────┘
                                       ▼
                    ┌──────────────┬───┴────┬──────────────┐
                    ▼              ▼        ▼              ▼
                kernels/        tensor/    kv/        tokenizer/
                    │              │
                    ▼              ▼
              kernels/quant/    storage/
                                   │
                          ┌────────┴────────┐
                          ▼                 ▼
                      formats/         file.c/mmap.c
```

Rules the layering enforces:

- **Backends may not include `storage/` or `formats/` directly.** They request tensors
  by name through the runtime. This is what stops `if (arch == …)` leaking downward.
- **`kernels/` depends only on `tensor/`.** No storage, no model, no runtime — which is
  what keeps every kernel independently unit-testable, as `k3_ops.c` already is.
- **`storage/` never learns what a layer is.** `k3_trunk.h`'s current include of
  `k3_bind.h` is the one existing violation and it goes away: the streamer keys on
  opaque block ids, and the K3 backend maps layer → block.
- **`runtime/` may not include any `models/<arch>/` header.** Only `models/model.h` and
  `models/registry.c`.

---

## 13. Environment findings — and one blocking decision

This was not in the requested outline but it changes the plan, so it goes in the report.

**Target machine** (probed, not assumed):

| | |
|---|---|
| CPU | Intel Core i5-6300U @ 2.40 GHz (Skylake) |
| Cores / threads | 2 physical / 4 logical |
| ISA | AVX2 + FMA, no AVX-512 |
| RAM | 7.86 GB total, **1.04 GB free at probe time** |
| Free disk (C:) | **24.6 GB** |
| OS | Windows 11 Pro 22000 |

The CPU is exactly the target the brief names (AVX2, FMA, 2P/4L), so kernel and
threading work can be validated for real here. The 1/2/3/4-thread benchmark sweep the
brief asks for is meaningful on this machine.

**Two consequences that constrain scope:**

1. **K3 itself cannot be run here.** The checkpoint is 1.56 TB and the packed trunk is
   109 GB against 24.6 GB free. This is not a defect in the plan — the upstream test
   suite is deliberately built to run *without weights* (`make test` needs no
   checkpoint, ~15 s, ~1.7 GB peak RSS; `scale_test`'s 1.77 GB allocation is the only
   real resource requirement). So the K3 regression baseline is fully obtainable:
   `test_ops`, `test_cache`, `test_st`, `test_cfg`, `scale_test`, `k3_model` (the
   3-gate end-to-end oracle on a tiny model with K3's exact tensor graph), plus
   `bench_kernels`. The checkpoint-dependent tests (`test_expert`, `test_real_layer`,
   `conform_all.py`) cannot run and will be documented as such.

   Qwen3-8B at Q4_K_M is ~4.7–5 GB, which **fits** in 24.6 GB of disk and is designed
   to be run below its file size. The primary goal is unaffected.

2. **There is no C compiler that can build this repository.** Probed:

   | toolchain | status |
   |---|---|
   | MSVC 14.44 (VS 2022 Build Tools) | **present** |
   | CMake 3.x | present |
   | Python (PyManager) | present |
   | gcc / clang / MinGW / MSYS2 | **absent** |
   | WSL2 | enabled, **no distribution installed** |

   The upstream build requires GCC/Clang: `-march=native`, `-mavx2 -mfma`, `-fopenmp`,
   `-std=gnu99`, `-ffp-contract=off`, plus POSIX `pread`, `O_DIRECT`,
   `posix_memalign`, `posix_fadvise`, `sys/resource.h`, and pthreads. Upstream CI
   covers Linux (gcc, clang) and macOS only — Windows is not a supported target
   upstream. MSVC supports none of those flags and lacks the POSIX I/O surface.

**This is the one decision I cannot make unilaterally, because each option is a
material download or a real change in project scope.** Everything up to this point —
audit, report, repo setup, branch — is done and did not depend on it.

| option | cost | consequence |
|---|---|---|
| **A. Install a WSL2 distro** (recommended) | ~500 MB–1 GB download, ~2–3 GB disk | Builds upstream unmodified. Reference platform, matches CI. O_DIRECT and `/proc/meminfo` work as designed. Best baseline fidelity. |
| **B. Install MinGW-w64 / LLVM for Windows** | ~100–300 MB download | Builds natively, but `O_DIRECT`, `posix_fadvise`, `getrusage` and `/proc/meminfo` still need shims. Baseline would differ from upstream in the I/O paths — exactly the paths being generalized. |
| **C. Port to MSVC with a platform layer** | no download; days of work | Writes a Win32 I/O shim (`ReadFile` with `OVERLAPPED`, `FILE_FLAG_NO_BUFFERING`, `GlobalMemoryStatusEx`) before any of the actual project work. Defers the real deliverable and forks from upstream immediately. |
| **D. Proceed without a baseline** | none | Violates §3 of the brief. Not recommended, and I will not do this silently. |

My recommendation is **A**, with the new engine's own build system targeting all three
of Linux, macOS and Windows from the start (CMake with a `platform/` I/O abstraction),
so option C's work happens *inside* the new architecture where it belongs, rather than
as a prerequisite patch to upstream.

---

## Summary judgement

The repository is a strong starting point and the brief's premise is correct: the
valuable technology here is the storage, streaming, caching and low-memory execution
machinery, and it is largely architecture-independent already. `k3_ops.c` depends on
nothing but `k3.h`. `K3ExpertSrc` is already a working vtable. Scratch sizing is
already externalized. The cache already separates prefetch bytes from hit rate so the
statistics do not lie.

The three real gaps are: **no tensor abstraction**, **no model abstraction**, and **a
76 KB CLI monolith that owns the forward pass**. Those are the rewrites. Everything
else is extraction and re-keying.

One caution worth stating plainly. K3's kernels accumulate in `double` and guarantee
the AVX2 path is *bit-identical* to scalar, because K3's whole value proposition is
that output is byte-identical from 8 GB to 224 GB. Qwen3 has no such requirement and
would run roughly twice as fast on AVX2 with fp32 accumulation. Making the runtime
generic therefore means making *numerical policy* a per-backend capability rather than
a global build flag — otherwise the generic engine either slows Qwen3 down or quietly
breaks K3's central guarantee. This is flagged as a first-class design constraint for
milestone M5, not an optimization detail.
