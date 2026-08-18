# Adding a model backend

What it takes to add an architecture, written from what adding Qwen3 actually cost rather
than from what the interface looks like on paper.

Everything model-specific lives in one directory. Nothing in `src/runtime/`,
`src/storage/`, `src/kernels/` or `src/quant/` should need to change, and if it does, that
is worth stopping to examine — the runtime carrying an `if (arch == ...)` is the condition
this design exists to prevent.

---

## What you implement

`src/models/model.h` defines a vtable. Three entries are mandatory and the registry
refuses a backend without them:

| | | |
|---|---|---|
| `probe` | required | does this container look like yours? metadata only |
| `inspect` | required | describe the model without loading weights |
| `caps` | required | what you support, and which arithmetic you need |
| `load` / `destroy` | to execute | |
| `state_create` / `state_destroy` / `state_bytes` | to execute | per-sequence state |
| `decode` | to execute | one position |
| `logits` | to execute | the distribution from the last decode |
| `stats` | optional | run counters for `benchmark` |

**You may register without an execution path.** Declare `caps` without
`ENG_MCAP_EXECUTE` and `inspect` works while `run` refuses with a clear message. Kimi K3
sits in exactly that state today. The registry checks the claim against the vtable at
startup, so declaring the capability without implementing it is a startup error rather
than a null-pointer call during generation.

---

## 1. probe

```c
static int qwen3_probe(const char *path)
{
    Gguf g;
    if (gguf_open(&g, path) != 0) return 0;
    const char *arch = NULL; int64_t n = 0;
    int score = 0;
    if (gguf_str(&g, "general.architecture", &arch, &n) == 0 &&
        n == 5 && !memcmp(arch, "qwen3", 5))
        score = 100;
    gguf_close(&g);
    return score;
}
```

Return a **confidence score**, not a boolean, because containers overlap — many
architectures ship as GGUF — and the most specific claim should win. `eng_model_probe`
runs every backend and takes the highest; equal claims are reported rather than resolved
silently.

Probe runs for every registered backend on every path, so it must be cheap and free of
side effects. Read metadata; do not allocate, and do not touch weights.

**Make the signature specific enough to be wrong about.** K3's probe requires *both* a
delta-attention head count and an MLA latent rank, because other linear-attention models
have the first and DeepSeek-family models have the second. A probe that claims too much
will happily load a model it cannot run.

---

## 2. inspect

Fill `EngModelFacts` from the container alone. This is what `engine inspect` prints and
what sizes the memory plan, so it must be answerable without loading weights.

**Measure, do not derive.** The tempting shortcut is arithmetic: experts × matrices ×
width × bits. It is wrong often enough to matter. K3's routed expert is documented at
33,030,144 parameters, which does not match `3 × latent × hidden` for the configured
latent — the expert's inner width is not the field an outside reader would assume.
Deriving it produces a confident, wrong memory plan. Read the sizes from the container's
own tensor table; report **zero** when you genuinely cannot know, which the planner reads
as unknown rather than as none.

**`kv_bytes_per_pos` is for the whole model, not one layer.** Attention runs independently
in every layer, so each keeps its own state per position. Reporting one layer's cost made
the planner accept a context 36× larger than the memory actually required: the plan
printed comfortably and the run was OOM-killed. If your figure does not have a layer count
in it, check why.

`bytes_per_weight` decides the thread count — see [the planner](../src/runtime/planner.h)
for the measured table. Compute it as total bytes over total elements rather than quoting
the nominal bit width, which ignores block padding.

---

## 3. load

The runtime has already chosen the plan. You do not argue with it; you comply or fail.

Weights come through the generic streamer. If your container lays a layer out as one
contiguous run — Qwen3's GGUF does, verified rather than assumed by
`gguf_layout_is_sequential` — then building the block table is mechanical:

```c
for (int L = 0; L < n_layers; L++) {
    layer_span(g, L, &blocks[L].off, &blocks[L].nbytes);
    /* record each tensor's offset RELATIVE to the block start */
}
cfg.budget_bytes = req->plan->stream_budget;
m->stream = eng_streamer_create(&cfg);
```

If it does not, either repack (see `tools/pack_trunk.py`, which is what K3 needs) or
accept the seeks and say so.

**Decide residency against measurement, not intuition.** Holding Qwen3's 511 MB LM head
resident looks obviously right: it is a serial read that cannot overlap compute. It is
2.57× *slower*, because 511 MB of budget is about four pinned layers, and a pinned layer
is read on every step while the LM head is read only on steps producing logits. Provide an
environment override so the A/B can run on one binary at one budget — comparing two runs
that differ in two ways is how that conclusion was got wrong the first time.

---

## 4. Sequence state

**The runtime does not know what your state is, deliberately.** The obvious KV cache —
`[layer][position][head][dim]` — fits Qwen3 and cannot express K3 at all: KDA carries a
recurrent state that is not per-position, and MLA caches expanded keys rather than the
latent. So you allocate it, you own it, and the runtime only asks how large it is.

`state_bytes` must answer **without allocating**, because the planner needs it before it
can choose a context length.

`decode` takes `pos` as an argument rather than tracking it internally, so a caller can
rewind — speculative decoding and beam search need that, and a hidden counter would make
it a per-backend correctness question.

---

## 5. decode

```c
int decode(EngModel *m, EngSeqState *s, int token, int pos, unsigned flags);
```

Honour `ENG_DEC_LOGITS`. Prompt positions do not need a distribution, and computing one
anyway cost 64 seconds per prompt token here — a 151,936-row projection plus, with a
streamed head, a 511 MB read. Update the sequence state either way, so the KV cache and
every later position are unaffected.

Use the generic kernels. `eng_matmul_quant` never materialises a packed weight; a
matrix-vector product is memory-bound, so reading 7.5× fewer bytes is *faster* than
dequantising first, not a trade.

---

## 6. caps, and register

```c
out->flags = ENG_MCAP_EXECUTE | ENG_MCAP_POSITIONAL | ENG_MCAP_STREAMABLE
           | ENG_MCAP_INCREMENTAL;
out->num_policy = ENG_NUM_FAST;
```

`num_policy` is the one to think about. `ENG_NUM_EXACT` requires every implementation of a
kernel to agree **bit for bit**, which is what backs K3's claim that its output does not
depend on the memory budget. It costs half the vector width — an AVX2 register holds 4
doubles against 8 floats. Ask for it only if you are making that claim; Qwen3 is not, and
should not pay for it.

Then add one line to `eng_model_register_builtins()` in `src/models/registry.c`, and your
sources to `MODEL_SRC` in the Makefile. That is the whole integration.

---

## Validating it

**Generating fluent text is not evidence of correctness, and this is the part to take
seriously.** A wrong RoPE base, the adjacent instead of the halved pairing, a missing
QK-norm, or a GQA head mapping done with `%` instead of `/` all yield a model that loads,
runs, and produces confident, plausible, wrong output. None of them change the shape of
anything.

So write an independent reference. `tools/qwen3_ref.py` is a numpy forward pass written
from the architecture, sharing no code and no loop structure with the C. It dumps the
hidden state after **every layer** plus named intermediates inside layer 0, because a
logits-only comparison tells you something is wrong across 36 layers while a per-layer one
tells you it started at layer 7.

What a good result looks like:

```
layer  0   2.670e-06        layer 25   3.105e-06
layer 10   3.335e-06        layer 35   3.888e-06
```

**Flat** drift at fp32-versus-f64 noise. A structural error is order-1 at the first
affected tensor, not a slow climb — so the shape of the curve tells you as much as its
magnitude.

Build the reference in layers you can trust separately. `tools/gguf_ref.py` decodes the
quantization independently and was validated bit-exactly against the C kernels first, so a
disagreement in the forward pass can only be an architecture difference. Debugging two
unvalidated layers at once is much harder than debugging one.

**Include a sabotage case.** A test that only ever passes proves nothing about its own
power. `test_gguf` deliberately swaps every quant byte's nibbles and confirms the
comparison catches it — the sabotaged weights are still finite and in range, which is
exactly the failure being guarded against.

---

## Traps

Every one of these was hit here or upstream, and none changes the shape of the output.

| trap | what it looks like |
|---|---|
| RoPE pairing (`i, i+dim/2` vs `2i, 2i+1`) | both preserve every pair norm; both run |
| RoPE base (1e6 vs the common 1e4) | plausible attention, wrong model |
| GQA mapping `h % n_kv` instead of `h / rep` | indexes in range, attends to the wrong head |
| QK-norm omitted | Qwen3 has it, K3 has nothing like it |
| Quant nibble order | decodes to equally plausible weights |
| Tying embeddings the container stores untied | Qwen3's are different tensors at different quantisations |
| `kv_bytes_per_pos` for one layer | plan looks comfortable, run is OOM-killed |
| Norm epsilon (1e-6 vs 1e-5) | small, systematic, invisible |

---

## Checklist

- [ ] `probe` reads metadata only, and is specific enough to refuse a near-miss
- [ ] `inspect` measures sizes rather than deriving them, and reports 0 for unknown
- [ ] `kv_bytes_per_pos` counts every layer
- [ ] `state_bytes` answers without allocating
- [ ] `decode` honours `ENG_DEC_LOGITS` and takes `pos` explicitly
- [ ] `caps` asks for `ENG_NUM_EXACT` only if you are claiming budget-independence
- [ ] an independent reference exists, and is compared **per layer**
- [ ] the comparison has a sabotage case proving it can fail
- [ ] `scripts/verify.sh` passes: tests, sanitizers, `-Werror`
- [ ] the streaming-equivalence gate passes at several budgets

---

## Worked examples

| | |
|---|---|
| [`src/models/qwen3/`](../src/models/qwen3/) | full execution path |
| [`src/models/kimi/`](../src/models/kimi/) | probe/inspect only, execution not yet migrated |
| [`tests/unit/test_model.c`](../tests/unit/test_model.c) | a synthetic backend exercising every entry point |

The synthetic backend in `test_model.c` is the shortest complete example, and it exists to
prove the interface is not shaped around either real architecture: it is dense where they
are not, positional where K3 is not, and `ENG_NUM_FAST` where K3 is exact.
