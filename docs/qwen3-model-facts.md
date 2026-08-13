# Qwen3-8B — measured model facts

Read directly from the target file, not from documentation. Everything here comes from
the GGUF header of:

```
C:\Users\SHAMI\HAKIM\AI\Qwen3-8B-Q4_K_M.gguf
4,682 MB (5,027,783,488 bytes)   GGUF version 3
general.name = "Qwen3 8B Awq Compatible Instruct"
```

This is the source of truth for the Qwen3 backend. Where it disagrees with the
published Qwen3 config, **this file wins** — it is what the engine has to run.

## Architecture

| GGUF key | value | meaning |
|---|---:|---|
| `general.architecture` | `qwen3` | backend selector |
| `qwen3.block_count` | 36 | decoder layers |
| `qwen3.embedding_length` | 4096 | hidden width |
| `qwen3.feed_forward_length` | 12288 | SwiGLU intermediate |
| `qwen3.attention.head_count` | 32 | query heads |
| `qwen3.attention.head_count_kv` | 8 | KV heads — **GQA, 4:1** |
| `qwen3.attention.key_length` | 128 | head_dim |
| `qwen3.attention.value_length` | 128 | head_dim |
| `qwen3.attention.layer_norm_rms_epsilon` | 1e-6 | note: **not** K3's 1e-5 |
| `qwen3.rope.freq_base` | 1,000,000.0 | θ base — not 10,000 |
| `qwen3.context_length` | 40,960 | max positions |
| vocab | 151,936 | from `tokenizer.ggml.tokens` |

Derived: `head_count × key_length = 32 × 128 = 4096 = embedding_length`, so Q projects
square. KV projects to `8 × 128 = 1024`, a 4× reduction — this is what makes the KV
cache cheap relative to K3's MLA.

## Three things that differ from K3 and will not be shared code

1. **QK-norm.** Every layer carries `attn_q_norm.weight` and `attn_k_norm.weight`, both
   `F32[128]` — i.e. RMSNorm applied **per head over head_dim**, to Q and K, *after*
   projection and *before* RoPE. This is a Qwen3 signature. K3 has nothing equivalent.
   Omitting it produces a model that runs and is wrong.
2. **RoPE with θ base 1e6.** K3 has no rotary embedding anywhere — MLA is NoPE and KDA
   is a recurrence. RoPE is new code, not a port.
3. **Untied embeddings.** `token_embd.weight` (Q4_K) and `output.weight` (Q6_K) are
   **separate tensors with different quantization**. Do not alias them.

## Weight inventory

399 tensors, 5,021.83 MB of tensor data.

| dtype | tensors | MB | what |
|---|---:|---:|---|
| Q4_K | 217 | 3,704.98 | token_embd, attn_q, attn_k, attn_output, ffn_gate, ffn_up |
| Q6_K | 37 | 1,315.61 | output (lm_head), attn_v, ffn_down |
| F32 | 145 | 1.23 | all norms: attn_norm, ffn_norm, attn_q_norm, attn_k_norm, output_norm |

The Q4_K/Q6_K split is the standard `Q4_K_M` mixture: the tensors most sensitive to
quantization error (V projection, the FFN down-projection, and the LM head) get 6 bits;
everything else gets 4.

Note that **all norms are already F32** — the same two-storage-class split K3 arrived at
by hand (`k3_bind.h`: big matrices stay narrow, elementwise-read vectors go wide) is
what GGUF does by default. That policy carries over unchanged.

## Per-layer tensors (11 per block, uniform across all 36)

```
blk.L.attn_norm.weight     F32  [4096]           pre-attention RMSNorm gain
blk.L.attn_q.weight        Q4_K [4096, 4096]     Q projection
blk.L.attn_k.weight        Q4_K [4096, 1024]     K projection  (GQA)
blk.L.attn_v.weight        Q6_K [4096, 1024]     V projection  (GQA)
blk.L.attn_q_norm.weight   F32  [128]            QK-norm on Q, per head
blk.L.attn_k_norm.weight   F32  [128]            QK-norm on K, per head
blk.L.attn_output.weight   Q4_K [4096, 4096]     O projection
blk.L.ffn_norm.weight      F32  [4096]           pre-FFN RMSNorm gain
blk.L.ffn_gate.weight      Q4_K [4096, 12288]    SwiGLU gate
blk.L.ffn_up.weight        Q4_K [4096, 12288]    SwiGLU up
blk.L.ffn_down.weight      Q6_K [12288, 4096]    SwiGLU down
```

Model-level: `token_embd.weight` Q4_K `[4096, 151936]`, `output_norm.weight` F32
`[4096]`, `output.weight` Q6_K `[4096, 151936]`.

## Storage layout — already good for streaming

- Tensor data begins at byte **5,956,416** (`general.alignment = 32`).
- **Tensor offsets are monotonically non-decreasing in table order.** Verified across
  all 399 entries.
- Table order is `output`, `output_norm`, `token_embd`, then `blk.0.*` … `blk.35.*`,
  with each block's tensors adjacent.

So a layer's weights are **already one contiguous run**, which is exactly the property
`k3_trunk.c` relies on and `tools/pack_trunk.py` had to *create* for K3. For Qwen3 the
packer is therefore optional — the GGUF is already close to the engine-optimal layout.
That is worth knowing before building a repacking step nobody needs: the repacker
earns its place only if measurement shows the residual gap matters.

Per-layer resident cost:
```
Q4_K:  attn_q 4096×4096 + attn_output 4096×4096 + ffn_gate/up 2×4096×12288
Q6_K:  attn_v 4096×1024 + ffn_down 12288×4096
       ≈ 130.4 MB per layer × 36 layers ≈ 4.70 GB
plus   token_embd 350 MB (Q4_K) + output 510 MB (Q6_K)
```

A 5 GB model on a machine with ~1 GB free RAM is precisely the streaming case, so the
memory-budget work is exercised for real here rather than hypothetically.

## Tokenizer

| key | value |
|---|---|
| `tokenizer.ggml.model` | `gpt2` (byte-level BPE) |
| `tokenizer.ggml.pre` | `qwen2` (pre-tokenizer regex variant) |
| tokens | 151,936 |
| merges | 151,387 |
| BOS / EOS / PAD | 151643 / 151645 / 151643 |
| `add_bos_token` | **false** |
| chat template | present (Jinja, ChatML-style `<\|im_start\|>`) |

The vocabulary and merges are **embedded in the GGUF**, so unlike K3 (whose tokenizer
test cannot run without downloading `tiktoken.model` separately) the Qwen3 tokenizer is
fully testable from the file already on disk. Byte-level BPE with the `qwen2`
pre-tokenizer regex is a different implementation from K3's tiktoken loader; it goes
behind the tokenizer vtable as a sibling, not a modification.

## Quantization formats to implement

Both are k-quants with 256-element superblocks:

- **Q4_K** — 144 bytes per 256 elements (4.5 bits/weight): 8 sub-blocks of 32, 6-bit
  scales and mins, two F16 super-scales.
- **Q6_K** — 210 bytes per 256 elements (6.5625 bits/weight): 4-bit low + 2-bit high
  nibbles, 8-bit sub-block scales, one F16 super-scale.

Neither resembles K3's MXFP4 (flat 32-element groups, E2M1 values, E8M0 exponent
scales) beyond both being "blocked with scales". This is the concrete evidence for the
report's §6 conclusion that a `QuantType` vtable is required rather than extending
`k3_mmw`'s three-way `int` tag.

Exact block layouts must be taken from the `ggml` reference and validated against real
bytes from this file before any kernel is optimized — same discipline as K3's
`fixtures/mxfp4.json`, which records the *wrong* nibble order specifically so that
getting it backwards fails a test instead of silently producing a plausible model.
