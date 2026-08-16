/* SPDX-License-Identifier: Apache-2.0 */
/*
 * qwen3.h - the Qwen3 backend.
 *
 * FIVE THINGS THAT ARE QWEN3'S AND NOT SHARED WITH K3. Each is a place where a wrong
 * choice produces a model that runs, sounds fluent, and is wrong:
 *
 *   1. QK-NORM. Every layer carries attn_q_norm and attn_k_norm, F32[head_dim],
 *      applied as RMSNorm PER HEAD over head_dim, AFTER the projection and BEFORE RoPE.
 *      K3 has nothing equivalent. Omitting it changes every attention score.
 *
 *   2. ROPE AT THETA 1e6, HALVED PAIRING. K3 has no rotary embedding at all -- MLA is
 *      NoPE and KDA encodes order by recurring. Both the base (1e6, not the common 1e4)
 *      and the pairing (i with i+dim/2, not 2i with 2i+1) are load-bearing, and both
 *      wrong choices preserve every invariant a smoke test would check.
 *
 *   3. GROUPED-QUERY ATTENTION, 32 query heads over 8 KV heads. Query head h reads KV
 *      head h / 4. Getting the divisor backwards (h % 8) still indexes in range and
 *      still attends to something.
 *
 *   4. SWIGLU, not K3's SiTU-GLU. The two are different activations; K3's caps both
 *      halves with tanh and has a different gate.
 *
 *   5. UNTIED EMBEDDINGS. token_embd.weight (Q4_K) and output.weight (Q6_K) are
 *      separate tensors at different quantisations. Aliasing them, which several
 *      architectures allow, silently substitutes one for the other.
 *
 * WHERE THE WEIGHTS LIVE. Qwen3's GGUF is already laid out so a layer is ONE contiguous
 * run -- verified, not assumed, by gguf_layout_is_sequential -- so the generic block
 * streamer serves it directly and the repacker K3 needed is unnecessary here.
 *
 *   layers        streamed through EngStreamer, pinned prefix plus ring
 *   token_embd    read one row per token, on demand. 350 MB resident to serve 4 KB
 *                 per step is a poor trade at this budget.
 *   output        the LM head, 510 MB. Read in row chunks per step. This is the
 *                 single most expensive part of a decode and the obvious thing to
 *                 optimise later; it is left simple because correctness comes first.
 *   norms         inside their layer's run, so they arrive with it.
 *
 * NUMERICAL POLICY IS ENG_NUM_FAST. Qwen3 makes no claim that its output is independent
 * of the memory configuration, so fp32 accumulation is appropriate and the exact path's
 * halved vector width is not worth paying for. K3 declares ENG_NUM_EXACT for the
 * opposite reason. See kernels/kernel.h.
 */
#ifndef ENG_QWEN3_H
#define ENG_QWEN3_H

#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

extern const EngModelBackend eng_backend_qwen3;

/* Debug capture. When a callback is installed, every named intermediate the forward
 * pass produces is handed to it: "embedding", "l0.q_proj", "layer7", "final_norm",
 * "logits" and so on, matching the names tools/qwen3_ref.py emits.
 *
 * This exists because comparing only the logits tells you that something is wrong
 * across 36 layers; comparing per layer tells you it started at layer 7. Costs nothing
 * when the callback is NULL, which is the production path. */
typedef void (*Qwen3Capture)(void *ctx, const char *name, const float *v, int n);
void qwen3_set_capture(EngModel *m, Qwen3Capture cb, void *ctx);

/* The layer streamer, for read statistics. Exposed so a caller can separate device time
 * from compute time without the backend having to grow its own profiling vocabulary. */
struct EngStreamer *qwen3_streamer(EngModel *m);

#ifdef __cplusplus
}
#endif

#endif /* ENG_QWEN3_H */
