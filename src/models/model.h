/* SPDX-License-Identifier: Apache-2.0 */
/*
 * model.h - the model backend interface.
 *
 * WHAT THIS HAS TO ABSORB. The interface was designed against BOTH architectures at
 * once, not around K3 with Qwen3 bolted on, because the two disagree on almost
 * everything an interface built from one of them would have assumed:
 *
 *                    Kimi K3                        Qwen3-8B
 *   attention        KDA (a recurrence) + MLA       GQA, 32 Q heads / 8 KV heads
 *                    (a shared latent)
 *   positional       NONE. MLA is NoPE and KDA      RoPE, theta 1e6, halved pairing
 *                    encodes order by recurring
 *   per-head norm    none                           QK-norm on Q and K, after
 *                                                   projection, before RoPE
 *   MLP              MoE: 896 experts, top-16,      dense SwiGLU
 *                    SiTU-GLU
 *   sequence state   carried KDA recurrent state    per-position K and V
 *                    PLUS expanded MLA keys
 *   numerics         bit-identical required         fp32 is fine
 *   norm eps         1e-5                           1e-6
 *   container        safetensors + a packed trunk   GGUF
 *   quantization     MXFP4 experts, bf16 trunk      Q4_K / Q6_K mixture
 *
 * THREE CONSEQUENCES FOR THE DESIGN, each of which rules out an obvious shortcut:
 *
 *   1. THE SEQUENCE STATE IS OPAQUE TO THE RUNTIME. The obvious KV cache -- an array
 *      indexed [layer][position][head][dim] -- fits Qwen3 and cannot express K3 at all.
 *      KDA carries a recurrent state that is not per-position, and MLA caches EXPANDED
 *      keys rather than the latent. So the backend allocates and owns its state, and the
 *      runtime only asks how large it is and how much one more position costs. A generic
 *      KV cache that both could use does not exist, and pretending otherwise would put
 *      an `if (arch == kimi)` inside the runtime -- exactly what this refactor removes.
 *
 *   2. POSITIONAL ENCODING IS NOT ASSUMED TO EXIST. K3 has no rotary embedding
 *      anywhere. An interface with a `rope_theta` field would be describing Qwen3 and
 *      lying about K3. It is a capability the backend declares and applies internally.
 *
 *   3. NUMERICAL POLICY TRAVELS WITH THE BACKEND. See kernels/kernel.h: K3 must be
 *      bit-identical across memory configurations and Qwen3 need not be. The backend
 *      declares which it needs and the kernels honour it per call.
 *
 * WHAT THE RUNTIME PROVIDES, and what a backend must therefore NOT reimplement:
 * memory budgeting, storage and streaming, the weight cache, tensor descriptors,
 * quantized dot products, CPU kernels, scheduling, profiling. A backend that opens its
 * own file or allocates its own arena has misunderstood the split.
 *
 * WHAT A BACKEND OWNS: its config, its tensor names, its execution order, its
 * attention, its MLP, its routing, its positional encoding, its state layout.
 */
#ifndef ENG_MODEL_H
#define ENG_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "kernel.h"
#include "planner.h"
#include "storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngModel EngModel;          /* a loaded model; backend-defined */
typedef struct EngSeqState EngSeqState;    /* per-sequence state; backend-defined */

enum {
    /* The backend can execute, not merely describe. A backend may be registered
     * without this while its execution path is still being migrated: `inspect` then
     * works and `run` refuses with a clear message instead of crashing. Declaring it
     * is how the CLI knows which of those to do. */
    ENG_MCAP_EXECUTE      = 1u << 0,
    ENG_MCAP_MOE          = 1u << 1,   /* has routed experts */
    ENG_MCAP_POSITIONAL   = 1u << 2,   /* applies a positional encoding */
    ENG_MCAP_RECURRENT    = 1u << 3,   /* carries state that is not per-position */
    ENG_MCAP_STREAMABLE   = 1u << 4,   /* weights can be streamed layer by layer */
    ENG_MCAP_INCREMENTAL  = 1u << 5    /* supports one-token-at-a-time decoding */
};

typedef struct {
    uint32_t     flags;
    EngNumPolicy num_policy;   /* what the kernels must honour for this architecture */
    const char  *notes;        /* one line, shown by `inspect` */
} EngModelCaps;

enum {
    /* Produce logits for this position. Omit it while consuming a prompt: only the
     * final position's distribution is ever read. The sequence state is updated either
     * way, so the KV cache is unaffected. */
    ENG_DEC_LOGITS = 1u << 0
};

/* Run counters, in terms every architecture shares.
 *
 * WALL SECONDS THROUGHOUT. clock() sums across threads under OpenMP, so on four threads
 * it reports about four times the elapsed time; mixing it with a monotonic read timer
 * makes compute look several times larger than it is, which this engine's first
 * instrumentation did.
 *
 * `io_stall_s` is what the decode WAITED for, which is not the same as the time the
 * device spent transferring: a prefetched read costs bytes but no stall. The gap
 * between io_stall_s and bytes_read/rate is what overlapping bought. */
typedef struct {
    double  total_s;
    double  io_stall_s;      /* blocked waiting for weights                      */
    double  compute_s;       /* total_s - io_stall_s                             */
    int64_t bytes_read;      /* weight bytes moved, however they were paid for   */
    double  device_s;        /* time inside the read loop: a device rate         */
    int     steps;
    int     logit_steps;     /* steps that produced a distribution               */
    char    notes[160];      /* one line the backend wants surfaced              */
} EngRunStats;

/* What a backend needs in order to load. The runtime has already chosen the plan; the
 * backend does not get to argue with it, only to fail if it cannot comply. */
typedef struct {
    const char     *path;       /* container: a GGUF file, or a shard directory */
    EngStorage     *store;      /* opened by the runtime; may be NULL for multi-file */
    const EngPlan  *plan;       /* budgets, threads, context, streaming decision */
    int             verbose;
} EngLoadReq;

typedef struct EngModelBackend {
    const char *name;           /* "kimi-k3", "qwen3" */
    const char *description;

    /* Does this backend claim this container? Reads METADATA ONLY -- no weights, no
     * allocation -- because probing runs over every registered backend in turn and must
     * be cheap and side-effect free. Returns a confidence score: 0 = not mine, higher
     * wins. A score rather than a boolean because containers overlap (several
     * architectures ship as GGUF) and the most specific claim should win. */
    int (*probe)(const char *path);

    /* Describe the model for the planner, WITHOUT loading weights. This is what
     * `engine inspect` prints and what sizes the memory plan, so it must be answerable
     * from the container's metadata alone. */
    int (*inspect)(const char *path, EngModelFacts *out);

    /* Load. Returns NULL on failure, having said why on stderr. */
    EngModel *(*load)(const EngLoadReq *req);
    void      (*destroy)(EngModel *m);

    /* Allocate per-sequence state for `context` positions. Backend-defined layout: see
     * consequence 1 above. Returns NULL on failure. */
    EngSeqState *(*state_create)(EngModel *m, int context);
    void         (*state_destroy)(EngSeqState *s);

    /* Bytes of sequence state for `context` positions, WITHOUT allocating. The planner
     * needs this before deciding a context length, so it cannot be answered by
     * allocating and measuring. */
    int64_t (*state_bytes)(const EngModel *m, int context);

    /* One incremental decode step: consume `token` at `pos`, update state, and produce
     * logits if asked. Returns 0 on success.
     *
     * `pos` is passed explicitly rather than tracked inside the state because the
     * caller may rewind (speculative decoding, beam search) and a hidden counter would
     * make that a backend-by-backend correctness question.
     *
     * `flags` carries ENG_DEC_LOGITS. It is a PER-CALL property, not model state,
     * because the same model alternates between positions whose logits are read and
     * positions whose are not -- every prompt token but the last falls in the second
     * group, and computing them there costs a full vocabulary-wide matmul plus, for a
     * streamed LM head, a re-read of it. On Qwen3 that is 510 MB and 151,936 rows
     * thrown away per prompt token. */
    int (*decode)(EngModel *m, EngSeqState *s, int token, int pos, uint32_t flags);

    /* Logits from the last decode. Borrowed, valid until the next decode. */
    const float *(*logits)(const EngModel *m, int *n_vocab);

    /* OPTIONAL, may be NULL. Run counters in terms every architecture shares, so an
     * application can report where the time went without knowing which backend it has.
     *
     * The alternative -- the CLI calling qwen3_profile() directly -- would put an
     * architecture conditional in the application, which is the thing this interface
     * exists to prevent. A backend with nothing to report leaves it NULL and the caller
     * prints nothing. */
    void (*stats)(const EngModel *m, EngRunStats *out);

    void (*caps)(EngModelCaps *out);
} EngModelBackend;

/* --------------------------------------------------------------- the registry -- */

/* Returns 0, or -1 if the name is taken or the backend is missing a required entry
 * point. A backend without probe/inspect/caps is refused outright: those three are what
 * make it usable at all, and a half-registered backend that fails later is worse than
 * one that fails at startup. */
int eng_model_register(const EngModelBackend *b);

const EngModelBackend *eng_model_find(const char *name);

/* Ask every registered backend to probe `path` and return the highest scorer, or NULL.
 * *score_out receives the winning score when non-NULL. */
const EngModelBackend *eng_model_probe(const char *path, int *score_out);

int                    eng_model_count(void);
const EngModelBackend *eng_model_at(int i);

/* Register the backends compiled into this binary. Idempotent; called automatically by
 * the lookup functions, so an application does not have to remember to. */
void eng_model_register_builtins(void);

void eng_model_caps_string(const EngModelCaps *c, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* ENG_MODEL_H */
