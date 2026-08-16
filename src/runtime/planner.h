/* SPDX-License-Identifier: Apache-2.0 */
/*
 * planner.h - turn a machine and a model into an execution plan.
 *
 * The user should be able to type `engine run model.gguf --auto` and get a sensible
 * configuration without knowing what a pinned prefix is. Everything the planner decides
 * is overridable; what it must never do is choose something that cannot run.
 *
 * EVERY POLICY HERE IS BACKED BY A MEASUREMENT, and the ones that are not are marked as
 * assumptions. Two matter most:
 *
 *   THREADS ARE NOT nproc. On the reference machine (2 physical / 4 logical), 7
 *   interleaved reps quoting the minimum:
 *
 *       threads   bf16 matmul        MXFP4 matmul
 *          1      26.12 ms  1.00x     9.05 ms  1.00x
 *          2      15.30 ms  1.71x     4.69 ms  1.93x
 *          3      13.88 ms  1.88x     3.59 ms  2.52x
 *          4      14.09 ms  1.85x     3.18 ms  2.85x
 *
 *   bf16 saturates at 2-3 and REGRESSES at 4; MXFP4 keeps scaling to 4. The difference
 *   is arithmetic intensity: a bf16 matvec moves 2 bytes per multiply-add and is
 *   bandwidth-bound, so hyperthreads contend for the same memory port rather than
 *   adding throughput, while a 4-bit weight moves ~0.5 bytes and leaves the threads
 *   something to hide. So the planner picks threads from the model's BYTES PER WEIGHT,
 *   not from the core count alone.
 *
 *   STREAM BEFORE CACHE. Weights re-read in a fixed order every token are worth more
 *   memory than a data-dependent cache, because pinning one gives a deterministic
 *   hit-rate gain while the cache's gain depends on reuse the model may not have. K3
 *   measured trunk-first as 1.69x faster than cache-first at a fixed 128 GB budget.
 *   The planner therefore fills the streamer first and gives the remainder to the
 *   cache, which is also why a dense model (no expert cache at all) is the simple case.
 */
#ifndef ENG_PLANNER_H
#define ENG_PLANNER_H

#include <stddef.h>
#include <stdint.h>

#include "hwinfo.h"
#include "memory.h"

#ifdef __cplusplus
extern "C" {
#endif

/* What the planner needs to know about a model. A backend fills this from the container
 * without loading any weights, which is also what `engine inspect` prints. */
typedef struct {
    const char *arch;
    int      n_layers;

    int64_t  total_weight_bytes;   /* everything on disk */
    int64_t  resident_bytes;       /* embeddings, LM head: cannot stream */
    int64_t  max_layer_bytes;      /* largest per-layer run; sizes a ring slot */
    int64_t  avg_layer_bytes;

    int64_t  activation_bytes;     /* hidden states etc. for one step */
    int64_t  scratch_bytes;        /* kernel working buffers */
    int64_t  kv_bytes_per_pos;

    int      context_max;          /* what the model itself supports */

    /* MoE. n_experts == 0 for a dense model, which turns the expert cache off. */
    int      n_experts;
    int      topk;
    int64_t  expert_bytes;         /* one expert, as stored */

    /* Mean bytes per stored weight across the dominant matmuls: 2.0 for bf16, ~0.55 for
     * 4-bit with scales. Drives the thread choice. Zero means unknown, and the planner
     * then assumes bandwidth-bound, which is the conservative direction. */
    double   bytes_per_weight;
} EngModelFacts;

/* What an unrequested context defaults to. Deliberately modest: see the reasoning in
 * eng_plan. Long contexts are available by asking for them, at a cost the planner then
 * states rather than absorbs silently. */
#define ENG_DEFAULT_CONTEXT 4096

typedef struct {
    int64_t memory_budget;
    int     threads;
    int     context;

    int     streaming;        /* 0 when every weight fits resident */
    int64_t stream_budget;    /* for the layer streamer */
    int64_t cache_budget;     /* for the expert cache; 0 on a dense model */
    int     prefetch_depth;

    EngMemPlan mem;

    /* Predicted, before anything runs. The streamer's hit rate is deterministic, so
     * this is arithmetic rather than a guess. -1 when not applicable. */
    double  predicted_stream_hit_rate;

    /* Human-readable reasoning, printed by `inspect` and by `run --auto`. The planner
     * explaining itself is the difference between a tool a user can trust and one they
     * have to reverse-engineer. */
    char    notes[10][192];
    int     n_notes;

    int     ok;
    char    problem[256];
} EngPlan;

/* Overrides. Any field left at its "unset" value is chosen by the planner:
 *   memory_budget  0 = auto
 *   threads        0 = auto
 *   context        0 = model default (capped by memory)
 *   force_stream  -1 = auto, 0 = require resident, 1 = force streaming */
typedef struct {
    int64_t memory_budget;
    int     threads;
    int     context;
    int     force_stream;
} EngPlanRequest;

void eng_plan_request_init(EngPlanRequest *r);

/* Produce a plan. Returns 0 when viable. Never allocates and never opens anything: it
 * is arithmetic over facts already gathered, so `inspect` can run it on a machine that
 * could not actually host the model. */
int eng_plan(EngPlan *plan, const EngHwInfo *hw, const EngModelFacts *m,
             const EngPlanRequest *req);

void eng_plan_report(const EngPlan *plan, const EngModelFacts *m, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* ENG_PLANNER_H */
