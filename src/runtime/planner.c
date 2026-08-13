/* SPDX-License-Identifier: Apache-2.0 */
/* planner.c - see planner.h. */
#include "planner.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

void eng_plan_request_init(EngPlanRequest *r)
{
    if (!r) return;
    memset(r, 0, sizeof *r);
    r->force_stream = -1;
}

#if defined(__GNUC__)
__attribute__((format(printf, 2, 3)))
#endif
static void note(EngPlan *p, const char *fmt, ...);

static void note(EngPlan *p, const char *fmt, ...)
{
    if (p->n_notes >= (int)(sizeof p->notes / sizeof p->notes[0])) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(p->notes[p->n_notes], sizeof p->notes[0], fmt, ap);
    va_end(ap);
    p->n_notes++;
}

/* THREADS. See planner.h for the measured table this implements.
 *
 * The rule is about arithmetic intensity, not core count: a kernel that moves 2 bytes
 * per multiply-add saturates memory bandwidth at roughly the physical core count and
 * regresses beyond it, while one moving ~0.5 bytes still has work for a hyperthread.
 * So heavily quantized models get the logical count and wide-dtype models get the
 * physical count.
 *
 * The 1.0 byte/weight threshold sits between the two measured cases (bf16 at 2.0,
 * MXFP4 at ~0.53) and is an interpolation, not a measurement -- there is no data point
 * between them. It is recorded as an assumption so that a future benchmark at, say,
 * int8 (1.0) can move it with evidence. */
static int choose_threads(const EngHwInfo *hw, const EngModelFacts *m, EngPlan *p)
{
    const int phys = hw->cores_physical > 0 ? hw->cores_physical : 1;
    const int logi = hw->cores_logical  > 0 ? hw->cores_logical  : phys;

    if (m->bytes_per_weight > 0.0 && m->bytes_per_weight < 1.0) {
        if (logi > phys)
            note(p, "threads %d (logical): weights average %.2f bytes each, so the "
                    "matmuls are compute-bound enough for SMT to help "
                    "(measured 2.85x at 4 threads on 2 cores for MXFP4)",
                 logi, m->bytes_per_weight);
        return logi;
    }

    if (logi > phys)
        note(p, "threads %d (physical, not the %d logical): at %.2f bytes per weight "
                "the matmuls are bandwidth-bound, and the measured bf16 case regressed "
                "1.5%% going from 3 threads to 4",
             phys, logi, m->bytes_per_weight > 0 ? m->bytes_per_weight : 2.0);
    return phys;
}

int eng_plan(EngPlan *plan, const EngHwInfo *hw, const EngModelFacts *m,
             const EngPlanRequest *req)
{
    if (!plan || !hw || !m) return -1;
    EngPlanRequest def;
    if (!req) { eng_plan_request_init(&def); req = &def; }

    memset(plan, 0, sizeof *plan);
    plan->predicted_stream_hit_rate = -1.0;

    char a[32], b[32];

    /* ---- budget ---- */
    if (req->memory_budget > 0) {
        plan->memory_budget = req->memory_budget;
    } else {
        plan->memory_budget = eng_mem_auto_budget(hw->ram_available, hw->ram_total);
        eng_mem_human(plan->memory_budget, a, sizeof a);
        eng_mem_human(hw->ram_available, b, sizeof b);
        note(plan, "budget %s chosen from %s available RAM, keeping the larger of "
                   "1 GB or 20%% for the rest of the system", a, b);
    }
    if (plan->memory_budget <= 0) {
        snprintf(plan->problem, sizeof plan->problem,
                 "cannot determine a memory budget: RAM detection returned nothing. "
                 "Pass --memory explicitly.");
        return -1;
    }

    /* ---- threads ---- */
    plan->threads = req->threads > 0 ? req->threads : choose_threads(hw, m, plan);
    if (req->threads > 0 && req->threads > hw->cores_logical)
        note(plan, "threads %d exceeds the %d logical processors on this host; "
                   "oversubscription usually costs throughput",
             req->threads, hw->cores_logical);

    /* ---- memory partition ---- */
    EngMemNeeds needs;
    memset(&needs, 0, sizeof needs);
    needs.resident    = m->resident_bytes;
    needs.activations = m->activation_bytes;
    needs.scratch     = m->scratch_bytes;
    needs.kv_per_pos  = m->kv_bytes_per_pos;
    needs.context     = req->context > 0 ? req->context : m->context_max;
    /* One layer must fit, plus a second slot so reads can overlap compute. Below that
     * the engine still runs, just serially -- but a budget that cannot hold two layers
     * is a configuration worth flagging rather than silently accepting. */
    needs.min_weights = m->max_layer_bytes > 0 ? m->max_layer_bytes : m->avg_layer_bytes;

    int ctx = needs.context;
    if (eng_mem_plan(&plan->mem, plan->memory_budget, &needs, &ctx) != 0) {
        snprintf(plan->problem, sizeof plan->problem, "%s", plan->mem.problem);
        return -1;
    }
    plan->context = ctx;
    if (needs.context > 0 && ctx < needs.context) {
        eng_mem_human(m->kv_bytes_per_pos, a, sizeof a);
        note(plan, "context reduced %d -> %d: the KV cache costs %s per position and "
                   "the full request would not fit", needs.context, ctx, a);
    }

    /* ---- resident or streaming ---- */
    const int64_t streamable = m->total_weight_bytes - m->resident_bytes;
    const int64_t wbudget = plan->mem.weights;

    if (req->force_stream == 0 && wbudget < streamable) {
        eng_mem_human(streamable, a, sizeof a);
        eng_mem_human(wbudget, b, sizeof b);
        snprintf(plan->problem, sizeof plan->problem,
                 "residency was required but the streamable weights are %s and only "
                 "%s is available for them", a, b);
        return -1;
    }

    plan->streaming = (req->force_stream == 1) || (wbudget < streamable);
    if (!plan->streaming) {
        eng_mem_human(streamable, a, sizeof a);
        note(plan, "all %s of streamable weights fit in the budget: holding them "
                   "resident, no streaming", a);
        plan->stream_budget = streamable;
        plan->cache_budget  = wbudget - streamable;
        plan->predicted_stream_hit_rate = 1.0;
    } else {
        /* STREAM BEFORE CACHE. See planner.h: a gigabyte given to the sequentially
         * rescanned weights buys a deterministic hit-rate gain, while the same
         * gigabyte given to a data-dependent cache buys whatever reuse happens to
         * exist. K3 measured 1.69x for trunk-first at a fixed budget. */
        if (m->n_experts > 0 && m->expert_bytes > 0) {
            /* The expert cache needs at least topk+1 slots or it evicts an expert that
             * is still being multiplied. Take that minimum off the top, then give the
             * streamer everything else. */
            const int64_t cache_min = (int64_t)(m->topk + 1) * m->expert_bytes;
            if (wbudget > cache_min + m->max_layer_bytes * 2) {
                plan->cache_budget  = cache_min;
                plan->stream_budget = wbudget - cache_min;
                eng_mem_human(cache_min, a, sizeof a);
                note(plan, "expert cache held at its %s minimum (top-%d plus one) and "
                           "the rest given to the layer stream: re-read-every-token "
                           "weights repay memory more reliably than a routed cache",
                     a, m->topk);
            } else {
                plan->cache_budget  = wbudget / 2;
                plan->stream_budget = wbudget - plan->cache_budget;
                note(plan, "budget is tight: splitting weights evenly between the layer "
                           "stream and the expert cache");
            }
        } else {
            plan->stream_budget = wbudget;
            plan->cache_budget  = 0;
        }

        /* The streamer pins a prefix and rings the rest, so the hit rate is knowable
         * in advance: it is npin/nblocks. Estimate npin from the average layer. */
        if (m->avg_layer_bytes > 0 && m->n_layers > 0) {
            const int64_t slot = m->max_layer_bytes > 0 ? m->max_layer_bytes
                                                        : m->avg_layer_bytes;
            const int64_t after_ring = plan->stream_budget - 2 * slot;
            int npin = after_ring > 0 ? (int)(after_ring / m->avg_layer_bytes) : 0;
            if (npin > m->n_layers) npin = m->n_layers;
            plan->predicted_stream_hit_rate = (double)npin / (double)m->n_layers;
            eng_mem_human(plan->stream_budget, a, sizeof a);
            note(plan, "streaming: %s pins about %d of %d layers, a deterministic "
                       "%.0f%% hit rate (a cyclic scan defeats LRU, so the prefix is "
                       "pinned instead)",
                 a, npin, m->n_layers, 100.0 * plan->predicted_stream_hit_rate);
        }
    }

    /* ---- prefetch depth ---- */
    /* Depth 1 is all the layer walk can use: the order is fixed, so exactly one read
     * can be outstanding while one layer computes, and a second in-flight read would
     * need a third ring slot to land in. A rotational device benefits from deeper
     * BATCHES on the expert cache, which is a different mechanism (get_many), not this. */
    plan->prefetch_depth = plan->streaming ? 1 : 0;
    if (hw->storage == ENG_STORE_ROTATIONAL && plan->streaming)
        note(plan, "storage looks rotational: expect seek time to dominate, and prefer "
                   "a larger memory budget over a larger context here");

    /* ---- sanity notes ---- */
    if (!(hw->isa & ENG_ISA_AVX2))
        note(plan, "this CPU reports no AVX2: the portable scalar kernels will run, "
                   "correctly but several times slower");
    else if (!(hw->isa_built & ENG_ISA_AVX2))
        note(plan, "this CPU has AVX2 but the binary was built without it; rebuild with "
                   "-mavx2 -mfma for the vector kernels");

    if (hw->storage_free > 0 && hw->storage_free < m->total_weight_bytes / 10)
        note(plan, "less than 10%% of the model size is free on this filesystem");

    plan->ok = 1;
    return 0;
}

void eng_plan_report(const EngPlan *p, const EngModelFacts *m, const char *label)
{
    if (!p) return;
    char t[32];

    printf("%s%splan\n", label ? label : "", label ? " " : "");
    if (!p->ok) {
        printf("  NOT VIABLE: %s\n", p->problem);
        return;
    }

    eng_mem_human(p->memory_budget, t, sizeof t);
    printf("  budget    : %s\n", t);
    printf("  threads   : %d\n", p->threads);
    printf("  context   : %d%s\n", p->context,
           m && p->context < m->context_max ? " (reduced to fit)" : "");
    printf("  weights   : %s\n", p->streaming ? "streamed" : "fully resident");

    eng_mem_human(p->stream_budget, t, sizeof t);
    printf("  stream    : %s", t);
    if (p->predicted_stream_hit_rate >= 0.0)
        printf("  (predicted hit rate %.0f%%)", 100.0 * p->predicted_stream_hit_rate);
    printf("\n");

    if (p->cache_budget > 0) {
        eng_mem_human(p->cache_budget, t, sizeof t);
        printf("  expert $  : %s\n", t);
    }

    printf("\n");
    eng_mem_report(&p->mem, "  ");

    if (p->n_notes) {
        printf("\n  why:\n");
        for (int i = 0; i < p->n_notes; i++)
            printf("    - %s\n", p->notes[i]);
    }
}
