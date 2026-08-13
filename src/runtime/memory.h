/* SPDX-License-Identifier: Apache-2.0 */
/*
 * memory.h - one budget, partitioned explicitly.
 *
 * WHAT WENT WRONG BEFORE
 *   K3 had no memory manager. It had two independent CLI flags, --trunk-gb and
 *   --cache-gb, each passed straight to its own allocator. Nothing checked their sum
 *   against RAM, nothing accounted for the KV cache or scratch, and the banner printed
 *   before allocating was documented as "a PLAN, not a measurement" that overstated by
 *   0.13-1.84 GB across a 12-rung ladder. The authoritative number came from getrusage
 *   AFTER the run, which is too late to prevent anything.
 *
 *   So a user could ask for budgets summing to more than the machine had, and find out
 *   by being killed by the OOM killer partway through a token.
 *
 * WHAT THIS DOES INSTEAD
 *   Takes ONE budget and divides it, in a fixed priority order, into the five things
 *   that consume memory:
 *
 *     reserve      the OS, the allocator's own overhead, and headroom for the
 *                  measurement error the plan itself carries
 *     resident     weights that CANNOT stream (embeddings, the LM head): a model that
 *                  cannot hold these cannot run at all, so they are taken first
 *     activations  hidden states and per-layer intermediates
 *     scratch      kernel working buffers, sized by the kernels themselves
 *     kv_cache     grows with context; the one component the user can trade directly
 *     weights      everything left, split between the streamer and the weight cache
 *
 *   The order is the point. Reserve and resident are not negotiable, so they come out
 *   first; the KV cache is bounded by what the requested context needs and no more; and
 *   `weights` is the remainder, which is what makes "memory is a dial" true. If the
 *   remainder cannot hold even one layer, that is a refusal at plan time with both
 *   figures side by side, not a crash later.
 *
 * THE RESERVE IS NOT A ROUND NUMBER PULLED FROM THE AIR
 *   It is max(768 MB, 12% of the budget). The floor covers the allocator, the
 *   safetensors or GGUF index, thread stacks and page tables; the proportion covers the
 *   fact that every component's estimate is approximate and the errors correlate
 *   upward. K3's own plan-versus-measured gap ran to 1.84 GB on a 128 GB budget, which
 *   is 1.4%, so 12% is roughly an order of magnitude of headroom over the observed
 *   error -- deliberately generous, because the failure mode it prevents is the OOM
 *   killer.
 */
#ifndef ENG_MEMORY_H
#define ENG_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t budget;       /* the total this plan may use */

    int64_t reserve;
    int64_t resident;     /* weights that cannot stream */
    int64_t activations;
    int64_t scratch;
    int64_t kv_cache;
    int64_t weights;      /* remainder, for the streamer and the weight cache */

    int     ok;           /* 0 when the budget cannot satisfy the fixed costs */
    char    problem[256]; /* why, when ok is 0 */
} EngMemPlan;

/* What the caller knows it needs. Anything unknown may be left zero. */
typedef struct {
    int64_t resident;
    int64_t activations;
    int64_t scratch;
    int64_t kv_per_pos;      /* bytes of KV cache per token position */
    int     context;         /* requested context length */
    int64_t min_weights;     /* smallest useful weight budget, e.g. one layer */
} EngMemNeeds;

/* Partition `budget` across the needs. Never allocates; this is arithmetic and a
 * verdict. Returns 0 when the plan is viable, non-zero when it is not (and fills
 * plan->problem with a message naming both the shortfall and what would fix it).
 *
 * When the KV cache for the requested context will not fit, the context is REDUCED to
 * what does fit rather than the plan being refused: a shorter run is almost always more
 * useful than no run. plan->kv_cache reflects the reduced figure, and the caller is
 * expected to read back the context it actually got. */
int eng_mem_plan(EngMemPlan *plan, int64_t budget, const EngMemNeeds *needs,
                 int *context_out);

/* Largest context whose KV cache fits in `avail` bytes. */
int eng_mem_max_context(int64_t avail, int64_t kv_per_pos);

/* Parse a size: "5G", "512M", "2048K", "1073741824", or "auto" (yields 0 and sets
 * *is_auto). Accepts lower case and a trailing 'B'. Returns -1 on anything else rather
 * than guessing, because a mistyped budget that silently becomes a different number is
 * how a machine gets OOM-killed. */
int64_t eng_mem_parse_size(const char *s, int *is_auto);

/* A safe budget derived from what the OS says is available. Deliberately conservative:
 * leaves the larger of 1 GB or 20% of available to the rest of the system, because the
 * engine is not the only thing running and the measurement is a moment in time. */
int64_t eng_mem_auto_budget(int64_t ram_available, int64_t ram_total);

void eng_mem_report(const EngMemPlan *plan, const char *label);

/* Human-readable byte count into buf: "4.68 GB", "512 MB", "17.55 MB". */
void eng_mem_human(int64_t bytes, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* ENG_MEMORY_H */
