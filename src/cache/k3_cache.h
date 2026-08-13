/* k3_cache.h - K3's routed-expert cache, now a thin binding over the generic one.
 *
 * WHAT MOVED, AT M2
 *   The replacement policy, slot bookkeeping, pinning, statistics, histogram, access
 *   trace and three-phase batch prefetch all live in src/storage/cache.c now. They were
 *   never K3-specific: what was K3-specific is only the KEY (a (layer, expert) pair)
 *   and the PAYLOAD (six MXFP4 tensors forming one contiguous run).
 *
 *   This file is what remains: the key encoding, the K3ExpertSrc vtable K3's MoE
 *   kernels already expect, and a K3ExpertRef stashed as the generic cache's per-slot
 *   metadata so fill_q can resolve a slot back to three (packed, scale) pairs.
 *
 *   The struct below keeps the same public fields it always had, mirrored from the
 *   generic cache after every operation. That is deliberate: it means tests/unit/
 *   test_cache.c and src/cli/k3_run.c are UNCHANGED across this refactor, so they are a
 *   real regression signal rather than a test edited to match new code.
 *
 * THE PARTS THAT STILL MATTER, and where they went
 *   - the cache holds MXFP4 and never floats. Dequantised an expert is 132 MB against
 *     17.55 MB packed, and k3_matmul_mxfp4 consumes the packed form directly, so
 *     caching floats would cut residency 7.5x for no benefit. That is a property of
 *     what K3 puts IN the cache, so it stays here.
 *   - capacity must exceed topk, or a slot handed out is evicted before it is used.
 *     Enforced by passing min_slots = topk + 1 to the generic constructor.
 *   - the histogram is the input to any pinning strategy, and the trace lets one
 *     expensive run yield the whole hit-rate-versus-capacity curve offline. Both are
 *     generic now; the dump functions here re-encode them into the (layer, expert)
 *     int32 pairs tools/sim_cache.py already reads.
 */
#ifndef K3_CACHE_H
#define K3_CACHE_H

#include "k3.h"
#include "k3_load.h"
#include "k3_st.h"

#include "cache.h"      /* src/storage: the generic keyed cache */

typedef struct {
    K3ExpertSrc  src;             /* MUST be first: &cache->src is passed to K3MoeW */

    EngCache    *gc;              /* the generic cache does the work */
    const K3St  *st;
    int          n_layers, n_experts;

    /* Mirrored from the generic cache after every operation. Present so existing
     * callers keep compiling and reading the same numbers; the authority is `gc`. */
    int64_t      slot_bytes;      /* per-slot stride, including alignment room */
    int          nslot;
    uint64_t     hits, misses, evictions, bytes_read, prefetch_reads;
    double       load_seconds;
} K3Cache;

/* budget_bytes is the arena size, rounded down to whole experts. Fails if that leaves
 * fewer than topk+1 slots. */
int  k3_cache_init(K3Cache *c, const K3St *st, const K3Cfg *cfg, int64_t budget_bytes);
void k3_cache_free(K3Cache *c);

/* Pin or unpin whatever slot currently holds this expert. Returns 0 if not resident. */
int  k3_cache_pin(K3Cache *c, int layer, int expert, int pin);

/* Load an expert without returning it, so a prefetcher can warm the cache. */
int  k3_cache_prefetch(K3Cache *c, int layer, int expert);

void k3_cache_reset_stats(K3Cache *c);
void k3_cache_report(const K3Cache *c, const char *label);

/* Request histogram as JSON, for offline analysis of the hot set. */
int  k3_cache_dump_hist(const K3Cache *c, const char *path);

/* Access trace as flat int32 (layer, expert) pairs in request order, which is the
 * format tools/sim_cache.py replays. The generic cache stores flat keys; this decodes
 * them back to pairs on the way out so the tool is unaffected. */
int  k3_cache_dump_trace(const K3Cache *c, const char *path);

#endif /* K3_CACHE_H */
