/* SPDX-License-Identifier: Apache-2.0 */
/*
 * cache.h - a keyed, slot-based weight cache with LRU replacement.
 *
 * This is k3_cache.c generalised. The policy, the statistics, the pinning, the
 * histogram, the access trace and the three-phase batch prefetch are all carried over;
 * what changes is that the key is now an opaque integer instead of
 * (layer, expert), and the payload is described by a caller-supplied vtable instead of
 * being hardcoded to a routed MXFP4 expert.
 *
 * FOUR THINGS HERE ARE FIXES FOR REAL BUGS, not design preferences. Each cost a wrong
 * answer on the real model and each is preserved deliberately:
 *
 *   1. A SLOT HAS THREE STATES, not two: holding a key, EMPTY, or INFLIGHT. The batch
 *      prefetch marks a slot before reading into it so a failed read cannot leave the
 *      slot claiming an object it does not hold. But the victim search returns an EMPTY
 *      slot immediately as a fast path, ahead of the pinned check -- so marking it EMPTY
 *      handed the SAME slot to the next key in the same batch, several parallel reads
 *      wrote into one buffer, and the model multiplied garbage. It cost exactly one
 *      wrong token on the real checkpoint and nothing at all in the fixtures.
 *
 *   2. PUBLISH ONLY AFTER THE READ SUCCEEDS. Registering a key up front and then
 *      reading leaves a failed read owning a slot that claims to hold the object, so
 *      the next request counts a HIT and multiplies whatever was in the buffer.
 *
 *   3. THE SLOT STRIDE MUST BE ALIGNMENT-ROUNDED, not just the arena base. Aligning the
 *      arena aligns slot 0 and nothing else, because slot N starts at
 *      arena + N*stride. On K3's real checkpoint an expert is 17,547,264 bytes, which
 *      happens to be exactly 4284*4096, so this held BY COINCIDENCE and the engine
 *      worked. At any other object size every direct read into every slot after the
 *      first returns zero bytes and the cache silently serves nothing.
 *
 *   4. PREFETCHED BYTES ARE COUNTED SEPARATELY. A prefetched object is resident by the
 *      time get() asks for it, so get() records a hit -- but the bytes still moved.
 *      Without prefetch_reads the report shows the hit rate climbing while the I/O does
 *      not fall. Effective hit rate is (hits - prefetch_reads) / requests.
 *
 * WHY LRU, AND WHERE IT IS THE WRONG POLICY
 *   LRU suits data-dependent reuse, which is what MoE expert routing is. It is the
 *   WORST possible policy for a cyclic sequential scan: with N < L slots over L blocks,
 *   by the time the walk returns to block 0 it is exactly the least recently used thing
 *   and has just been evicted, so the hit rate is zero no matter how much RAM is added.
 *   Anything walking layers in fixed order wants the pinned-prefix streamer
 *   (storage/streamer.c, M3) instead. Choosing between them is the planner's job; this
 *   cache does not pretend to cover both.
 *
 * KEY SPACE
 *   Keys are dense integers in [0, nkeys). The index is a direct array, which is what
 *   K3 used (82,432 entries, 330 KB) and what a per-layer cache needs (tens). A sparse
 *   or unbounded key space would want a hash instead; it is not built until something
 *   needs it, and the constructor rejects an out-of-range key rather than silently
 *   growing.
 *
 * THREAD SAFETY
 *   Not thread-safe for concurrent get() calls. The batch path parallelises only the
 *   READS, which target distinct pre-assigned buffers and go through pread (which takes
 *   its offset as an argument and touches no shared file position). All bookkeeping is
 *   serial, by construction.
 */
#ifndef ENG_CACHE_H
#define ENG_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t EngCacheKey;

typedef struct EngCache EngCache;

typedef struct {
    unsigned char *data;    /* payload start, already past any alignment padding */
    int64_t        nbytes;
    const void    *meta;    /* the meta_bytes this key was described with */
} EngCacheEntry;

/* How the cache learns about, and fetches, an object. */
typedef struct {
    /* Resolve a key: how many payload bytes, and fill `meta` (meta_bytes wide) with
     * whatever the owner needs to interpret the buffer later. Return 0 on success,
     * non-zero to refuse the key. Called serially. */
    int (*describe)(void *ctx, EngCacheKey key, int64_t *nbytes, void *meta);

    /* Read the object into buf. Returns payload bytes delivered and sets *payload_off
     * to where the payload begins inside buf (non-zero when a direct read was widened
     * to alignment boundaries).
     *
     * MUST BE SAFE TO CALL CONCURRENTLY for distinct keys and distinct buffers: the
     * batch path calls it from an OpenMP loop. */
    int64_t (*load)(void *ctx, EngCacheKey key, const void *meta,
                    unsigned char *buf, int64_t bufcap, int64_t *payload_off);

    /* OPTIONAL ordering hint, may be NULL. Fills a (device, offset) pair so a batch can
     * be issued in storage order rather than key order. Objects are not stored
     * key-ordered on disk, so sorting by where the bytes actually live turns a
     * scattered set of seeks into a mostly forward sweep. */
    void (*locate)(void *ctx, const void *meta, int *device, int64_t *offset);

    void *ctx;
} EngCacheSrc;

typedef struct {
    const char *name;          /* for diagnostics, e.g. "expert" or "layer" */
    int64_t budget_bytes;      /* arena size; rounded DOWN to whole slots */
    int64_t slot_bytes;        /* max PAYLOAD per object; the cache adds alignment room */
    int64_t nkeys;             /* key space, must be > 0 */
    int     min_slots;         /* refuse to build with fewer (K3 passes topk+1) */
    size_t  meta_bytes;        /* per-slot opaque metadata */
    int     want_hist;         /* per-key request counts */
    int     want_trace;        /* record request order for offline replay */
    int     hugepages;         /* 1 on, 0 off, -1 auto (on unless ENG_NOHUGE) */
} EngCacheCfg;

/* Bytes one slot occupies for a given max payload, including the alignment room a
 * widened direct read needs at both ends.
 *
 * EXPOSED SO NOBODY RE-DERIVES IT. The rounding here is fix 3 from the header comment:
 * a caller that computes `payload + 2*align` without rounding the result gets a stride
 * that misaligns every slot after the first, and on K3's real expert size that mistake
 * is invisible because 17,547,264 happens to be a whole number of pages. Use this to
 * size a budget: budget_for(n) == n * eng_cache_stride_for(payload). */
int64_t eng_cache_stride_for(int64_t payload_bytes);

/* Returns NULL and explains on stderr when the budget cannot supply min_slots. That
 * refusal matters: a cache smaller than one step's working set evicts an object that is
 * still being read. */
EngCache *eng_cache_create(const EngCacheCfg *cfg, const EngCacheSrc *src);
void      eng_cache_destroy(EngCache *c);

/* Serve a key, loading it if absent. Records the request in the histogram and trace
 * BEFORE serving, so a replay reflects what the model asked for rather than what the
 * cache happened to hold. Returns 0 on success. */
int eng_cache_get(EngCache *c, EngCacheKey key, EngCacheEntry *out);

/* Resident-only lookup: 0 if it would need a read. Records nothing. */
int eng_cache_peek(EngCache *c, EngCacheKey key, EngCacheEntry *out);

/* Bring up to n keys resident with the reads issued CONCURRENTLY. Returns the number
 * newly admitted, or -1. A short return is NOT fatal: get() will miss on the remainder
 * and read it serially, which is always correct and only slower. */
int eng_cache_get_many(EngCache *c, const EngCacheKey *keys, int n);

/* Pin or unpin whatever slot holds this key. Returns 0 if it is not resident. */
int eng_cache_pin(EngCache *c, EngCacheKey key, int pin);

/* ---------------------------------------------------------------- statistics -- */

typedef struct {
    uint64_t hits, misses, evictions, bytes_read, prefetch_reads, requests;
    double   load_seconds;
    int      nslot;
    int64_t  slot_stride;
    int64_t  arena_bytes;
} EngCacheStats;

void eng_cache_stats(const EngCache *c, EngCacheStats *out);
void eng_cache_reset_stats(EngCache *c);
void eng_cache_report(const EngCache *c, const char *label);

int  eng_cache_slots(const EngCache *c);

/* Per-key request counts, or NULL when want_hist was 0. Length is nkeys. Which objects
 * are hot is not knowable in advance and is the INPUT to any pinning strategy, so this
 * is data, not instrumentation. */
const uint32_t *eng_cache_hist(const EngCache *c, int64_t *nkeys);

/* Request order, or NULL when want_trace was 0. Length in *n.
 *
 * Worth recording because replacement policy can then be evaluated offline: routing
 * decisions do not depend on the cache, so ONE expensive run yields the entire
 * hit-rate-versus-capacity curve at any capacity and under any policy. */
const EngCacheKey *eng_cache_trace(const EngCache *c, int64_t *n);

#ifdef __cplusplus
}
#endif

#endif /* ENG_CACHE_H */
