/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_cache_generic.c - the generic keyed cache, independent of any model.
 *
 * The cases here target the four fixes documented in cache.h, each of which was a real
 * bug that produced a wrong answer rather than a crash:
 *
 *   1. batch prefetch must not hand the same slot to two keys in one batch;
 *   2. a failed load must leave its slot EMPTY, never owning the key -- otherwise the
 *      next request counts a HIT and serves whatever was in the buffer;
 *   3. the slot STRIDE must be alignment-rounded, not just the arena base. K3's real
 *      expert size happened to be a multiple of 4096 so this held by coincidence; the
 *      payload size here is deliberately NOT, which is what makes the test bite;
 *   4. prefetched bytes must be counted separately or the hit rate becomes a lie.
 *
 * Every object is filled with a key-derived byte pattern, so a slot that gets the wrong
 * bytes -- from a stride bug, an aliased slot, or a published failed read -- is
 * detected rather than assumed absent.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cache.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-44s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

static void eqi(long long got, long long want, const char *what)
{
    char d[96];
    snprintf(d, sizeof d, "got %lld want %lld", got, want);
    ok(got == want, what, d);
}

/* ---------------------------------------------------------------- the source -- */

/* DELIBERATELY NOT a multiple of ENG_IO_ALIGN (4096). If the cache rounds only the
 * arena base and not the per-slot stride, slots after the first are misaligned and a
 * direct read into them fails -- which is exactly the bug this size exposes. */
#define OBJ_BYTES 5000
#define NKEYS     32

typedef struct {
    int64_t fail_key;      /* this key's load returns short, to test fix 2 */
    int     describe_calls;
    int     load_calls;
} Src;

/* Byte pattern: object k, byte i -> (k*7 + i) mod 251. Prime modulus so the period
 * does not align with any block or page size. */
static unsigned char pat(int64_t k, int64_t i) { return (unsigned char)((k * 7 + i) % 251); }

static int src_describe(void *ctx, EngCacheKey key, int64_t *nbytes, void *meta)
{
    Src *s = (Src *)ctx;
    s->describe_calls++;
    if (key >= NKEYS) return -1;
    *(int64_t *)meta = (int64_t)key;      /* meta remembers which object this slot holds */
    *nbytes = OBJ_BYTES;
    return 0;
}

static int64_t src_load(void *ctx, EngCacheKey key, const void *meta,
                        unsigned char *buf, int64_t bufcap, int64_t *payload_off)
{
    Src *s = (Src *)ctx;
    if (bufcap < OBJ_BYTES) return 0;

    /* The meta must be the one describe() wrote for THIS key. If the cache handed two
     * keys the same slot, this fires. */
    if (*(const int64_t *)meta != (int64_t)key) return -1;

    /* Emulate a direct read widened to an alignment boundary: payload does not start
     * at zero. A caller that ignores payload_off reads the wrong bytes. */
    const int64_t pad = 64;
    if (bufcap < OBJ_BYTES + pad) return 0;
    *payload_off = pad;

    if ((int64_t)key == s->fail_key) return 17;      /* short read, on purpose */

    for (int64_t i = 0; i < OBJ_BYTES; i++) buf[pad + i] = pat((int64_t)key, i);
    __sync_fetch_and_add(&s->load_calls, 1);
    return OBJ_BYTES;
}

static void src_locate(void *ctx, const void *meta, int *device, int64_t *offset)
{
    (void)ctx;
    *device = 0;
    /* Reverse of key order, so a cache that sorts by locate() visibly reorders. */
    *offset = (NKEYS - *(const int64_t *)meta) * OBJ_BYTES;
}

static int verify(const EngCacheEntry *e, int64_t key)
{
    if (!e->data || e->nbytes != OBJ_BYTES) return 0;
    for (int64_t i = 0; i < OBJ_BYTES; i++)
        if (e->data[i] != pat(key, i)) return 0;
    return 1;
}

static EngCache *build(Src *s, int64_t budget, int min_slots, int trace)
{
    EngCacheCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.name         = "test";
    cfg.budget_bytes = budget;
    cfg.slot_bytes   = OBJ_BYTES;
    cfg.nkeys        = NKEYS;
    cfg.min_slots    = min_slots;
    cfg.meta_bytes   = sizeof(int64_t);
    cfg.want_hist    = 1;
    cfg.want_trace   = trace;
    cfg.hugepages    = 0;          /* deterministic in a test */

    EngCacheSrc cs;
    memset(&cs, 0, sizeof cs);
    cs.describe = src_describe;
    cs.load     = src_load;
    cs.locate   = src_locate;
    cs.ctx      = s;

    return eng_cache_create(&cfg, &cs);
}

/* Budget for exactly n slots. Uses the cache's own stride function rather than
 * restating the formula: a test that re-derives the rounding would agree with a buggy
 * implementation. */
static int64_t budget_for(int n)
{
    return eng_cache_stride_for(OBJ_BYTES) * n;
}

int main(void)
{
    printf("generic keyed cache\n\n");

    Src s;

    /* ---- construction refusals ---- */
    printf("== construction ==\n");
    memset(&s, 0, sizeof s); s.fail_key = -1;
    ok(build(&s, budget_for(2), 8, 0) == NULL,
       "refuses a budget below min_slots", "2 slots, 8 required");
    {
        EngCacheCfg bad; memset(&bad, 0, sizeof bad);
        bad.nkeys = 0; bad.slot_bytes = 100; bad.budget_bytes = 1 << 20;
        EngCacheSrc cs; memset(&cs, 0, sizeof cs);
        cs.describe = src_describe; cs.load = src_load; cs.ctx = &s;
        ok(eng_cache_create(&bad, &cs) == NULL, "refuses empty key space", NULL);
    }

    /* ---- basic serve, and the STRIDE ALIGNMENT case ---- */
    printf("\n== serve and slot addressing ==\n");
    memset(&s, 0, sizeof s); s.fail_key = -1;
    EngCache *c = build(&s, budget_for(8), 4, 1);
    ok(c != NULL, "create 8-slot cache", NULL);
    if (!c) return 1;
    eqi(eng_cache_slots(c), 8, "slot count");

    /* Fill every slot and verify the CONTENT of each. With a 5000-byte object, an
     * unrounded stride puts slot 1 onward off a page boundary; the bytes would be
     * wrong or absent, not merely slow. */
    int good = 1;
    for (int64_t k = 0; k < 8; k++) {
        EngCacheEntry e;
        if (eng_cache_get(c, (EngCacheKey)k, &e) != 0 || !verify(&e, k)) good = 0;
    }
    ok(good, "8 distinct objects, all bytes correct", "5000-byte payload, unaligned");

    /* Every slot must still hold its own object: re-read them all. */
    good = 1;
    for (int64_t k = 0; k < 8; k++) {
        EngCacheEntry e;
        if (eng_cache_get(c, (EngCacheKey)k, &e) != 0 || !verify(&e, k)) good = 0;
    }
    ok(good, "re-read all 8 still correct", "no slot aliasing");

    {
        EngCacheStats st; eng_cache_stats(c, &st);
        eqi((long long)st.misses, 8, "8 misses on first pass");
        eqi((long long)st.hits, 8, "8 hits on second pass");
        eqi((long long)st.evictions, 0, "no evictions yet");
    }

    /* ---- peek must not load ---- */
    printf("\n== peek ==\n");
    {
        EngCacheEntry e;
        ok(eng_cache_peek(c, 3, &e) == 1, "peek finds resident key", NULL);
        ok(eng_cache_peek(c, 20, &e) == 0, "peek misses absent key", NULL);
        const int before = s.load_calls;
        eng_cache_peek(c, 20, &e);
        eqi(s.load_calls, before, "peek issued no load");
    }

    /* ---- LRU eviction ---- */
    printf("\n== LRU eviction ==\n");
    {
        EngCacheEntry e;
        /* Touch 1..7 so 0 is the least recently used, then admit a 9th object. */
        for (int64_t k = 1; k < 8; k++) eng_cache_get(c, (EngCacheKey)k, &e);
        ok(eng_cache_get(c, 8, &e) == 0 && verify(&e, 8), "admit 9th object", NULL);
        ok(eng_cache_peek(c, 0, &e) == 0, "LRU victim (key 0) was evicted", NULL);
        ok(eng_cache_peek(c, 1, &e) == 1, "recently used key 1 retained", NULL);
    }

    /* ---- pinning ---- */
    printf("\n== pinning ==\n");
    {
        EngCacheEntry e;
        ok(eng_cache_pin(c, 1, 1) == 1, "pin resident key", NULL);
        ok(eng_cache_pin(c, 30, 1) == 0, "pin absent key returns 0", NULL);
        /* Churn enough distinct objects to cycle every unpinned slot several times. */
        for (int64_t k = 9; k < NKEYS; k++) eng_cache_get(c, (EngCacheKey)k, &e);
        for (int64_t k = 9; k < NKEYS; k++) eng_cache_get(c, (EngCacheKey)k, &e);
        ok(eng_cache_peek(c, 1, &e) == 1, "pinned key survived full churn", NULL);
        ok(eng_cache_peek(c, 1, &e) && verify(&e, 1), "pinned key bytes intact", NULL);
        eng_cache_pin(c, 1, 0);
    }

    /* ---- histogram and trace ---- */
    printf("\n== histogram and trace ==\n");
    {
        int64_t nk = 0;
        const uint32_t *h = eng_cache_hist(c, &nk);
        ok(h != NULL && nk == NKEYS, "histogram present", NULL);
        ok(h && h[1] > 1, "hot key counted more than once", NULL);

        int64_t n = 0;
        const EngCacheKey *tr = eng_cache_trace(c, &n);
        ok(tr != NULL && n > 0, "trace recorded", NULL);
        /* The trace records the REQUEST, not the outcome: total trace length must equal
         * total requests, hits included. Recording only misses would make a replay at a
         * different capacity meaningless. */
        EngCacheStats st; eng_cache_stats(c, &st);
        eqi((long long)n, (long long)st.requests, "trace length == requests");
    }
    eng_cache_report(c, "  ");
    eng_cache_destroy(c);

    /* ---- FIX 2: a failed load must not be published ---- */
    printf("\n== failed load is not served as a hit ==\n");
    memset(&s, 0, sizeof s);
    s.fail_key = 5;                      /* key 5's load returns short */
    c = build(&s, budget_for(4), 2, 0);
    ok(c != NULL, "create 4-slot cache", NULL);
    if (c) {
        EngCacheEntry e;
        ok(eng_cache_get(c, 5, &e) != 0, "short load reported as failure", NULL);
        ok(eng_cache_peek(c, 5, &e) == 0, "failed key is NOT resident afterwards",
           "publishing it would serve garbage as a hit");
        /* And the slot must be reusable, not leaked as INFLIGHT forever. */
        ok(eng_cache_get(c, 6, &e) == 0 && verify(&e, 6),
           "slot reusable after failed load", NULL);
        ok(eng_cache_get(c, 5, &e) != 0, "retry of failing key still fails", NULL);
        eng_cache_destroy(c);
    }

    /* ---- FIX 1 + 4: batch prefetch ---- */
    printf("\n== batch prefetch ==\n");
    memset(&s, 0, sizeof s); s.fail_key = -1;
    c = build(&s, budget_for(8), 4, 0);
    ok(c != NULL, "create 8-slot cache", NULL);
    if (c) {
        const EngCacheKey batch[6] = { 10, 11, 12, 13, 14, 15 };
        const int got = eng_cache_get_many(c, batch, 6);
        eqi(got, 6, "batch admitted all 6");

        /* THE ALIASING CASE. If two keys in one batch were handed the same slot, at
         * least one of these reads back the wrong object. src_load also asserts its
         * meta matches its key, so aliasing fails twice over. */
        int allgood = 1;
        for (int i = 0; i < 6; i++) {
            EngCacheEntry e;
            if (eng_cache_get(c, batch[i], &e) != 0 || !verify(&e, (int64_t)batch[i]))
                allgood = 0;
        }
        ok(allgood, "all 6 batched objects have correct bytes",
           "distinct slots, no aliasing");

        /* Statistics honesty: those 6 reads happened during the batch, so they must be
         * counted as prefetch_reads even though get() now records hits. */
        EngCacheStats st; eng_cache_stats(c, &st);
        eqi((long long)st.prefetch_reads, 6, "prefetch_reads counted");
        ok(st.prefetch_reads <= st.hits + st.misses, "prefetch_reads <= requests", NULL);
        eqi((long long)st.hits, 6, "batched objects then read as hits");

        /* A duplicate key inside one batch must be admitted once, not twice. */
        const EngCacheKey dup[4] = { 20, 20, 21, 20 };
        eqi(eng_cache_get_many(c, dup, 4), 2, "duplicates collapsed in a batch");

        /* Re-batching resident keys must do no work at all. */
        eqi(eng_cache_get_many(c, batch, 6), 0, "re-batch of resident keys is a no-op");

        eng_cache_report(c, "  ");
        eng_cache_destroy(c);
    }

    /* ---- range checks ---- */
    printf("\n== range checks ==\n");
    memset(&s, 0, sizeof s); s.fail_key = -1;
    c = build(&s, budget_for(4), 2, 0);
    if (c) {
        EngCacheEntry e;
        ok(eng_cache_get(c, NKEYS + 5, &e) != 0, "out-of-range key refused", NULL);
        ok(eng_cache_peek(c, NKEYS + 5, &e) == 0, "out-of-range peek refused", NULL);
        ok(eng_cache_pin(c, NKEYS + 5, 1) == 0, "out-of-range pin refused", NULL);
        eng_cache_destroy(c);
    }

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("GENERIC CACHE TESTS FAILED\n"); return 1; }
    printf("GENERIC CACHE TESTS PASSED\n");
    return 0;
}
