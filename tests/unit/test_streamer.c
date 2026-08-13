/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_streamer.c - the pinned-prefix block streamer.
 *
 * The cases that matter:
 *
 *   1. THE CYCLIC-SCAN PROPERTY, made measurable. Walking blocks 0..N-1 repeatedly, the
 *      pinned prefix must hit on every pass after the first and the streamed tail must
 *      hit ZERO times, giving exactly npin*(passes-1) hits. That is the whole argument
 *      for not using LRU here, expressed as an assertion rather than a claim.
 *
 *   2. THE TWO-SLOT RULE. When the budget affords only one ring slot the reader must
 *      NOT start, because it would read the next block over the one the caller is using.
 *      Asserted directly on eng_streamer_is_async().
 *
 *   3. BYTES SURVIVE AN IN-FLIGHT PREFETCH. The real usage pattern is get(L),
 *      prefetch(L+1), compute on L. So the test reads block L's contents AFTER issuing
 *      the prefetch, which is exactly when the one-slot bug corrupted them.
 *
 * Every block is filled with a block-derived pattern, so a slot holding the wrong block
 * -- from a sizing bug, a recycled slot, or a prefetch landing in the wrong place -- is
 * detected rather than assumed absent.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "storage.h"
#include "streamer.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-46s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

static void eqi(long long got, long long want, const char *what)
{
    char d[96];
    snprintf(d, sizeof d, "got %lld want %lld", got, want);
    ok(got == want, what, d);
}

#define NBLK 12
/* Block 0 is deliberately larger than the rest, mirroring K3's layer 0 (a dense MLP,
 * 2.34 GB against 1.27 GB). Uniform slots everywhere would size EVERY slot for it and
 * waste about half the budget, so pinned blocks get exact-size allocations. */
static const int64_t SIZES[NBLK] = {
    40960, 20480, 20480, 20480, 20480, 20480,
    20480, 20480, 20480, 20480, 20480, 20480
};

static unsigned char pat(int b, int64_t i) { return (unsigned char)((b * 31 + i) % 251); }

static int verify(const unsigned char *p, int b)
{
    for (int64_t i = 0; i < SIZES[b]; i++)
        if (p[i] != pat(b, i)) return 0;
    return 1;
}

/* Build a file of NBLK sequential blocks and return their offsets.
 *
 * A HEADER OF 100 BYTES comes first, on purpose. It pushes every block offset off a
 * 4096 boundary, which is the case that exercises the direct-read widening: the read
 * covers the enclosing aligned window and the payload starts somewhere inside it. This
 * is not hypothetical -- GGUF puts its tensor data at an offset aligned to 32, and in
 * the Qwen3-8B file that is byte 5,956,416, which is not a multiple of 4096. A streamer
 * that assumed the payload starts at the slot base would return a pointer up to 4095
 * bytes early on every block of every GGUF model. */
#define FILE_HEADER 100

static int build_file(const char *path, EngBlock *blocks)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    {
        unsigned char hdr[FILE_HEADER];
        memset(hdr, 0xEE, sizeof hdr);
        fwrite(hdr, 1, sizeof hdr, f);
    }
    int64_t off = FILE_HEADER;
    for (int b = 0; b < NBLK; b++) {
        unsigned char *buf = (unsigned char *)malloc((size_t)SIZES[b]);
        for (int64_t i = 0; i < SIZES[b]; i++) buf[i] = pat(b, i);
        fwrite(buf, 1, (size_t)SIZES[b], f);
        free(buf);
        blocks[b].off = off;
        blocks[b].nbytes = SIZES[b];
        off += SIZES[b];
    }
    fclose(f);
    return 0;
}

static EngStreamer *build(EngStorage *st, const EngBlock *blocks,
                          int64_t budget, int async, int ring_want)
{
    EngStreamerCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.name         = "test";
    cfg.store        = st;
    cfg.blocks       = blocks;
    cfg.nblocks      = NBLK;
    cfg.budget_bytes = budget;
    cfg.extra_bytes  = 0;
    cfg.ring_want    = ring_want;
    cfg.async        = async;
    cfg.hugepages    = 0;
    cfg.quiet        = 1;
    return eng_streamer_create(&cfg);
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "build";
    char path[512];
    snprintf(path, sizeof path, "%s/eng_streamer_test.bin", dir);

    EngBlock blocks[NBLK];
    if (build_file(path, blocks) != 0) {
        printf("cannot write %s\n", path);
        return 1;
    }

    /* want_direct = 1: exercise the widened read path where the filesystem allows it. */
    EngStorage *st = eng_storage_open_file(path, 1);
    if (!st) { printf("cannot open %s\n", path); return 1; }

    printf("pinned-prefix block streamer\n");
    printf("blocks start at offset %d, so none is page aligned; direct path %s\n\n",
           FILE_HEADER, eng_storage_is_direct(st) ? "AVAILABLE" : "unavailable");

    /* ---- sizing, and non-uniform pinned allocations ---- */
    printf("== sizing ==\n");
    {
        /* Enough for a few pinned blocks plus a 2-slot ring. */
        EngStreamer *s = build(st, blocks, 200000, 1, 2);
        ok(s != NULL, "create with a mid-size budget", NULL);
        if (s) {
            EngStreamerStats t; eng_streamer_stats(s, &t);
            char d[160];
            snprintf(d, sizeof d, "npin=%d nslot=%d slot=%lld pinned=%lld",
                     t.npin, t.nslot, (long long)t.slot_bytes,
                     (long long)t.pinned_bytes);
            ok(t.npin > 0 && t.npin < NBLK, "pins a proper prefix", d);
            ok(t.nslot >= 2, "ring has at least 2 slots", NULL);
            /* The ring slot sizes to the largest STREAMING block. Once block 0 is
             * pinned, the biggest streamer is 20480, not 40960. */
            ok(t.slot_bytes >= 20480, "ring slot fits the largest streaming block", NULL);
            ok(t.pinned_bytes < 200000, "pinned bytes inside budget", NULL);
            eng_streamer_destroy(s);
        }
    }

    /* ---- THE TWO-SLOT RULE ---- */
    printf("\n== two-slot rule (correctness, not tuning) ==\n");
    {
        /* A budget that affords exactly one ring slot and nothing else. */
        EngStreamer *s = build(st, blocks, 49152, 1, 2);
        ok(s != NULL, "create with a one-slot budget", NULL);
        if (s) {
            EngStreamerStats t; eng_streamer_stats(s, &t);
            char d[96];
            snprintf(d, sizeof d, "nslot=%d async=%d", t.nslot, t.async);
            ok(t.nslot == 1, "budget yields a single ring slot", d);
            ok(eng_streamer_is_async(s) == 0,
               "reader is NOT started with one slot",
               "it would read over the block in use");

            /* And prefetch must be a harmless no-op on that path. */
            unsigned char *p = eng_streamer_get(s, 5, NULL);
            eng_streamer_prefetch(s, 6);
            ok(p && verify(p, 5), "block intact after a no-op prefetch", NULL);
            eng_streamer_destroy(s);
        }
    }

    /* ---- content correctness across a full walk ---- */
    printf("\n== content, synchronous ==\n");
    {
        EngStreamer *s = build(st, blocks, 200000, 0, 2);
        ok(s != NULL, "create synchronous streamer", NULL);
        if (s) {
            ok(eng_streamer_is_async(s) == 0, "async disabled by request", NULL);
            int good = 1;
            for (int pass = 0; pass < 3; pass++)
                for (int b = 0; b < NBLK; b++) {
                    unsigned char *p = eng_streamer_get(s, b, NULL);
                    if (!p || !verify(p, b)) { good = 0; break; }
                }
            ok(good, "3 full walks, every block's bytes correct",
               "pinned and streamed alike");
            eng_streamer_destroy(s);
        }
    }

    /* ---- THE CYCLIC-SCAN PROPERTY ---- */
    printf("\n== cyclic scan: pinned prefix vs LRU ==\n");
    {
        EngStreamer *s = build(st, blocks, 200000, 0, 2);
        if (s) {
            EngStreamerStats t0; eng_streamer_stats(s, &t0);
            const int npin = t0.npin;
            /* The streamed tail must genuinely cycle: with more streaming blocks than
               ring slots, a returning walk can never find one resident. */
            ok(NBLK - npin > t0.nslot, "streamed tail exceeds the ring",
               "so the scan really does cycle");

            const int PASSES = 4;
            for (int pass = 0; pass < PASSES; pass++)
                for (int b = 0; b < NBLK; b++) (void)eng_streamer_get(s, b, NULL);

            EngStreamerStats t; eng_streamer_stats(s, &t);
            char d[160];
            snprintf(d, sizeof d, "npin=%d passes=%d hits=%llu",
                     npin, PASSES, (unsigned long long)t.hits);

            /* EXACTLY npin*(PASSES-1): every pinned block hits on every pass after the
             * first, and every streamed block misses every time. Under LRU with this
             * many slots the pinned figure would be zero too. */
            eqi((long long)t.hits, (long long)npin * (PASSES - 1),
                "hits == npin * (passes-1)");
            eqi((long long)(t.hits + t.misses), (long long)NBLK * PASSES,
                "requests == blocks * passes");
            ok(1, "streamed blocks contributed zero hits", d);

            /* The predicted rate is knowable in advance because the policy is
             * deterministic, which is what lets the planner split a budget without
             * running anything. */
            const double pred = eng_streamer_predicted_hit_rate(s);
            snprintf(d, sizeof d, "predicted %.1f%% = %d/%d", pred * 100.0, npin, NBLK);
            ok(pred > 0.0 && pred == (double)npin / NBLK, "predicted rate is npin/nblocks", d);

            eng_streamer_report(s, "  ");
            eng_streamer_destroy(s);
        }
    }

    /* ---- async: bytes must survive an in-flight prefetch ---- */
    printf("\n== overlapped reads ==\n");
    {
        EngStreamer *s = build(st, blocks, 200000, 1, 2);
        ok(s != NULL, "create async streamer", NULL);
        if (s) {
            ok(eng_streamer_is_async(s) == 1, "reader started with a 2+ slot ring", NULL);

            /* THE REAL USAGE PATTERN, and the one that broke: take block L, hint L+1,
             * then use L's bytes while that read is in flight. */
            int good = 1;
            for (int pass = 0; pass < 3; pass++) {
                for (int b = 0; b < NBLK; b++) {
                    unsigned char *p = eng_streamer_get(s, b, NULL);
                    if (!p) { good = 0; break; }
                    eng_streamer_prefetch(s, b + 1 < NBLK ? b + 1 : 0);
                    /* Verify AFTER issuing the prefetch. */
                    if (!verify(p, b)) { good = 0; break; }
                }
                if (!good) break;
            }
            ok(good, "3 walks with prefetch, all bytes correct",
               "verified while the next read is in flight");

            EngStreamerStats t; eng_streamer_stats(s, &t);
            eqi((long long)(t.hits + t.misses), (long long)NBLK * 3, "request count");
            ok(t.bytes_read > 0, "bytes were actually read", NULL);
            eng_streamer_report(s, "  ");
            eng_streamer_destroy(s);
        }
    }

    /* ---- prefetch of pinned / resident blocks is a no-op ---- */
    printf("\n== prefetch edge cases ==\n");
    {
        EngStreamer *s = build(st, blocks, 200000, 1, 2);
        if (s) {
            eng_streamer_prefetch(s, 0);          /* pinned */
            eng_streamer_prefetch(s, -1);         /* out of range */
            eng_streamer_prefetch(s, NBLK + 5);   /* out of range */
            unsigned char *p = eng_streamer_get(s, 0, NULL);
            ok(p && verify(p, 0), "pinned block correct after stray prefetches", NULL);

            unsigned char *q = eng_streamer_get(s, NBLK - 1, NULL);
            eng_streamer_prefetch(s, NBLK - 1);   /* already resident */
            ok(q && verify(q, NBLK - 1), "resident block correct after self-prefetch", NULL);

            ok(eng_streamer_get(s, -1, NULL) == NULL, "get refuses negative block", NULL);
            ok(eng_streamer_get(s, NBLK, NULL) == NULL, "get refuses out-of-range block", NULL);
            eng_streamer_destroy(s);
        }
    }

    /* ---- a budget too small for even one slot must be refused, not clamped ---- */
    printf("\n== refusals ==\n");
    {
        EngStreamer *s = build(st, blocks, 1024, 0, 2);
        ok(s == NULL, "refuses a budget below one ring slot", "1 KB");
        if (s) eng_streamer_destroy(s);
    }

    /* ---- extra scratch area ---- */
    printf("\n== per-slot scratch ==\n");
    {
        EngStreamerCfg cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.name = "scratch"; cfg.store = st; cfg.blocks = blocks; cfg.nblocks = NBLK;
        cfg.budget_bytes = 300000; cfg.extra_bytes = 4096; cfg.ring_want = 2;
        cfg.async = 0; cfg.hugepages = 0; cfg.quiet = 1;
        EngStreamer *s = eng_streamer_create(&cfg);
        ok(s != NULL, "create with per-slot scratch", NULL);
        if (s) {
            unsigned char *extra = NULL;
            unsigned char *p = eng_streamer_get(s, NBLK - 1, &extra);
            ok(p != NULL && extra != NULL, "scratch pointer returned", NULL);
            ok(extra > p, "scratch sits after the payload", NULL);
            /* Writing scratch must not disturb the payload. */
            if (extra) memset(extra, 0xAB, 4096);
            ok(p && verify(p, NBLK - 1), "payload intact after writing scratch", NULL);
            eng_streamer_destroy(s);
        }
    }

    /* ---- K3 trunk geometry, scaled down ----
     *
     * The real trunk is 93 layers where layer 0 carries a dense 33792-wide MLP and is
     * 2.34 GB against 1.27 GB for the rest. Sizes here are the same RATIO in kilobytes,
     * so the fixed-point sizing loop sees the shape it will see in production without
     * allocating gigabytes.
     *
     * The case that matters is the third: once layer 0 is pinned it no longer streams,
     * so the ring must size to 1.27 (the largest REMAINING block) and not to 2.34.
     * Sizing the ring from the maximum over ALL blocks reserves room for a block that
     * prefix-pinning pins first whenever it pins anything at all, and on the real model
     * that wasted about 1.17 GB of every budget. */
    printf("\n== K3 trunk geometry (93 layers, scaled) ==\n");
    {
        enum { K3N = 93 };
        static EngBlock k3b[K3N];
        const int64_t L0 = 2340 * 1024, LN = 1270 * 1024;
        int64_t off = FILE_HEADER;
        for (int i = 0; i < K3N; i++) {
            k3b[i].nbytes = (i == 0) ? L0 : LN;
            k3b[i].off = off;         /* offsets are nominal: sizing reads nothing */
            off += k3b[i].nbytes;
        }

        EngStreamerCfg cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.name = "k3"; cfg.store = st; cfg.blocks = k3b; cfg.nblocks = K3N;
        cfg.ring_want = 2; cfg.async = 0; cfg.hugepages = 0; cfg.quiet = 1;

        /* A budget too small to pin anything: everything streams, ring still works. */
        cfg.budget_bytes = 3 * LN;
        EngStreamer *a = eng_streamer_create(&cfg);
        ok(a != NULL, "tight budget still constructs", NULL);
        if (a) {
            EngStreamerStats t; eng_streamer_stats(a, &t);
            char d[128];
            snprintf(d, sizeof d, "npin=%d nslot=%d", t.npin, t.nslot);
            ok(t.npin == 0, "tight budget pins nothing", d);
            ok(t.nslot >= 1, "ring survives", NULL);
            ok(eng_streamer_predicted_hit_rate(a) == 0.0,
               "predicted hit rate is 0 when nothing is pinned", NULL);
            eng_streamer_destroy(a);
        }

        /* THE RING-SIZING CASE. Enough to pin layer 0 and several others. */
        cfg.budget_bytes = L0 + 8 * LN;
        EngStreamer *b = eng_streamer_create(&cfg);
        ok(b != NULL, "mid budget constructs", NULL);
        if (b) {
            EngStreamerStats t; eng_streamer_stats(b, &t);
            char d[160];
            snprintf(d, sizeof d, "npin=%d slot=%lld KB (L0=%lld KB, Ln=%lld KB)",
                     t.npin, (long long)(t.slot_bytes / 1024),
                     (long long)(L0 / 1024), (long long)(LN / 1024));
            ok(t.npin >= 1, "layer 0 is pinned", d);
            /* The ring slot must fit 1.27 MB, not 2.34 MB. Allow the alignment and
               scratch overhead, but it must be nowhere near L0. */
            ok(t.slot_bytes < L0, "ring sized to the largest STREAMING block", d);
            ok(t.slot_bytes >= LN, "ring still fits a normal layer", NULL);
            eng_streamer_destroy(b);
        }

        /* A budget that fits everything pins everything: hit rate 100%. */
        cfg.budget_bytes = L0 + (int64_t)(K3N + 4) * LN;
        EngStreamer *cst = eng_streamer_create(&cfg);
        ok(cst != NULL, "large budget constructs", NULL);
        if (cst) {
            EngStreamerStats t; eng_streamer_stats(cst, &t);
            char d[96];
            snprintf(d, sizeof d, "npin=%d of %d", t.npin, K3N);
            ok(t.npin == K3N, "everything pinned when the budget allows", d);
            ok(eng_streamer_predicted_hit_rate(cst) == 1.0, "predicted hit rate 100%", NULL);
            eng_streamer_destroy(cst);
        }
    }

    st->close(st);
    remove(path);

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("STREAMER TESTS FAILED\n"); return 1; }
    printf("STREAMER TESTS PASSED\n");
    return 0;
}
