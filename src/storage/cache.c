/* SPDX-License-Identifier: Apache-2.0 */
/* cache.c - see cache.h. Extracted from src/cache/k3_cache.c. */
#define _POSIX_C_SOURCE 200809L

#include "cache.h"
#include "storage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__) || defined(__APPLE__)
#  include <sys/mman.h>
#endif

/* Slot states. EMPTY must stay -1 so a memset-to-0xff array reads as empty and several
 * "< 0 means holds nothing" tests keep working. INFLIGHT is DISTINCT so the victim
 * search can refuse a slot whose read has not landed -- see cache.h, fix 1. */
#define SLOT_EMPTY     (-1)
#define SLOT_INFLIGHT  (-2)

/* Bounded by the largest batch any caller issues. K3's top-16 is the motivating case;
 * 64 matches K3_MAX_TOPK so a config that passes K3's validation fits here too. */
#define ENG_CACHE_MAX_BATCH 64

struct EngCache {
    EngCacheCfg cfg;
    EngCacheSrc src;

    unsigned char *arena;
    int64_t  stride;          /* per-slot bytes INCLUDING alignment room */
    int64_t  arena_bytes;
    int      nslot;

    int32_t  *slot_of;        /* [nkeys] -> slot, or -1        */
    int64_t  *key_of;         /* [nslot] -> key, EMPTY, INFLIGHT */
    uint64_t *used_at;        /* [nslot] LRU stamp             */
    unsigned char *pinned;    /* [nslot]                       */
    unsigned char *meta;      /* [nslot * meta_bytes]          */
    int32_t  *pad;            /* [nslot] payload offset in slot */
    int64_t  *nbytes;         /* [nslot] payload size          */

    uint64_t clock;
    uint64_t hits, misses, evictions, bytes_read, prefetch_reads, requests;
    double   load_seconds;

    uint32_t *hist;           /* [nkeys] or NULL */
    EngCacheKey *trace;
    int64_t   ntrace, captrace;
};

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* THE SLOT STRIDE, not just the arena base, must be alignment-rounded. See cache.h
 * fix 3: aligning the arena aligns slot 0 and nothing else, because slot N starts at
 * arena + N*stride. */
int64_t eng_cache_stride_for(int64_t payload_bytes)
{
    if (payload_bytes < 0) return 0;
    const int64_t s = payload_bytes + 2 * ENG_IO_ALIGN;
    return (s + ENG_IO_ALIGN - 1) & ~(int64_t)(ENG_IO_ALIGN - 1);
}

static void *slot_meta(EngCache *c, int slot)
{
    return c->cfg.meta_bytes ? (void *)(c->meta + (size_t)slot * c->cfg.meta_bytes) : NULL;
}

static unsigned char *slot_base(EngCache *c, int slot)
{
    return c->arena + (size_t)slot * c->stride;
}

/* Least recently used unpinned slot. Linear by choice: a few hundred comparisons
 * against a multi-megabyte read is not where the time goes, and a heap would have to be
 * kept consistent through pinning and inflight states for no measurable gain. */
static int pick_victim(EngCache *c)
{
    int best = -1;
    uint64_t oldest = (uint64_t)-1;
    for (int i = 0; i < c->nslot; i++) {
        if (c->key_of[i] == SLOT_INFLIGHT) continue;   /* being read into RIGHT NOW */
        if (c->key_of[i] == SLOT_EMPTY) return i;      /* free, take it */
        if (c->pinned[i]) continue;
        if (c->used_at[i] < oldest) { oldest = c->used_at[i]; best = i; }
    }
    return best;
}

static void entry_of(EngCache *c, int slot, EngCacheEntry *out)
{
    if (!out) return;
    out->data   = slot_base(c, slot) + c->pad[slot];
    out->nbytes = c->nbytes[slot];
    out->meta   = slot_meta(c, slot);
}

/* Bring a key resident and return its slot, or -1. */
static int admit(EngCache *c, EngCacheKey key)
{
    int slot = c->slot_of[key];
    if (slot >= 0) {
        c->hits++;
        c->used_at[slot] = ++c->clock;
        return slot;
    }
    c->misses++;

    /* Describe into a scratch meta first: the victim's meta must not be clobbered
     * before we know this key is admissible. */
    unsigned char tmp[256];
    unsigned char *m = NULL;
    unsigned char *heap = NULL;
    if (c->cfg.meta_bytes) {
        if (c->cfg.meta_bytes <= sizeof tmp) m = tmp;
        else { heap = (unsigned char *)malloc(c->cfg.meta_bytes); m = heap; }
        if (!m) return -1;
        memset(m, 0, c->cfg.meta_bytes);
    }

    int64_t need = 0;
    if (c->src.describe(c->src.ctx, key, &need, m) != 0) { free(heap); return -1; }
    if (need > c->cfg.slot_bytes) {
        fprintf(stderr, "%s cache: key %llu is %lld bytes, slot payload is %lld\n",
                c->cfg.name, (unsigned long long)key,
                (long long)need, (long long)c->cfg.slot_bytes);
        free(heap);
        return -1;
    }

    slot = pick_victim(c);
    if (slot < 0) {
        fprintf(stderr, "%s cache: every slot is pinned, cannot admit key %llu\n",
                c->cfg.name, (unsigned long long)key);
        free(heap);
        return -1;
    }
    if (c->key_of[slot] >= 0) { c->slot_of[c->key_of[slot]] = -1; c->evictions++; }

    if (c->cfg.meta_bytes) memcpy(slot_meta(c, slot), m, c->cfg.meta_bytes);
    free(heap);

    const double t0 = now_s();
    int64_t pad = 0;
    const int64_t got = c->src.load(c->src.ctx, key, slot_meta(c, slot),
                                    slot_base(c, slot), c->stride, &pad);
    c->load_seconds += now_s() - t0;

    if (got != need) {
        fprintf(stderr, "%s cache: short load of key %llu (%lld of %lld)\n",
                c->cfg.name, (unsigned long long)key, (long long)got, (long long)need);
        /* Leave the slot EMPTY rather than owning a key it does not hold: fix 2. */
        c->key_of[slot] = SLOT_EMPTY;
        return -1;
    }
    c->bytes_read += (uint64_t)got;

    c->pad[slot]    = (int32_t)pad;
    c->nbytes[slot] = got;
    c->key_of[slot] = (int64_t)key;
    c->slot_of[key] = slot;
    c->used_at[slot] = ++c->clock;
    return slot;
}

int eng_cache_get(EngCache *c, EngCacheKey key, EngCacheEntry *out)
{
    if (!c || key >= (EngCacheKey)c->cfg.nkeys) {
        if (c) fprintf(stderr, "%s cache: key %llu out of range (nkeys %lld)\n",
                       c->cfg.name, (unsigned long long)key, (long long)c->cfg.nkeys);
        return -1;
    }
    c->requests++;
    if (c->hist) c->hist[key]++;

    /* Record BEFORE serving. The trace must reflect what the caller asked for,
     * independent of what the cache held, or replaying it at another capacity is
     * meaningless. */
    if (c->cfg.want_trace) {
        if (c->ntrace + 1 > c->captrace) {
            const int64_t nc = c->captrace ? c->captrace * 2 : (1 << 15);
            EngCacheKey *nt = (EngCacheKey *)realloc(c->trace, (size_t)nc * sizeof *nt);
            if (nt) { c->trace = nt; c->captrace = nc; }
        }
        if (c->ntrace + 1 <= c->captrace) c->trace[c->ntrace++] = key;
    }

    const int slot = admit(c, key);
    if (slot < 0) return -1;
    entry_of(c, slot, out);
    return 0;
}

int eng_cache_peek(EngCache *c, EngCacheKey key, EngCacheEntry *out)
{
    if (!c || key >= (EngCacheKey)c->cfg.nkeys) return 0;
    const int slot = c->slot_of[key];
    if (slot < 0) return 0;
    entry_of(c, slot, out);
    return 1;
}

/* Batch admit, reads issued CONCURRENTLY.
 *
 * The serial path admits one object per call, so the device sees a queue depth of one:
 * read, wait, repeat. NVMe needs depth to reach its rated bandwidth, so that pattern
 * leaves most of the drive idle. This hands the whole set over at once.
 *
 * THREE PHASES, and the split is not cosmetic:
 *   1 SERIAL   resolve each miss and reserve it a slot. Slot allocation touches LRU
 *              bookkeeping, which is shared mutable state and must not race.
 *   2 PARALLEL do the reads. Each targets a distinct, already-assigned buffer.
 *   3 SERIAL   publish, and only what actually arrived. See cache.h fixes 1 and 2.
 */
int eng_cache_get_many(EngCache *c, const EngCacheKey *keys, int n)
{
    if (!c || !keys || n <= 0) return 0;

    typedef struct {
        int slot; EngCacheKey key; int64_t need, got, pad;
        int device; int64_t offset;
    } Work;
    Work w[ENG_CACHE_MAX_BATCH];
    int nw = 0;
    const int cap = (int)(sizeof w / sizeof *w);

    /* ---- phase 1: reserve, serially ---- */
    for (int i = 0; i < n && nw < cap; i++) {
        const EngCacheKey k = keys[i];
        if (k >= (EngCacheKey)c->cfg.nkeys) continue;
        if (c->slot_of[k] >= 0) continue;                 /* already resident */

        int dup = 0;                                      /* same key twice in one batch */
        for (int j = 0; j < nw; j++) if (w[j].key == k) { dup = 1; break; }
        if (dup) continue;

        const int slot0 = pick_victim(c);
        if (slot0 < 0) break;

        /* Describe into the victim's meta only after the victim is chosen, but BEFORE
         * evicting the old key, so a refusal costs nothing. */
        int64_t need = 0;
        if (c->src.describe(c->src.ctx, k, &need, slot_meta(c, slot0)) != 0) continue;
        if (need > c->cfg.slot_bytes) continue;

        if (c->key_of[slot0] >= 0) { c->slot_of[c->key_of[slot0]] = -1; c->evictions++; }
        /* INFLIGHT, not EMPTY: marking it empty makes pick_victim's fast path hand the
         * same slot to the next key in this very batch. */
        c->key_of[slot0]  = SLOT_INFLIGHT;
        c->used_at[slot0] = ++c->clock;

        w[nw].slot = slot0; w[nw].key = k; w[nw].need = need;
        w[nw].got = -1; w[nw].pad = 0; w[nw].device = 0; w[nw].offset = 0;
        if (c->src.locate)
            c->src.locate(c->src.ctx, slot_meta(c, slot0), &w[nw].device, &w[nw].offset);
        nw++;
    }
    if (nw == 0) return 0;

    /* Issue in STORAGE order. Objects are not laid out key-ordered, so sorting by where
     * the bytes live turns a scattered set of seeks into a mostly forward sweep.
     * Insertion sort: nw is at most the batch size. */
    if (c->src.locate) {
        for (int i = 1; i < nw; i++) {
            Work t = w[i];
            int j = i - 1;
            while (j >= 0 && (w[j].device > t.device ||
                             (w[j].device == t.device && w[j].offset > t.offset))) {
                w[j + 1] = w[j]; j--;
            }
            w[j + 1] = t;
        }
    }

    /* ---- phase 2: read, concurrently ---- */
    const double t0 = now_s();
#ifdef _OPENMP
#   pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int i = 0; i < nw; i++) {
        int64_t pad = 0;
        const int64_t got = c->src.load(c->src.ctx, w[i].key, slot_meta(c, w[i].slot),
                                        slot_base(c, w[i].slot), c->stride, &pad);
        w[i].got = got;
        w[i].pad = pad;
    }
    c->load_seconds += now_s() - t0;

    /* ---- phase 3: publish only what actually arrived ---- */
    int ok = 0;
    for (int i = 0; i < nw; i++) {
        if (w[i].got != w[i].need) {
            fprintf(stderr, "%s cache: short prefetch of key %llu (%lld of %lld); "
                            "leaving the slot empty so it cannot be served as a hit\n",
                    c->cfg.name, (unsigned long long)w[i].key,
                    (long long)w[i].got, (long long)w[i].need);
            c->key_of[w[i].slot] = SLOT_EMPTY;
            continue;
        }
        c->pad[w[i].slot]     = (int32_t)w[i].pad;
        c->nbytes[w[i].slot]  = w[i].got;
        c->key_of[w[i].slot]  = (int64_t)w[i].key;
        c->slot_of[w[i].key]  = w[i].slot;
        c->used_at[w[i].slot] = ++c->clock;
        c->bytes_read    += (uint64_t)w[i].got;
        c->prefetch_reads++;
        ok++;
    }
    return ok;
}

int eng_cache_pin(EngCache *c, EngCacheKey key, int pin)
{
    if (!c || key >= (EngCacheKey)c->cfg.nkeys) return 0;
    const int slot = c->slot_of[key];
    if (slot < 0) return 0;
    c->pinned[slot] = pin ? 1 : 0;
    return 1;
}

/* ------------------------------------------------------------- construction -- */

EngCache *eng_cache_create(const EngCacheCfg *cfg, const EngCacheSrc *src)
{
    if (!cfg || !src || !src->describe || !src->load) return NULL;
    if (cfg->nkeys <= 0 || cfg->slot_bytes <= 0) return NULL;

    EngCache *c = (EngCache *)calloc(1, sizeof *c);
    if (!c) return NULL;
    c->cfg = *cfg;
    c->src = *src;
    if (!c->cfg.name) c->cfg.name = "weight";

    c->stride = eng_cache_stride_for(cfg->slot_bytes);
    c->nslot = (int)(cfg->budget_bytes / c->stride);
    if (c->nslot < cfg->min_slots || c->nslot < 1) {
        fprintf(stderr,
                "%s cache: budget %.2f GB gives %d slots of %.2f MB, but at least %d are "
                "needed. A cache smaller than one step's working set would evict an "
                "object that is still in use.\n",
                c->cfg.name, (double)cfg->budget_bytes / 1e9, c->nslot,
                (double)c->stride / 1e6, cfg->min_slots);
        free(c);
        return NULL;
    }

    /* 2 MB aligned and hugepage-advised: every direct read pins its destination pages,
     * and a 17.55 MB slot on 4 KB pages is 4,284 pins per read. ENG_NOHUGE=1 restores
     * 4 KB so the two can be compared on ONE binary rather than two builds. */
    {
        const int huge = cfg->hugepages < 0 ? (getenv("ENG_NOHUGE") ? 0 : 1)
                                            : (cfg->hugepages ? 1 : 0);
        const size_t al = huge ? (2u << 20) : (size_t)ENG_IO_ALIGN;
        size_t want = (size_t)c->nslot * (size_t)c->stride;
        if (huge) want = (want + al - 1) & ~(al - 1);
        if (posix_memalign((void **)&c->arena, al, want) != 0) {
            fprintf(stderr, "%s cache: cannot allocate %.2f GB arena\n",
                    c->cfg.name, (double)want / 1e9);
            free(c);
            return NULL;
        }
#if defined(MADV_HUGEPAGE)
        if (huge) madvise(c->arena, want, MADV_HUGEPAGE);
#endif
        c->arena_bytes = (int64_t)want;
    }

    c->slot_of = (int32_t  *)malloc((size_t)cfg->nkeys * sizeof *c->slot_of);
    c->key_of  = (int64_t  *)malloc((size_t)c->nslot   * sizeof *c->key_of);
    c->used_at = (uint64_t *)calloc((size_t)c->nslot, sizeof *c->used_at);
    c->pinned  = (unsigned char *)calloc((size_t)c->nslot, 1);
    c->pad     = (int32_t  *)calloc((size_t)c->nslot, sizeof *c->pad);
    c->nbytes  = (int64_t  *)calloc((size_t)c->nslot, sizeof *c->nbytes);
    c->meta    = cfg->meta_bytes
               ? (unsigned char *)calloc((size_t)c->nslot, cfg->meta_bytes) : NULL;
    c->hist    = cfg->want_hist
               ? (uint32_t *)calloc((size_t)cfg->nkeys, sizeof *c->hist) : NULL;

    if (!c->slot_of || !c->key_of || !c->used_at || !c->pinned || !c->pad || !c->nbytes ||
        (cfg->meta_bytes && !c->meta) || (cfg->want_hist && !c->hist)) {
        eng_cache_destroy(c);
        return NULL;
    }

    for (int64_t i = 0; i < cfg->nkeys; i++) c->slot_of[i] = -1;
    for (int i = 0; i < c->nslot; i++)       c->key_of[i]  = SLOT_EMPTY;
    return c;
}

void eng_cache_destroy(EngCache *c)
{
    if (!c) return;
    free(c->arena); free(c->slot_of); free(c->key_of); free(c->used_at);
    free(c->pinned); free(c->pad); free(c->nbytes); free(c->meta);
    free(c->hist); free(c->trace);
    free(c);
}

/* --------------------------------------------------------------- statistics -- */

void eng_cache_stats(const EngCache *c, EngCacheStats *out)
{
    if (!c || !out) return;
    out->hits           = c->hits;
    out->misses         = c->misses;
    out->evictions      = c->evictions;
    out->bytes_read     = c->bytes_read;
    out->prefetch_reads = c->prefetch_reads;
    out->requests       = c->requests;
    out->load_seconds   = c->load_seconds;
    out->nslot          = c->nslot;
    out->slot_stride    = c->stride;
    out->arena_bytes    = c->arena_bytes;
}

void eng_cache_reset_stats(EngCache *c)
{
    if (!c) return;
    c->hits = c->misses = c->evictions = 0;
    c->bytes_read = c->prefetch_reads = c->requests = 0;
    c->load_seconds = 0.0;
}

int eng_cache_slots(const EngCache *c) { return c ? c->nslot : 0; }

const uint32_t *eng_cache_hist(const EngCache *c, int64_t *nkeys)
{
    if (!c) return NULL;
    if (nkeys) *nkeys = c->cfg.nkeys;
    return c->hist;
}

const EngCacheKey *eng_cache_trace(const EngCache *c, int64_t *n)
{
    if (!c) return NULL;
    if (n) *n = c->ntrace;
    return c->trace;
}

void eng_cache_report(const EngCache *c, const char *label)
{
    if (!c) return;
    const uint64_t req = c->hits + c->misses;
    const double hitpct = req ? 100.0 * (double)c->hits / (double)req : 0.0;
    /* Effective hit rate discounts prefetched objects: they read as hits but their
     * bytes still moved during this step. See cache.h fix 4. */
    const double effpct = req
        ? 100.0 * (double)(c->hits > c->prefetch_reads ? c->hits - c->prefetch_reads : 0)
              / (double)req
        : 0.0;

    printf("%s%s%s cache: %d slots x %.2f MB (%.2f GB arena)\n",
           label ? label : "", label ? " " : "", c->cfg.name,
           c->nslot, (double)c->stride / 1e6, (double)c->arena_bytes / 1e9);
    printf("  requests %llu  hits %llu (%.1f%%, effective %.1f%%)  misses %llu  evictions %llu\n",
           (unsigned long long)req, (unsigned long long)c->hits, hitpct, effpct,
           (unsigned long long)c->misses, (unsigned long long)c->evictions);
    printf("  read %.2f GB in %.2f s", (double)c->bytes_read / 1e9, c->load_seconds);
    if (c->load_seconds > 0.0)
        printf(" (%.0f MB/s)", (double)c->bytes_read / 1e6 / c->load_seconds);
    printf(", %llu via prefetch\n", (unsigned long long)c->prefetch_reads);
}
