/* k3_cache.c - see k3_cache.h. The policy lives in src/storage/cache.c; this is the
 * K3 binding: key encoding, expert geometry as slot metadata, and the K3ExpertSrc
 * vtable the MoE kernels already consume. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_cache.h"

/* key = layer * n_experts + expert. The same encoding the old direct-indexed slot_of[]
 * and hist[] used, so dumps and tools/sim_cache.py see identical numbering. */
static EngCacheKey k3key(const K3Cache *c, int layer, int expert)
{
    return (EngCacheKey)((int64_t)layer * c->n_experts + expert);
}

/* Copy the generic cache's counters into the mirrored fields. Called after every
 * operation that can change them, so callers reading c->hits directly (k3_run.c, and
 * tests/unit/test_cache.c) see what they always did. */
static void sync_stats(K3Cache *c)
{
    EngCacheStats s;
    eng_cache_stats(c->gc, &s);
    c->hits           = s.hits;
    c->misses         = s.misses;
    c->evictions      = s.evictions;
    c->bytes_read     = s.bytes_read;
    c->prefetch_reads = s.prefetch_reads;
    c->load_seconds   = s.load_seconds;
    c->nslot          = s.nslot;
    c->slot_bytes     = s.slot_stride;
}

/* Resolve a cache entry to the three (packed, scale) pairs the kernels want. The base
 * pointer is already past any O_DIRECT padding: the generic cache applies it. */
static void fill_q(const EngCacheEntry *e, K3ExpertQ *q)
{
    const unsigned char *b = e->data;
    const K3ExpertRef *r = (const K3ExpertRef *)e->meta;
    q->p1 = b + r->m[0].p_off; q->s1 = b + r->m[0].s_off;
    q->p2 = b + r->m[1].p_off; q->s2 = b + r->m[1].s_off;
    q->p3 = b + r->m[2].p_off; q->s3 = b + r->m[2].s_off;
}

/* ------------------------------------------------- the generic cache's source -- */

static int k3_describe(void *ctx, EngCacheKey key, int64_t *nbytes, void *meta)
{
    K3Cache *c = (K3Cache *)ctx;
    const int layer  = (int)(key / (EngCacheKey)c->n_experts);
    const int expert = (int)(key % (EngCacheKey)c->n_experts);
    K3ExpertRef *r = (K3ExpertRef *)meta;
    if (k3_expert_ref(c->st, layer, expert, r) != 0) return -1;
    *nbytes = r->nbytes;
    return 0;
}

static int64_t k3_load(void *ctx, EngCacheKey key, const void *meta,
                       unsigned char *buf, int64_t bufcap, int64_t *payload_off)
{
    K3Cache *c = (K3Cache *)ctx;
    (void)key;
    /* Safe under the batch path's parallel loop: pread takes its offset as an argument
     * and touches no shared file position, and each call has its own buffer. */
    return k3_expert_load_direct(c->st, (const K3ExpertRef *)meta,
                                 buf, bufcap, payload_off);
}

/* Experts are not stored id-ordered inside a shard, so handing the batch a
 * (shard, offset) pair turns a scattered set of seeks into a mostly forward sweep. */
static void k3_locate(void *ctx, const void *meta, int *device, int64_t *offset)
{
    (void)ctx;
    const K3ExpertRef *r = (const K3ExpertRef *)meta;
    *device = r->shard;
    *offset = r->off;
}

/* ------------------------------------------------------- K3ExpertSrc vtable -- */

static int cache_get(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Cache *c = (K3Cache *)self;          /* src is the first member, by contract */
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts) {
        fprintf(stderr, "k3_cache: out of range L%d expert %d\n", layer, expert);
        return -1;
    }
    EngCacheEntry e;
    const int rc = eng_cache_get(c->gc, k3key(c, layer, expert), &e);
    sync_stats(c);
    if (rc != 0) return -1;
    fill_q(&e, out);
    return 0;
}

static int cache_getmany(K3ExpertSrc *self, int layer, const int *ids, int n)
{
    K3Cache *c = (K3Cache *)self;
    if (n <= 0) return 0;
    if (n > K3_MAX_TOPK) n = K3_MAX_TOPK;

    EngCacheKey keys[K3_MAX_TOPK];
    int nk = 0;
    for (int i = 0; i < n; i++) {
        const int e = ids[i];
        if (e < 0 || e >= c->n_experts) continue;
        keys[nk++] = k3key(c, layer, e);
    }
    const int got = eng_cache_get_many(c->gc, keys, nk);
    sync_stats(c);
    return got;
}

/* Is this expert already resident, i.e. would get() serve it with no disk read? The
 * draft model's cache-only routing uses this to propose tokens with zero expert I/O. */
static int cache_resident(K3ExpertSrc *self, int layer, int expert, K3ExpertQ *out)
{
    K3Cache *c = (K3Cache *)self;
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts)
        return 0;
    EngCacheEntry e;
    if (!eng_cache_peek(c->gc, k3key(c, layer, expert), &e)) return 0;
    if (out) fill_q(&e, out);
    return 1;
}

/* ---------------------------------------------------------------- lifecycle -- */

int k3_cache_init(K3Cache *c, const K3St *st, const K3Cfg *cfg, int64_t budget_bytes)
{
    memset(c, 0, sizeof *c);
    c->src.get      = cache_get;
    c->src.resident = cache_resident;
    /* K3_NOPREFETCH=1 disables the batch path at runtime. An A/B between two BUILDS
     * compares two binaries; an A/B on one binary compares one decision, which is the
     * only way to attribute a timing difference to the prefetch rather than to the
     * compiler, the layout, or the weather. */
    c->src.getmany  = getenv("K3_NOPREFETCH") ? NULL : cache_getmany;
    if (!c->src.getmany)
        fprintf(stderr, "k3_cache: batch prefetch DISABLED by K3_NOPREFETCH\n");
    c->src.ctx   = c;
    c->st        = st;
    c->n_layers  = cfg->n_layers;
    c->n_experts = cfg->n_experts;

    /* Size a slot from the checkpoint rather than from arithmetic: find any expert and
     * ask how many bytes it actually occupies. The generic cache adds the alignment
     * room and rounds the STRIDE, which is the part that must not be done by hand. */
    K3ExpertRef probe;
    int found = 0;
    for (int L = 0; L < cfg->n_layers && !found; L++) {
        if (k3_is_dense(cfg, L)) continue;
        if (k3_expert_ref(st, L, 0, &probe) == 0) found = 1;
    }
    if (!found) {
        fprintf(stderr, "k3_cache: no routed experts in this shard set\n");
        return -1;
    }

    EngCacheCfg cc;
    memset(&cc, 0, sizeof cc);
    cc.name         = "expert";
    cc.budget_bytes = budget_bytes;
    cc.slot_bytes   = probe.nbytes;
    cc.nkeys        = (int64_t)cfg->n_layers * cfg->n_experts;
    cc.min_slots    = cfg->topk + 1;
    cc.meta_bytes   = sizeof(K3ExpertRef);
    cc.want_hist    = 1;
    cc.want_trace   = 1;
    cc.hugepages    = -1;         /* auto: on unless ENG_NOHUGE */

    EngCacheSrc cs;
    memset(&cs, 0, sizeof cs);
    cs.describe = k3_describe;
    cs.load     = k3_load;
    cs.locate   = k3_locate;
    cs.ctx      = c;

    c->gc = eng_cache_create(&cc, &cs);
    if (!c->gc) return -1;
    sync_stats(c);
    return 0;
}

void k3_cache_free(K3Cache *c)
{
    if (!c) return;
    eng_cache_destroy(c->gc);
    memset(c, 0, sizeof *c);
}

int k3_cache_pin(K3Cache *c, int layer, int expert, int pin)
{
    if (layer < 0 || layer >= c->n_layers || expert < 0 || expert >= c->n_experts)
        return 0;
    return eng_cache_pin(c->gc, k3key(c, layer, expert), pin);
}

int k3_cache_prefetch(K3Cache *c, int layer, int expert)
{
    EngCacheEntry e;
    const int rc = eng_cache_get(c->gc, k3key(c, layer, expert), &e);
    sync_stats(c);
    return rc;
}

void k3_cache_reset_stats(K3Cache *c)
{
    eng_cache_reset_stats(c->gc);
    sync_stats(c);
}

void k3_cache_report(const K3Cache *c, const char *label)
{
    eng_cache_report(c->gc, label);
}

/* ------------------------------------------------------------------- dumps -- */

int k3_cache_dump_hist(const K3Cache *c, const char *path)
{
    int64_t nkeys = 0;
    const uint32_t *h = eng_cache_hist(c->gc, &nkeys);
    if (!h) return -1;

    FILE *f = fopen(path, "w");
    if (!f) return -1;

    fprintf(f, "{\n  \"n_layers\": %d,\n  \"n_experts\": %d,\n  \"counts\": {\n",
            c->n_layers, c->n_experts);
    int first = 1;
    for (int L = 0; L < c->n_layers; L++) {
        for (int e = 0; e < c->n_experts; e++) {
            const int64_t k = (int64_t)L * c->n_experts + e;
            if (k >= nkeys) break;
            const uint32_t v = h[k];
            if (!v) continue;
            fprintf(f, "%s    \"%d,%d\": %u", first ? "" : ",\n", L, e, v);
            first = 0;
        }
    }
    fprintf(f, "\n  }\n}\n");
    fclose(f);
    printf("wrote %s\n", path);
    return 0;
}

int k3_cache_dump_trace(const K3Cache *c, const char *path)
{
    int64_t n = 0;
    const EngCacheKey *tr = eng_cache_trace(c->gc, &n);
    if (!tr || n == 0) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    /* Decode flat keys back to the (layer, expert) int32 pairs tools/sim_cache.py
     * reads. Buffered rather than written one pair at a time: this is up to a few
     * hundred thousand requests on a real run. */
    int32_t buf[1024];
    int nb = 0;
    size_t written = 0;
    for (int64_t i = 0; i < n; i++) {
        buf[nb++] = (int32_t)(tr[i] / (EngCacheKey)c->n_experts);
        buf[nb++] = (int32_t)(tr[i] % (EngCacheKey)c->n_experts);
        if (nb >= (int)(sizeof buf / sizeof *buf)) {
            written += fwrite(buf, sizeof *buf, (size_t)nb, f);
            nb = 0;
        }
    }
    if (nb) written += fwrite(buf, sizeof *buf, (size_t)nb, f);
    fclose(f);

    printf("wrote %s: %lld requests (%.1f KB)\n",
           path, (long long)n, (double)n * 8 / 1024.0);
    return written == (size_t)(n * 2) ? 0 : -1;
}
