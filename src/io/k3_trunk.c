/* k3_trunk.c - see k3_trunk.h. Residency is the generic streamer's job; what is left
 * here is the trunk.json format and binding its tensors by name. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k3_trunk.h"
#include "k3_st.h"
#include "json.h"

/* WHERE THE TIME IN A BIND ACTUALLY GOES.
 *
 * The streamer's load_seconds brackets ONLY the read, so it reports a DEVICE rate.
 * Everything else a bind does -- widening bf16 vectors to fp32, resolving names, kernel
 * page bookkeeping -- is invisible to it while still being paid on every layer of every
 * token, and that residual is large enough to change conclusions drawn from the device
 * rate alone.
 *
 * These three close the gap by measurement rather than estimate: wall clock around the
 * whole of k3_trunk_bind, of which the widen loop is tracked separately, so
 * bind_wall - load_seconds - widen_wall is the remaining unattributed time. */
double k3_trunk_bind_wall = 0.0;
double k3_trunk_widen_wall = 0.0;
long   k3_trunk_binds = 0;

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int dt_of(const char *s)
{
    if (!strcmp(s, "BF16")) return K3_DT_BF16;
    if (!strcmp(s, "F32"))  return K3_DT_F32;
    if (!strcmp(s, "U8"))   return K3_DT_U8;
    if (!strcmp(s, "F16"))  return K3_DT_F16;
    if (!strcmp(s, "I8R"))  return K3_DT_I8R;
    return K3_DT_UNKNOWN;
}

static char *slurp(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)sz + 1);
    if (!b) { fclose(f); return NULL; }
    const size_t got = fread(b, 1, (size_t)sz, f);
    fclose(f);
    b[got] = '\0';
    if (n) *n = got;
    return b;
}

typedef struct { const K3TrunkLayer *L; } Finder;

static int find_in_layer(void *ctx, const char *name,
                         int64_t *off, int64_t *nbytes, int *dtype)
{
    const K3TrunkLayer *L = ((Finder *)ctx)->L;
    for (int i = 0; i < L->nt; i++)
        if (!strcmp(L->t[i].name, name)) {
            *off = L->t[i].off; *nbytes = L->t[i].nbytes; *dtype = L->t[i].dtype;
            return 0;
        }
    return -1;
}

static void sync_stats(K3Trunk *tr)
{
    EngStreamerStats s;
    eng_streamer_stats(tr->sm, &s);
    tr->hits         = s.hits;
    tr->misses       = s.misses;
    tr->bytes_read   = s.bytes_read;
    tr->load_seconds = s.load_seconds;
}

int k3_trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget_bytes)
{
    memset(tr, 0, sizeof *tr);

    char p[1024];
    snprintf(p, sizeof p, "%s/trunk.json", dir);
    size_t jn = 0;
    char *txt = slurp(p, &jn);
    if (!txt) { fprintf(stderr, "k3_trunk: cannot read %s\n", p); return -1; }

    /* The parser arena backs every K3TrunkTensor.name, so it must outlive the struct.
     * Owned here and freed in k3_trunk_close. */
    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    tr->json_arena = arena;
    if (!root) {
        fprintf(stderr, "k3_trunk: %s is not valid JSON\n", p);
        free(txt);
        return -1;
    }

    jval *jl = json_get(root, "layers");
    if (!jl || jl->t != J_ARR) {
        fprintf(stderr, "k3_trunk: no layers array\n");
        free(txt);
        return -1;
    }
    tr->n_layers = jl->len;
    tr->lay = (K3TrunkLayer *)calloc((size_t)tr->n_layers, sizeof(K3TrunkLayer));
    if (!tr->lay) { free(txt); return -1; }

    for (int i = 0; i < jl->len; i++) {
        jval *e = jl->kids[i];
        jval *v;
        K3TrunkLayer *L = &tr->lay[i];
        if ((v = json_get(e, "file_off")) && v->t == J_NUM) L->file_off = (int64_t)v->num;
        if ((v = json_get(e, "nbytes"))   && v->t == J_NUM) L->nbytes   = (int64_t)v->num;
        jval *ts = json_get(e, "tensors");
        if (!ts || ts->t != J_OBJ) {
            fprintf(stderr, "k3_trunk: layer %d has no tensors\n", i);
            free(txt);
            return -1;
        }
        L->nt = ts->len;
        L->t = (K3TrunkTensor *)calloc((size_t)L->nt, sizeof(K3TrunkTensor));
        if (!L->t) { free(txt); return -1; }
        for (int k = 0; k < ts->len; k++) {
            K3TrunkTensor *t = &L->t[k];
            t->name = ts->keys[k];      /* lives in the parser arena */
            jval *o = ts->kids[k];
            if ((v = json_get(o, "off"))    && v->t == J_NUM) t->off    = (int64_t)v->num;
            if ((v = json_get(o, "nbytes")) && v->t == J_NUM) t->nbytes = (int64_t)v->num;
            if ((v = json_get(o, "dtype"))  && v->t == J_STR) t->dtype  = dt_of(v->str);
        }
    }

    /* A trunk packed before the alignment convention cannot be read with O_DIRECT: its
     * run offsets are arbitrary. Ask for buffered reads in that case rather than
     * discovering it one failed read at a time. */
    int want_direct = 1;
    {
        jval *a = json_get(root, "align");
        const int64_t got = (a && a->t == J_NUM) ? (int64_t)a->num : 0;
        if (got != K3_TRUNK_ALIGN) {
            fprintf(stderr, "k3_trunk: trunk.json reports align %lld, expected %d; "
                            "using buffered reads (repack to enable O_DIRECT)\n",
                    (long long)got, K3_TRUNK_ALIGN);
            want_direct = 0;
        }
    }
    free(txt);                      /* arena holds the strings; txt itself is done */

    snprintf(p, sizeof p, "%s/trunk.bin", dir);
    tr->store = eng_storage_open_file(p, want_direct);
    if (!tr->store) { fprintf(stderr, "k3_trunk: cannot open %s\n", p); return -1; }
    tr->direct = eng_storage_is_direct(tr->store);

    EngBlock *blocks = (EngBlock *)malloc((size_t)tr->n_layers * sizeof *blocks);
    if (!blocks) return -1;
    int64_t total = 0;
    for (int i = 0; i < tr->n_layers; i++) {
        blocks[i].off    = tr->lay[i].file_off;
        blocks[i].nbytes = tr->lay[i].nbytes;
        total += tr->lay[i].nbytes;
    }

    EngStreamerCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.name         = "trunk";
    cfg.store        = tr->store;
    cfg.blocks       = blocks;
    cfg.nblocks      = tr->n_layers;
    cfg.budget_bytes = budget_bytes;
    /* Room for the handful of bf16 vectors k3_bind_layer_mem widens to fp32 in place. */
    cfg.extra_bytes  = (int64_t)k3_bind_widen_bytes(c);
    cfg.ring_want    = 2;
    cfg.async        = 1;
    cfg.hugepages    = -1;
    cfg.quiet        = 0;

    tr->sm = eng_streamer_create(&cfg);
    free(blocks);
    if (!tr->sm) return -1;

    printf("            %.2f GB packed, reads use %s\n", (double)total / 1e9,
           tr->direct ? "O_DIRECT (page cache bypassed)" : "buffered I/O");
    sync_stats(tr);
    return 0;
}

void k3_trunk_close(K3Trunk *tr)
{
    if (!tr) return;
    eng_streamer_destroy(tr->sm);
    if (tr->store && tr->store->close) tr->store->close(tr->store);
    if (tr->lay) {
        for (int i = 0; i < tr->n_layers; i++) free(tr->lay[i].t);
        free(tr->lay);
    }
    free(tr->json_arena);   /* every K3TrunkTensor.name points into this */
    memset(tr, 0, sizeof *tr);
}

int k3_trunk_bind(K3Trunk *tr, const K3Cfg *c, int L, K3LayerBind *b)
{
    if (L < 0 || L >= tr->n_layers) return -1;
    const double t_bind0 = now_s();
    k3_trunk_binds++;

    unsigned char *widen = NULL;
    unsigned char *base = eng_streamer_get(tr->sm, L, &widen);
    if (!base) return -1;

    Finder f; f.L = &tr->lay[L];
    K3MemSrc src; src.find = find_in_layer; src.ctx = &f;

    const double tw = now_s();
    const int rc = k3_bind_layer_mem(c, L, b, base, &src, widen,
                                     (size_t)k3_bind_widen_bytes(c), NULL);
    const double tnow = now_s();
    k3_trunk_widen_wall += tnow - tw;
    k3_trunk_bind_wall  += tnow - t_bind0;
    sync_stats(tr);
    return rc;
}

void k3_trunk_prefetch(K3Trunk *tr, int L)
{
    eng_streamer_prefetch(tr->sm, L);
}

void k3_trunk_report(const K3Trunk *tr, const char *label)
{
    eng_streamer_report(tr->sm, label);
}
