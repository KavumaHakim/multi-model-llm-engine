/* SPDX-License-Identifier: Apache-2.0 */
/*
 * safetensors.c - the safetensors container behind EngFormat.
 *
 * THIS IS AN ADAPTER, NOT A SECOND READER.
 *   The header parsing, the escaped-name handling, the dtype widening and the
 *   open-addressed hash index over 497,220 tensors all stay in src/io/k3_st.c, which
 *   is tested by tests/unit/test_st.c against fixtures covering tail bytes, escaped
 *   names and non-finite values. Rewriting that here would mean re-earning coverage
 *   that already exists.
 *
 *   What this file adds is the generic surface: EngTensor descriptors with proper
 *   dtypes and shapes, and an EngStorage handle per shard so callers read a TENSOR
 *   rather than a shard-and-offset.
 *
 * TEMPORARY DUPLICATION, and when it goes away
 *   K3St keeps its own descriptors for the existing K3 code paths, and this adapter
 *   opens EngStorage handles on the same files, so a shard is open twice while both
 *   exist. That is the migration bridge the brief allows: at M7, when the K3 backend
 *   moves onto the generic runtime, k3_st.c's readers are deleted and only the
 *   EngStorage handles remain. It is recorded here so it is not mistaken for a design.
 *
 *   Cost while it lasts: two file descriptors per shard instead of one. K3's 96-shard
 *   checkpoint is the worst case at 192, well inside any default ulimit.
 *
 * SHAPE ORDER
 *   Safetensors lists shape ROW-MAJOR (slowest axis first); EngTensor uses
 *   fastest-first to match GGUF. So the dimensions are REVERSED on the way in. A
 *   [151936, 4096] embedding in the file becomes shape[0]=4096, shape[1]=151936 here,
 *   which is the same memory. Getting this backwards transposes every matrix and is
 *   the single most likely loader bug, so test_format asserts it explicitly.
 */
#include "format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_st.h"

typedef struct {
    K3St        st;
    EngStorage **shard;      /* [st.nshard] */
    EngTensor   *t;          /* [st.nt] descriptors, discovery order */
    int64_t      nt;
    char        *path;
} StCtx;

/* K3Dtype is the safetensors reader's own enum; EngDType is the engine's. The mapping
 * is total in the direction that matters -- every dtype the reader can produce has an
 * engine equivalent -- and an unknown one yields ENG_DT_INVALID, which makes
 * eng_tensor_init_file fail loudly rather than sizing the tensor wrongly. */
static EngDType map_dtype(K3Dtype d)
{
    switch (d) {
        case K3_DT_F32:  return ENG_DT_F32;
        case K3_DT_BF16: return ENG_DT_BF16;
        case K3_DT_F16:  return ENG_DT_F16;
        case K3_DT_U8:   return ENG_DT_U8;
        case K3_DT_I8R:  return ENG_DT_I8R;
        default:         return ENG_DT_INVALID;
    }
}

static int64_t st_ntensors(EngFormat *f)
{
    return ((StCtx *)f->ctx)->nt;
}

static const EngTensor *st_at(EngFormat *f, int64_t i)
{
    StCtx *c = (StCtx *)f->ctx;
    if (i < 0 || i >= c->nt) return NULL;
    return &c->t[i];
}

static const EngTensor *st_find(EngFormat *f, const char *name)
{
    StCtx *c = (StCtx *)f->ctx;
    const K3Tensor *kt = k3_st_find(&c->st, name);
    if (!kt) return NULL;
    /* K3Tensor lives in the same discovery-ordered array as our descriptors, so its
     * index is the descriptor's index. Derived by pointer arithmetic rather than by a
     * second lookup: the hash index is the expensive thing to build and there is no
     * reason to build two. */
    const int64_t idx = (int64_t)(kt - c->st.t);
    if (idx < 0 || idx >= c->nt) return NULL;
    return &c->t[idx];
}

/* Safetensors carries no model metadata of its own -- the released checkpoints put it
 * in a sibling config.json, which k3_cfg.h reads. Reporting "absent" is correct and is
 * not a stub: a caller must not silently get a default. */
static int st_meta_str(EngFormat *f, const char *key, const char **out)
{
    (void)f; (void)key; (void)out; return -1;
}
static int st_meta_i64(EngFormat *f, const char *key, int64_t *out)
{
    (void)f; (void)key; (void)out; return -1;
}
static int st_meta_f64(EngFormat *f, const char *key, double *out)
{
    (void)f; (void)key; (void)out; return -1;
}
static const char *st_arch(EngFormat *f) { (void)f; return NULL; }

static void st_close(EngFormat *f)
{
    if (!f) return;
    StCtx *c = (StCtx *)f->ctx;
    if (c) {
        if (c->shard) {
            for (int i = 0; i < c->st.nshard; i++)
                if (c->shard[i] && c->shard[i]->close) c->shard[i]->close(c->shard[i]);
            free(c->shard);
        }
        free(c->t);
        k3_st_close(&c->st);
        free(c->path);
        free(c);
    }
    free(f);
}

EngFormat *eng_format_open_safetensors(const char *dir)
{
    if (!dir) return NULL;

    StCtx *c = (StCtx *)calloc(1, sizeof *c);
    EngFormat *f = (EngFormat *)calloc(1, sizeof *f);
    if (!c || !f) { free(c); free(f); return NULL; }

    if (k3_st_open(&c->st, dir) != 0) { free(c); free(f); return NULL; }

    c->path  = strdup(dir);
    c->nt    = c->st.nt;
    c->shard = (EngStorage **)calloc((size_t)(c->st.nshard > 0 ? c->st.nshard : 1),
                                     sizeof *c->shard);
    c->t     = (EngTensor *)calloc((size_t)(c->nt > 0 ? c->nt : 1), sizeof *c->t);
    if (!c->shard || !c->t) { st_close(f); return NULL; }

    for (int i = 0; i < c->st.nshard; i++) {
        c->shard[i] = eng_storage_open_file(c->st.path[i], 1);
        if (!c->shard[i]) {
            fprintf(stderr, "safetensors: cannot open shard %s\n", c->st.path[i]);
            st_close(f);
            return NULL;
        }
    }

    for (int64_t i = 0; i < c->nt; i++) {
        const K3Tensor *kt = &c->st.t[i];
        const EngDType dt = map_dtype(kt->dtype);

        /* Reverse the axis order: safetensors is slowest-first, EngTensor is
         * fastest-first. Same bytes, and the reversal is the whole conversion. */
        int64_t shape[ENG_MAX_DIMS];
        const int nd = kt->ndim > ENG_MAX_DIMS ? ENG_MAX_DIMS : kt->ndim;
        for (int d = 0; d < nd; d++) shape[d] = kt->shape[nd - 1 - d];

        EngStorage *sto = (kt->shard >= 0 && kt->shard < c->st.nshard)
                        ? c->shard[kt->shard] : NULL;

        if (eng_tensor_init_file(&c->t[i], kt->name, dt, shape, nd, sto, kt->off) != 0) {
            /* A descriptor we cannot size is not something to skip past: it means the
             * dtype or the shape is not what this engine believes, and every later
             * offset computed from it would be wrong. */
            fprintf(stderr, "safetensors: cannot describe tensor '%s' "
                            "(dtype %d, ndim %d)\n", kt->name, (int)kt->dtype, kt->ndim);
            st_close(f);
            return NULL;
        }

        /* The reader knows the true on-disk length; trust it over our computed one and
         * say so if they disagree, because a mismatch means the dtype table is wrong. */
        if (c->t[i].nbytes != kt->nbytes) {
            fprintf(stderr, "safetensors: size mismatch on '%s': file says %lld, "
                            "dtype %s over shape implies %lld\n",
                    kt->name, (long long)kt->nbytes,
                    eng_dtype_name(dt), (long long)c->t[i].nbytes);
            st_close(f);
            return NULL;
        }
    }

    f->name     = "safetensors";
    f->path     = c->path;
    f->ntensors = st_ntensors;
    f->find     = st_find;
    f->at       = st_at;
    f->meta_str = st_meta_str;
    f->meta_i64 = st_meta_i64;
    f->meta_f64 = st_meta_f64;
    f->arch     = st_arch;
    f->close    = st_close;
    f->ctx      = c;
    return f;
}

/* ------------------------------------------------------------------ dispatch -- */

EngFormat *eng_format_open(const char *path)
{
    if (!path) return NULL;

    /* GGUF announces itself in its first four bytes. Sniffing beats trusting the
     * extension: a .bin that is really a GGUF should still load. */
    FILE *fp = fopen(path, "rb");
    if (fp) {
        char magic[4] = {0};
        const size_t got = fread(magic, 1, 4, fp);
        fclose(fp);
        if (got == 4 && !memcmp(magic, "GGUF", 4)) {
            fprintf(stderr, "format: %s is GGUF; the GGUF reader lands in M8\n", path);
            return NULL;
        }
    }

    /* Otherwise treat it as a safetensors file or directory; k3_st_open handles both. */
    return eng_format_open_safetensors(path);
}

/* ------------------------------------------------------------------ helpers -- */

int64_t eng_format_read_tensor(const EngTensor *t, void *dst)
{
    if (!t || !dst) return 0;
    if (t->data) {                       /* already resident: nothing to read */
        memcpy(dst, t->data, (size_t)t->nbytes);
        return t->nbytes;
    }
    if (!t->store || !t->store->read) return 0;
    return t->store->read(t->store, t->file_off, t->nbytes, dst);
}

int64_t eng_format_total_bytes(EngFormat *f)
{
    if (!f || !f->at || !f->ntensors) return 0;
    const int64_t n = f->ntensors(f);
    int64_t tot = 0;
    for (int64_t i = 0; i < n; i++) {
        const EngTensor *t = f->at(f, i);
        if (t) tot += t->nbytes;
    }
    return tot;
}

void eng_format_report(EngFormat *f, const char *label)
{
    if (!f) return;
    printf("%s%s%s: %lld tensors, %.2f MB\n",
           label ? label : "", label ? " " : "", f->name,
           (long long)(f->ntensors ? f->ntensors(f) : 0),
           (double)eng_format_total_bytes(f) / 1e6);
}
