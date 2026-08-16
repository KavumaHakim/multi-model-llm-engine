/* SPDX-License-Identifier: Apache-2.0 */
/* gguf.c - see gguf.h. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "gguf.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ cursor -- */
/*
 * A bounds-checked walk over the header blob. Every read goes through this, so a
 * truncated or hostile file fails a length check instead of running off the end. `bad`
 * latches: once set, every subsequent read is a no-op, so the parse loop does not need
 * a check after each field.
 */
typedef struct {
    const unsigned char *p, *end;
    int bad;
} Cur;

static uint64_t rd(Cur *c, int n)
{
    if (c->bad || c->end - c->p < n) { c->bad = 1; return 0; }
    /* Byte-assembled rather than cast: GGUF is little-endian regardless of host, and
     * the u64 dimensions in the tensor table follow a variable-length name, so they are
     * not aligned to 8 and a cast would be an unaligned load. */
    uint64_t v = 0;
    for (int i = 0; i < n; i++) v |= (uint64_t)c->p[i] << (8 * i);
    c->p += n;
    return v;
}

static uint8_t  rd_u8 (Cur *c) { return (uint8_t)rd(c, 1); }
static uint16_t rd_u16(Cur *c) { return (uint16_t)rd(c, 2); }
static uint32_t rd_u32(Cur *c) { return (uint32_t)rd(c, 4); }
static uint64_t rd_u64(Cur *c) { return rd(c, 8); }

static float rd_f32(Cur *c)
{
    union { uint32_t u; float f; } v;
    v.u = rd_u32(c);
    return v.f;
}

static double rd_f64(Cur *c)
{
    union { uint64_t u; double f; } v;
    v.u = rd_u64(c);
    return v.f;
}

/* A GGUF string is a u64 length then that many bytes, NOT NUL-terminated. Returns a
 * pointer into the blob and the length; the caller copies if it needs a C string. */
static const char *rd_str(Cur *c, int64_t *len)
{
    const uint64_t n = rd_u64(c);
    if (c->bad || n > (uint64_t)(c->end - c->p)) { c->bad = 1; if (len) *len = 0; return NULL; }
    const char *s = (const char *)c->p;
    c->p += n;
    if (len) *len = (int64_t)n;
    return s;
}

/* Bytes one element of a scalar metadata type occupies. 0 for STRING and ARRAY, whose
 * sizes are not fixed. */
static int scalar_size(GgufType t)
{
    switch (t) {
        case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL:   return 1;
        case GGUF_T_UINT16: case GGUF_T_INT16:                   return 2;
        case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
        case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
        default: return 0;
    }
}

/* --------------------------------------------------------------- ggml types -- */

/* The ggml type ids, which are a wire format and not ours to choose. Only the ones this
 * engine can actually represent are mapped; anything else returns ENG_DT_INVALID and the
 * caller refuses the file rather than guessing a size. */
EngDType gguf_dtype(uint32_t t)
{
    switch (t) {
        case 0:  return ENG_DT_F32;
        case 1:  return ENG_DT_F16;
        case 2:  return ENG_DT_Q4_0;
        case 3:  return ENG_DT_Q4_1;
        case 6:  return ENG_DT_Q5_0;
        case 7:  return ENG_DT_Q5_1;
        case 8:  return ENG_DT_Q8_0;
        case 9:  return ENG_DT_Q8_1;
        case 10: return ENG_DT_Q2_K;
        case 11: return ENG_DT_Q3_K;
        case 12: return ENG_DT_Q4_K;
        case 13: return ENG_DT_Q5_K;
        case 14: return ENG_DT_Q6_K;
        case 15: return ENG_DT_Q8_K;
        case 24: return ENG_DT_I8;
        case 25: return ENG_DT_I16;
        case 26: return ENG_DT_I32;
        case 27: return ENG_DT_I64;
        case 28: return ENG_DT_F64;
        case 30: return ENG_DT_BF16;
        default: return ENG_DT_INVALID;
    }
}

const char *gguf_type_name(uint32_t t)
{
    const EngDType d = gguf_dtype(t);
    return d == ENG_DT_INVALID ? "?" : eng_dtype_name(d);
}

/* ------------------------------------------------------------- string pool -- */
/*
 * Keys and tensor names are copied out of the blob and NUL-terminated here. One bump
 * allocation rather than 400-odd strdups, and crucially NEVER reallocated: pointers
 * into it are handed out as they are made, so growing would invalidate every one.
 *
 * The capacity is therefore an upper bound computed up front. Every name lives inside
 * the header, so the header size bounds their total, and each contributes one extra
 * terminator byte -- bounded in turn by the number of names.
 */
static int pool_init(Gguf *g, int64_t cap)
{
    free(g->strpool);
    g->strpool = (char *)malloc((size_t)cap);
    g->strcap = cap;
    g->strused = 0;
    return g->strpool ? 0 : -1;
}

static const char *pool_add(Gguf *g, const char *s, int64_t n)
{
    if (!g->strpool || g->strused + n + 1 > g->strcap) {
        fprintf(stderr, "gguf: string pool exhausted (%lld of %lld)\n",
                (long long)g->strused, (long long)g->strcap);
        return NULL;
    }
    char *d = g->strpool + g->strused;
    memcpy(d, s, (size_t)n);
    d[n] = '\0';
    g->strused += n + 1;
    return d;
}

/* ------------------------------------------------------------------- open -- */

/* Read the whole header. Its size is not known until it has been walked, so this reads
 * a generous prefix, walks it, and grows once if the walk ran out. The target file's
 * header is 5.9 MB, dominated by 151,936 tokenizer strings. */
#define GGUF_HDR_FIRST  (8u << 20)
#define GGUF_HDR_MAX    (256u << 20)

static int parse_header(Gguf *g);

int gguf_open(Gguf *g, const char *path)
{
    if (!g || !path) return -1;
    memset(g, 0, sizeof *g);

    g->store = eng_storage_open_file(path, 0);
    if (!g->store) {
        fprintf(stderr, "gguf: cannot open %s\n", path);
        return -1;
    }
    g->owns_store = 1;
    g->file_size = g->store->size(g->store);

    if (g->file_size < 24) {
        fprintf(stderr, "gguf: %s is %lld bytes, too short to be a container\n",
                path, (long long)g->file_size);
        gguf_close(g);
        return -1;
    }

    int64_t want = GGUF_HDR_FIRST;
    if (want > g->file_size) want = g->file_size;

    for (;;) {
        unsigned char *b = (unsigned char *)realloc(g->blob, (size_t)want);
        if (!b) { gguf_close(g); return -1; }
        g->blob = b;
        const int64_t got = g->store->read(g->store, 0, want, g->blob);
        if (got != want) {
            fprintf(stderr, "gguf: short read of the header (%lld of %lld)\n",
                    (long long)got, (long long)want);
            gguf_close(g);
            return -1;
        }
        g->blob_size = want;

        const int rc = parse_header(g);
        if (rc == 0) break;
        if (rc == 1 && (want >= (int64_t)GGUF_HDR_MAX || want >= g->file_size)) {
            /* Ran out of blob with nowhere left to grow. Say which limit was hit: a
             * silent -1 here is indistinguishable from a parse error, and the two have
             * completely different causes. */
            fprintf(stderr, "gguf: header does not fit in %lld bytes "
                            "(file is %lld, cap is %u)\n",
                    (long long)want, (long long)g->file_size, GGUF_HDR_MAX);
            gguf_close(g);
            return -1;
        }
        if (rc == 1 && want < (int64_t)GGUF_HDR_MAX && want < g->file_size) {
            /* Ran out of blob: the header is bigger than the prefix read. Grow and
             * retry rather than failing, since the size cannot be known in advance. */
            want *= 4;
            if (want > g->file_size) want = g->file_size;
            if (want > (int64_t)GGUF_HDR_MAX) want = GGUF_HDR_MAX;
            free(g->kv); g->kv = NULL;
            free(g->t);  g->t = NULL;
            continue;
        }
        gguf_close(g);
        return -1;
    }
    return 0;
}

/* Returns 0 on success, 1 when the blob was too small (caller may grow), -1 on a file
 * that is malformed regardless of size. */
static int parse_header(Gguf *g)
{
    Cur c = { g->blob, g->blob + g->blob_size, 0 };

    if (g->blob_size < 24) return 1;
    if (memcmp(c.p, "GGUF", 4) != 0) {
        fprintf(stderr, "gguf: not a GGUF file (bad magic)\n");
        return -1;
    }
    c.p += 4;

    g->version = rd_u32(&c);
    /* v1 put counts in u32 and is long obsolete; v2 and v3 share this layout. Refusing
     * an unknown version is better than parsing it as v3 and reporting nonsense. */
    if (g->version != 2 && g->version != 3) {
        fprintf(stderr, "gguf: unsupported version %u (this reader handles 2 and 3)\n",
                g->version);
        return -1;
    }

    g->n_tensors = (int64_t)rd_u64(&c);
    g->n_kv      = (int64_t)rd_u64(&c);
    if (c.bad) return 1;
    if (g->n_tensors < 0 || g->n_tensors > (1 << 22) ||
        g->n_kv < 0 || g->n_kv > (1 << 20)) {
        fprintf(stderr, "gguf: implausible counts (%lld tensors, %lld kv)\n",
                (long long)g->n_tensors, (long long)g->n_kv);
        return -1;
    }

    g->kv = (GgufKV *)calloc((size_t)(g->n_kv ? g->n_kv : 1), sizeof *g->kv);
    g->t  = (GgufTensor *)calloc((size_t)(g->n_tensors ? g->n_tensors : 1), sizeof *g->t);
    if (!g->kv || !g->t) return -1;

    /* Upper bound: every name lies inside the header, plus one terminator each. */
    if (pool_init(g, g->blob_size + g->n_kv + g->n_tensors + 2) != 0) return -1;

    /* ---- metadata ---- */
    for (int64_t i = 0; i < g->n_kv; i++) {
        GgufKV *k = &g->kv[i];
        int64_t klen = 0;
        const char *key = rd_str(&c, &klen);
        if (c.bad) return 1;

        /* COPY. Do NOT NUL-terminate in place.
         *
         * GGUF strings are length-prefixed and not terminated, so the byte immediately
         * after a key is the first byte of its type tag -- and the cursor reads that tag
         * FROM THE BLOB. Writing a terminator there zeroes the low byte of the u32 type,
         * which for every real type (all < 256) turns it into 0, GGUF_T_UINT8. Each
         * value then consumes one byte instead of its real width, the parse desyncs
         * within the first key, and the walk eventually runs off the end -- reported as
         * "header does not fit" no matter how much is read. */
        k->key = pool_add(g, key, klen);
        if (!k->key) return -1;

        k->type = (GgufType)rd_u32(&c);
        switch (k->type) {
            case GGUF_T_UINT8:   k->i = rd_u8(&c);  break;
            case GGUF_T_INT8:    k->i = (int8_t)rd_u8(&c); break;
            case GGUF_T_UINT16:  k->i = rd_u16(&c); break;
            case GGUF_T_INT16:   k->i = (int16_t)rd_u16(&c); break;
            case GGUF_T_UINT32:  k->i = rd_u32(&c); break;
            case GGUF_T_INT32:   k->i = (int32_t)rd_u32(&c); break;
            case GGUF_T_UINT64:  k->i = (int64_t)rd_u64(&c); break;
            case GGUF_T_INT64:   k->i = (int64_t)rd_u64(&c); break;
            case GGUF_T_BOOL:    k->i = rd_u8(&c) ? 1 : 0; break;
            case GGUF_T_FLOAT32: k->f = rd_f32(&c); break;
            case GGUF_T_FLOAT64: k->f = rd_f64(&c); break;
            case GGUF_T_STRING:  k->str = rd_str(&c, &k->slen); break;
            case GGUF_T_ARRAY: {
                k->arr_type = (GgufType)rd_u32(&c);
                k->arr_len  = (int64_t)rd_u64(&c);
                if (c.bad) return 1;
                k->arr_data = c.p;
                /* SKIP the payload without materialising it. Fixed-width elements are
                 * one multiply; strings must be walked, which is why the token list is
                 * the expensive part of opening this file at all. */
                const int es = scalar_size(k->arr_type);
                if (es > 0) {
                    const uint64_t need = (uint64_t)es * (uint64_t)k->arr_len;
                    if (need > (uint64_t)(c.end - c.p)) return 1;
                    c.p += need;
                } else if (k->arr_type == GGUF_T_STRING) {
                    for (int64_t j = 0; j < k->arr_len; j++) {
                        rd_str(&c, NULL);
                        if (c.bad) return 1;
                    }
                } else {
                    fprintf(stderr, "gguf: array of unsupported type %d in '%s'\n",
                            (int)k->arr_type, k->key);
                    return -1;
                }
                break;
            }
            default:
                fprintf(stderr, "gguf: unknown metadata type %d for '%s'\n",
                        (int)k->type, k->key);
                return -1;
        }
        if (c.bad) return 1;
    }

    /* ---- tensor table ---- */
    for (int64_t i = 0; i < g->n_tensors; i++) {
        GgufTensor *t = &g->t[i];
        int64_t nlen = 0;
        const char *nm = rd_str(&c, &nlen);
        if (c.bad) return 1;
        /* Copied, for the same reason as the metadata keys above: the n_dims field
         * follows immediately and would lose its low byte. */
        t->name = pool_add(g, nm, nlen);
        if (!t->name) return -1;

        t->n_dims = (int)rd_u32(&c);
        if (c.bad) return 1;
        if (t->n_dims < 1 || t->n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "gguf: tensor '%s' has %d dimensions\n", t->name, t->n_dims);
            return -1;
        }
        for (int d = 0; d < t->n_dims; d++) t->shape[d] = (int64_t)rd_u64(&c);
        t->ggml_type = rd_u32(&c);
        t->rel_off   = (int64_t)rd_u64(&c);
        if (c.bad) return 1;

        t->dtype = gguf_dtype(t->ggml_type);
        if (t->dtype == ENG_DT_INVALID) {
            fprintf(stderr, "gguf: tensor '%s' has unsupported ggml type %u\n",
                    t->name, t->ggml_type);
            return -1;
        }
        t->nbytes = eng_tensor_nbytes(t->dtype, t->shape, t->n_dims);
        if (t->nbytes < 0) {
            fprintf(stderr, "gguf: tensor '%s' shape is not expressible in %s "
                            "(a partial block)\n", t->name, eng_dtype_name(t->dtype));
            return -1;
        }
    }

    /* ---- alignment and the data section ---- */
    g->alignment = 32;      /* the GGUF default when the key is absent */
    for (int64_t i = 0; i < g->n_kv; i++)
        if (!strcmp(g->kv[i].key, "general.alignment")) {
            if (g->kv[i].i > 0) g->alignment = g->kv[i].i;
            break;
        }
    if (g->alignment <= 0 || (g->alignment & (g->alignment - 1))) {
        fprintf(stderr, "gguf: general.alignment %lld is not a power of two\n",
                (long long)g->alignment);
        return -1;
    }

    const int64_t hdr_end = (int64_t)(c.p - g->blob);
    g->data_start = (hdr_end + g->alignment - 1) & ~(g->alignment - 1);

    for (int64_t i = 0; i < g->n_tensors; i++) {
        /* THE OFFSET IN THE TABLE IS RELATIVE TO THE DATA SECTION. Using it as an
         * absolute file offset reads the header instead of the weights and still
         * decodes to finite floats, so nothing downstream would catch it. */
        g->t[i].file_off = g->data_start + g->t[i].rel_off;
        const int64_t end = g->t[i].file_off + g->t[i].nbytes;
        if (end > g->file_size) {
            fprintf(stderr, "gguf: tensor '%s' ends at %lld, past the %lld-byte file\n",
                    g->t[i].name, (long long)end, (long long)g->file_size);
            return -1;
        }
        if (g->t[i].rel_off + g->t[i].nbytes > g->data_bytes)
            g->data_bytes = g->t[i].rel_off + g->t[i].nbytes;
    }
    return 0;
}

void gguf_close(Gguf *g)
{
    if (!g) return;
    if (g->owns_store && g->store && g->store->close) g->store->close(g->store);
    free(g->blob);
    free(g->strpool);
    free(g->kv);
    free(g->t);
    memset(g, 0, sizeof *g);
}

/* --------------------------------------------------------------- accessors -- */

const GgufKV *gguf_find(const Gguf *g, const char *key)
{
    if (!g || !key) return NULL;
    for (int64_t i = 0; i < g->n_kv; i++)
        if (!strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

static int kv_is_int(const GgufKV *k)
{
    switch (k->type) {
        case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_UINT16: case GGUF_T_INT16:
        case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_UINT64: case GGUF_T_INT64:
        case GGUF_T_BOOL:
            return 1;
        default: return 0;
    }
}

int gguf_i64(const Gguf *g, const char *key, int64_t *out)
{
    const GgufKV *k = gguf_find(g, key);
    if (!k || !kv_is_int(k)) return -1;
    if (out) *out = k->i;
    return 0;
}

int gguf_u32(const Gguf *g, const char *key, uint32_t *out)
{
    int64_t v = 0;
    if (gguf_i64(g, key, &v) != 0) return -1;
    if (v < 0 || v > 0xFFFFFFFFLL) return -1;
    if (out) *out = (uint32_t)v;
    return 0;
}

int gguf_f32(const Gguf *g, const char *key, float *out)
{
    const GgufKV *k = gguf_find(g, key);
    if (!k) return -1;
    if (k->type == GGUF_T_FLOAT32 || k->type == GGUF_T_FLOAT64) {
        if (out) *out = (float)k->f;
        return 0;
    }
    /* An integer where a float was expected is fine and happens: some writers store
     * 1.0 as an int. The reverse is not, and is not accepted. */
    if (kv_is_int(k)) {
        if (out) *out = (float)k->i;
        return 0;
    }
    return -1;
}

int gguf_str(const Gguf *g, const char *key, const char **out, int64_t *len)
{
    const GgufKV *k = gguf_find(g, key);
    if (!k || k->type != GGUF_T_STRING) return -1;
    if (out) *out = k->str;
    if (len) *len = k->slen;
    return 0;
}

const char *gguf_arr_str(const Gguf *g, const GgufKV *kv, int64_t i, int64_t *len)
{
    if (!g || !kv || kv->type != GGUF_T_ARRAY || kv->arr_type != GGUF_T_STRING) return NULL;
    if (i < 0 || i >= kv->arr_len) return NULL;

    /* Walk. Strings are variable-length, so there is no index to jump with; a caller
     * iterating the whole array in order pays O(n) total, one that random-accesses pays
     * O(n) each. The tokenizer builds its own table once, so this is a diagnostic path
     * rather than a hot one. */
    Cur c = { kv->arr_data, g->blob + g->blob_size, 0 };
    const char *s = NULL;
    for (int64_t j = 0; j <= i; j++) {
        s = rd_str(&c, len);
        if (c.bad) return NULL;
    }
    return s;
}

const GgufTensor *gguf_tensor(const Gguf *g, const char *name)
{
    if (!g || !name) return NULL;
    for (int64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->t[i].name, name)) return &g->t[i];
    return NULL;
}

int gguf_layout_is_sequential(const Gguf *g)
{
    if (!g) return 0;
    for (int64_t i = 1; i < g->n_tensors; i++)
        if (g->t[i].rel_off < g->t[i - 1].rel_off) return 0;
    return 1;
}

void gguf_report(const Gguf *g, const char *label)
{
    if (!g) return;
    const char *arch = NULL, *name = NULL;
    int64_t l = 0;
    gguf_str(g, "general.architecture", &arch, &l);
    gguf_str(g, "general.name", &name, &l);

    printf("%s%sgguf v%u\n", label ? label : "", label ? " " : "", g->version);
    printf("  arch      : %.*s\n", arch ? 32 : 7, arch ? arch : "unknown");
    if (name) printf("  name      : %.*s\n", (int)l, name);
    printf("  tensors   : %lld,  metadata keys: %lld\n",
           (long long)g->n_tensors, (long long)g->n_kv);
    printf("  alignment : %lld,  data starts at %lld\n",
           (long long)g->alignment, (long long)g->data_start);
    printf("  data      : %.2f GB of %.2f GB file\n",
           (double)g->data_bytes / 1e9, (double)g->file_size / 1e9);
    printf("  layout    : %s\n", gguf_layout_is_sequential(g)
           ? "sequential (a layer is one contiguous run)"
           : "NOT sequential (streaming will seek)");
}
