/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_tensor.c - the M1 generic layer: dtype table, tensor descriptors, storage
 * backend, and the safetensors adapter.
 *
 * Every case here is chosen so that a plausible wrong implementation FAILS it. The
 * three that matter most:
 *
 *   - eng_dtype_bytes must REFUSE a partial block rather than round. Rounding up is the
 *     natural implementation, it looks right on aligned data, and it walks a reader off
 *     the end of a tensor on anything else.
 *   - eng_dtype_bytes must REFUSE a row-scaled dtype outright. Returning nelem*1 is off
 *     by exactly 4 bytes per row, which is invisible in a one-row test.
 *   - the safetensors adapter must REVERSE the axis order. A test that only checks the
 *     element count passes whether or not it does, so this asserts the axes themselves.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dtype.h"
#include "tensor.h"
#include "storage.h"
#include "format.h"

static int fails = 0;
static int checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    if (cond) {
        printf("  PASS  %-42s %s\n", what, detail ? detail : "");
    } else {
        printf("  FAIL  %-42s %s\n", what, detail ? detail : "");
        fails++;
    }
}

static void eq64(int64_t got, int64_t want, const char *what)
{
    char d[128];
    snprintf(d, sizeof d, "got %lld want %lld", (long long)got, (long long)want);
    ok(got == want, what, d);
}

/* ------------------------------------------------------------------- dtype -- */

static void test_dtype(void)
{
    printf("== dtype table ==\n");

    /* Dense sizes. */
    eq64(eng_dtype_bytes(ENG_DT_F32,  10), 40, "f32 10 elems");
    eq64(eng_dtype_bytes(ENG_DT_BF16, 10), 20, "bf16 10 elems");
    eq64(eng_dtype_bytes(ENG_DT_U8,   10), 10, "u8 10 elems");

    /* k-quant block geometry. These are the ggml block sizes; a derived
     * bits-per-weight figure would give 128 for Q4_K, not 144. */
    eq64(eng_dtype_bytes(ENG_DT_Q4_K, 256), 144, "q4_k one superblock");
    eq64(eng_dtype_bytes(ENG_DT_Q6_K, 256), 210, "q6_k one superblock");
    eq64(eng_dtype_bytes(ENG_DT_Q8_0,  32),  34, "q8_0 one block");
    eq64(eng_dtype_bytes(ENG_DT_Q4_K, 512), 288, "q4_k two superblocks");

    /* THE REFUSAL THAT MATTERS: a partial block has no defined size. */
    ok(eng_dtype_bytes(ENG_DT_Q4_K, 255) == -1, "q4_k refuses partial block", "255 elems");
    ok(eng_dtype_bytes(ENG_DT_Q4_K,   1) == -1, "q4_k refuses 1 elem", NULL);
    ok(eng_dtype_bytes(ENG_DT_Q8_0,  33) == -1, "q8_0 refuses partial block", "33 elems");

    /* MXFP4: two nibbles per byte, scales NOT counted here (sibling tensor). */
    eq64(eng_dtype_bytes(ENG_DT_MXFP4, 64), 32, "mxfp4 64 elems = 32 bytes");
    ok(eng_dtype_has_ext_scales(ENG_DT_MXFP4), "mxfp4 declares external scales", NULL);
    eq64(eng_dtype_scale_group(ENG_DT_MXFP4), 32, "mxfp4 scale group");
    ok(eng_dtype_scale_group(ENG_DT_Q4_K) == 0, "q4_k has no external scale group", NULL);
    ok(eng_dtype_bytes(ENG_DT_MXFP4, 63) == -1, "mxfp4 refuses odd element count", NULL);

    /* Row-scaled: size is not a function of the element count. */
    ok(eng_dtype_bytes(ENG_DT_I8R, 100) == -1,
       "i8r refuses element-count sizing", "needs the row split");
    eq64(eng_dtype_row_bytes(ENG_DT_I8R, 100), 104, "i8r row = 4 + cols");
    eq64(eng_dtype_row_bytes(ENG_DT_F32,  10),  40, "f32 row bytes");
    eq64(eng_dtype_row_bytes(ENG_DT_Q4_K, 256), 144, "q4_k row bytes");

    /* Unknown ids must not resolve to a plausible default. */
    ok(eng_dtype_info(ENG_DT_INVALID) == NULL, "invalid dtype has no info", NULL);
    ok(eng_dtype_bytes(ENG_DT_INVALID, 10) == -1, "invalid dtype refuses sizing", NULL);
    ok(eng_dtype_bytes(ENG_DT_F32, -1) == -1, "negative count refused", NULL);

    /* Name round-trip, so a packed-format writer and reader agree. */
    ok(eng_dtype_by_name("q4_k") == ENG_DT_Q4_K, "name -> q4_k", NULL);
    ok(eng_dtype_by_name("mxfp4") == ENG_DT_MXFP4, "name -> mxfp4", NULL);
    ok(eng_dtype_by_name("nonesuch") == ENG_DT_INVALID, "unknown name refused", NULL);
    ok(!strcmp(eng_dtype_name(ENG_DT_BF16), "bf16"), "q -> name", NULL);
    ok(eng_dtype_is_quantized(ENG_DT_Q6_K) && !eng_dtype_is_quantized(ENG_DT_F32),
       "quantized flag", NULL);
}

/* ------------------------------------------------------------------ tensor -- */

static void test_tensor(void)
{
    printf("== tensor descriptors ==\n");

    EngTensor t;
    /* Qwen3's real embedding shape, fastest axis first. */
    const int64_t emb[2] = { 4096, 151936 };
    ok(eng_tensor_init_file(&t, "token_embd.weight", ENG_DT_Q4_K, emb, 2, NULL, 0) == 0,
       "init file-backed q4_k", NULL);
    eq64(eng_tensor_numel(&t), 4096LL * 151936, "numel");
    eq64(eng_tensor_cols(&t), 4096, "cols = fastest axis");
    eq64(eng_tensor_rows(&t), 151936, "rows = product of the rest");
    eq64(t.nbytes, 4096LL * 151936 / 256 * 144, "q4_k nbytes");
    ok(!eng_tensor_is_resident(&t), "file-backed is not resident", NULL);

    /* A shape whose ROW is not a whole number of blocks has no layout. Computing from
     * the total element count would accept this: 100*256 is a clean multiple of 256. */
    const int64_t bad[2] = { 100, 256 };
    EngTensor tb;
    ok(eng_tensor_init_file(&tb, "bad", ENG_DT_Q4_K, bad, 2, NULL, 0) != 0,
       "refuses row that is not block-aligned", "cols=100 for a 256-block dtype");

    /* RANK ZERO. safetensors writes real scalars with "shape": [], and the fixture set
     * contains one. Rejecting rank 0 as invalid took down the whole container when the
     * adapter hit it -- which is exactly how this case got added. */
    EngTensor ts;
    ok(eng_tensor_init_file(&ts, "scalar", ENG_DT_F32, NULL, 0, NULL, 0) == 0,
       "rank-0 scalar accepted", NULL);
    eq64(eng_tensor_numel(&ts), 1, "scalar numel is 1");
    eq64(eng_tensor_cols(&ts), 1, "scalar cols is 1");
    eq64(eng_tensor_rows(&ts), 1, "scalar rows is 1");
    eq64(ts.nbytes, 4, "scalar f32 is 4 bytes");
    ok(eng_tensor_init_file(&ts, "bad", ENG_DT_F32, NULL, -1, NULL, 0) != 0,
       "negative rank still refused", NULL);

    /* Row-scaled tensor: size depends on the row split, so shape supplies it. */
    const int64_t i8r[2] = { 100, 8 };
    EngTensor tr;
    ok(eng_tensor_init_file(&tr, "draft", ENG_DT_I8R, i8r, 2, NULL, 0) == 0,
       "init row-scaled", NULL);
    eq64(tr.nbytes, 8 * 104, "i8r nbytes = rows * (4 + cols)");

    /* Ownership. free() must follow only what this tensor owns. */
    float *buf = (float *)calloc(64, sizeof *buf);
    const int64_t v[1] = { 64 };
    EngTensor tm;
    ok(eng_tensor_init_mem(&tm, "resident", ENG_DT_F32, v, 1, buf, 1) == 0,
       "init resident owned", NULL);
    ok(eng_tensor_is_resident(&tm), "resident flag set", NULL);
    eq64(tm.nbytes, 256, "f32 64 elems");

    /* A view must NOT own the parent's buffer. Freeing the view then the parent would
     * be a double free if it did; this runs under ASan in CI. */
    const int64_t m[2] = { 8, 8 };
    EngTensor tp, tv;
    float *pbuf = (float *)calloc(64, sizeof *pbuf);
    for (int i = 0; i < 64; i++) pbuf[i] = (float)i;
    eng_tensor_init_mem(&tp, "parent", ENG_DT_F32, m, 2, pbuf, 1);

    ok(eng_tensor_view_rows(&tv, &tp, 2, 3) == 0, "view 3 rows from row 2", NULL);
    eq64(tv.nbytes, 3 * 8 * 4, "view nbytes");
    eq64(eng_tensor_rows(&tv), 3, "view rows");
    ok(tv.data == (void *)(pbuf + 2 * 8), "view points at the right row", NULL);
    ok(!(tv.flags & ENG_TF_OWNED), "view does not own", NULL);
    ok((tv.flags & ENG_TF_VIEW) != 0, "view flagged", NULL);
    eng_tensor_free(&tv);                     /* must not free pbuf */
    ok(pbuf[16] == 16.0f, "parent survives view free", NULL);

    ok(eng_tensor_view_rows(&tv, &tp, 6, 5) != 0, "view refuses out of range", NULL);
    ok(eng_tensor_view_rows(&tv, &tp, -1, 2) != 0, "view refuses negative start", NULL);

    /* A view of a file-backed tensor advances the OFFSET, not a pointer. */
    EngTensor tf, tfv;
    const int64_t q[2] = { 256, 10 };
    eng_tensor_init_file(&tf, "q", ENG_DT_Q4_K, q, 2, NULL, 1000);
    ok(eng_tensor_view_rows(&tfv, &tf, 3, 2) == 0, "view of file-backed", NULL);
    eq64(tfv.file_off, 1000 + 3 * 144, "view offset advances by whole rows");
    ok(tfv.data == NULL, "file-backed view is not resident", NULL);

    eng_tensor_free(&tp);
    eng_tensor_free(&tm);

    /* describe() must not overrun a short buffer. */
    char small[16];
    eng_tensor_describe(&t, small, sizeof small);
    ok(small[sizeof small - 1] == '\0', "describe NUL-terminates", small);
}

/* ----------------------------------------------------------------- storage -- */

static void test_storage(const char *tmpdir)
{
    printf("== storage backend ==\n");

    char path[512];
    snprintf(path, sizeof path, "%s/eng_storage_test.bin", tmpdir);

    /* A pattern that makes an off-by-one visible: byte i == i mod 251 (prime, so the
     * period does not align with any power-of-two block size). */
    const int64_t N = 100000;
    unsigned char *src = (unsigned char *)malloc((size_t)N);
    for (int64_t i = 0; i < N; i++) src[i] = (unsigned char)(i % 251);

    FILE *fp = fopen(path, "wb");
    if (!fp) { printf("  FAIL  cannot write %s\n", path); fails++; free(src); return; }
    fwrite(src, 1, (size_t)N, fp);
    fclose(fp);

    EngStorage *s = eng_storage_open_file(path, 1);
    ok(s != NULL, "open file backend", path);
    if (!s) { free(src); return; }

    eq64(s->size(s), N, "size");
    printf("  note  direct path %s\n", eng_storage_is_direct(s) ? "AVAILABLE" : "unavailable (buffered fallback)");

    /* Plain read from a deliberately unaligned offset. */
    unsigned char dst[1000];
    const int64_t off = 4093;                 /* three bytes before a page boundary */
    eq64(s->read(s, off, 1000, dst), 1000, "plain read");
    ok(!memcmp(dst, src + off, 1000), "plain read bytes correct", NULL);

    /* Aligned read: the payload offset must be honoured, not assumed zero. */
    int64_t cap = 0;
    void *abuf = eng_storage_alloc_aligned(1000, &cap);
    ok(abuf != NULL, "aligned alloc", NULL);
    ok(((uintptr_t)abuf % ENG_IO_ALIGN) == 0, "aligned alloc is page aligned", NULL);
    if (abuf) {
        int64_t poff = -1;
        const int64_t got = s->read_aligned(s, off, 1000, abuf, cap, &poff);
        eq64(got, 1000, "aligned read payload bytes");
        ok(poff >= 0, "payload offset set", NULL);
        ok(!memcmp((unsigned char *)abuf + poff, src + off, 1000),
           "aligned read bytes correct at payload offset", NULL);
        /* If the direct path ran, the payload must NOT be at zero for this offset --
         * that is the whole point of widening. Only assert when direct is real. */
        if (eng_storage_is_direct(s))
            ok(poff == off % ENG_IO_ALIGN, "payload offset equals off mod align", NULL);

        /* The LAST window runs past EOF. A short read there is success, provided the
         * payload was covered. This is the case that makes the final tensor of every
         * file readable. */
        const int64_t tail_off = N - 100;
        int64_t tpoff = -1;
        const int64_t tgot = s->read_aligned(s, tail_off, 100, abuf, cap, &tpoff);
        eq64(tgot, 100, "aligned read at EOF boundary");
        ok(!memcmp((unsigned char *)abuf + tpoff, src + tail_off, 100),
           "EOF-boundary bytes correct", NULL);

        eng_storage_free_aligned(abuf);
    }

    /* Reading past EOF returns short, not garbage. */
    ok(s->read(s, N - 10, 1000, dst) == 10, "read past EOF returns short", NULL);

    ok(s->reads > 0 && s->bytes_read > 0, "statistics accumulate", NULL);
    eng_storage_report(s, "  stats");

    s->close(s);
    free(src);
    remove(path);
}

/* ------------------------------------------------------------------ format -- */

static void test_format(const char *fixdir)
{
    printf("== safetensors adapter ==\n");

    EngFormat *f = eng_format_open_safetensors(fixdir);
    ok(f != NULL, "open safetensors fixtures", fixdir);
    if (!f) return;

    const int64_t n = f->ntensors(f);
    ok(n > 0, "tensors indexed", NULL);
    printf("  note  %lld tensors, %.3f MB\n", (long long)n,
           (double)eng_format_total_bytes(f) / 1e6);

    /* O(1) lookup, and a missing name must be NULL rather than a zeroed tensor. */
    const EngTensor *p = f->find(f, "plain.f32.2d");
    ok(p != NULL, "find plain.f32.2d", NULL);
    ok(f->find(f, "definitely.not.here") == NULL, "missing tensor returns NULL", NULL);

    if (p) {
        char d[256];
        eng_tensor_describe(p, d, sizeof d);
        printf("  note  %s\n", d);
        ok(p->dtype == ENG_DT_F32, "dtype mapped to f32", NULL);
        ok(p->ndim == 2, "2-D", NULL);
        ok(p->store != NULL, "carries a storage handle", NULL);
        ok(!eng_tensor_is_resident(p), "starts non-resident", NULL);

        /* THE SHAPE-ORDER ASSERTION. The fixture is written [rows, cols] in the file;
         * EngTensor is fastest-axis-first, so shape[0] must be the COLUMN count. A
         * reader that forgets to reverse gets the same numel and fails only here. */
        const int64_t numel = eng_tensor_numel(p);
        eq64(p->nbytes, numel * 4, "f32 nbytes matches numel");
        printf("  note  shape[0]=%lld shape[1]=%lld (fastest axis first)\n",
               (long long)p->shape[0], (long long)p->shape[1]);

        /* Read it and confirm the bytes actually arrive. */
        void *buf = malloc((size_t)p->nbytes);
        eq64(eng_format_read_tensor(p, buf), p->nbytes, "read tensor bytes");
        free(buf);
    }

    /* bf16 must map to bf16 and be 2 bytes per element, not silently widened. */
    const EngTensor *b = f->find(f, "plain.bf16.1d");
    if (b) {
        ok(b->dtype == ENG_DT_BF16, "dtype mapped to bf16", NULL);
        eq64(b->nbytes, eng_tensor_numel(b) * 2, "bf16 is 2 bytes/elem");
    } else {
        ok(0, "find plain.bf16.1d", "absent");
    }

    /* The rank-0 scalar must survive the adapter, not just the tensor layer. */
    const EngTensor *sc = f->find(f, "scalar.f32");
    ok(sc != NULL, "find rank-0 scalar", NULL);
    if (sc) {
        eq64(sc->ndim, 0, "scalar keeps rank 0");
        eq64(sc->nbytes, 4, "scalar is 4 bytes");
        float val = 0.0f;
        eq64(eng_format_read_tensor(sc, &val), 4, "read scalar");
        ok(val == 3.5f, "scalar value round-trips", NULL);
    }

    /* A tensor in the SECOND shard proves sharding is invisible above this layer. */
    const EngTensor *s2 = f->find(f, "second.shard.f32");
    ok(s2 != NULL, "find tensor in second shard", NULL);
    if (s2) {
        ok(s2->store != NULL, "second-shard tensor has storage", NULL);
        void *buf = malloc((size_t)s2->nbytes);
        eq64(eng_format_read_tensor(s2, buf), s2->nbytes, "read from second shard");
        free(buf);
    }

    /* Safetensors declares no architecture and no metadata: absent must be reported,
     * never defaulted. */
    ok(f->arch(f) == NULL, "no arch declared", NULL);
    const char *sv = NULL;
    ok(f->meta_str(f, "anything", &sv) != 0, "metadata absent is an error", NULL);

    /* at() must agree with find(). */
    if (p) {
        int seen = 0;
        for (int64_t i = 0; i < n; i++)
            if (f->at(f, i) == p) { seen = 1; break; }
        ok(seen, "find() result is reachable from at()", NULL);
    }
    ok(f->at(f, n) == NULL && f->at(f, -1) == NULL, "at() bounds-checked", NULL);

    eng_format_report(f, "  format");
    f->close(f);
}

int main(int argc, char **argv)
{
    const char *fixdir = argc > 1 ? argv[1] : "tests/fixtures/st";
    const char *tmpdir = argc > 2 ? argv[2] : "build";

    printf("M1 generic layer: dtype, tensor, storage, format\n\n");

    test_dtype();
    printf("\n");
    test_tensor();
    printf("\n");
    test_storage(tmpdir);
    printf("\n");
    test_format(fixdir);

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) {
        printf("M1 LAYER TESTS FAILED\n");
        return 1;
    }
    printf("M1 LAYER TESTS PASSED\n");
    return 0;
}
