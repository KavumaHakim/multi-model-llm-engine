/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_gguf.c - the GGUF reader and the Q4_K / Q6_K kernels.
 *
 * TWO TIERS, and the split is what makes this useful rather than decorative:
 *
 *   ALWAYS. The k-quant kernels are checked against tests/fixtures/gguf/golden.bin,
 *   which carries REAL quantised blocks lifted out of Qwen3-8B-Q4_K_M.gguf together
 *   with the values an INDEPENDENT numpy implementation (tools/gguf_ref.py) decoded
 *   them to. Because the raw bytes travel with the expected values, this runs on a
 *   machine that has never seen the 5 GB model -- which is the machine where a broken
 *   kernel is most likely to go unnoticed.
 *
 *   WHEN THE MODEL IS PRESENT (GGUF_MODEL, or the default path below). The reader is
 *   run against the real container and checked against facts the same reference derived
 *   independently: 399 tensors, data at 5,956,416, and a tensor-byte total that must
 *   close the file exactly.
 *
 * WHY AN INDEPENDENT REFERENCE AND NOT A SELF-CONSISTENCY CHECK. A wrong nibble order,
 * a wrong scale index, an unsigned read of Q6_K's signed scales, or a dropped affine
 * term in Q4_K all yield weights that are finite, plausibly scaled, and wrong. The
 * engine would load, run, and emit fluent text. Comparing the fused dot against this
 * engine's own dequantiser catches none of those, because both share the mistake. The
 * numpy reference was written from the block layout and vectorised over blocks, so it
 * shares neither code nor loop structure with the C.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dtype.h"
#include "gguf.h"
#include "quant.h"

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

static uint32_t rs = 424242u;
static float frand(void)
{
    rs = rs * 1664525u + 1013904223u;
    return (float)((double)(rs >> 8) / (double)(1u << 24) * 2.0 - 1.0);
}

/* ------------------------------------------------------------- golden file -- */

typedef struct {
    char     name[96];
    uint32_t gtype;
    uint64_t row;
    uint32_t n_vals;
    float   *vals;
    uint32_t n_raw;
    unsigned char *raw;
} Golden;

static Golden *g_cases = NULL;
static int     g_n = 0;

static int load_golden(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "GGREF2", 6) != 0) {
        fclose(f);
        return -1;
    }
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1 || n > 4096) { fclose(f); return -1; }

    g_cases = (Golden *)calloc(n, sizeof *g_cases);
    if (!g_cases) { fclose(f); return -1; }

    for (uint32_t i = 0; i < n; i++) {
        Golden *c = &g_cases[i];
        uint32_t nl = 0;
        if (fread(&nl, 4, 1, f) != 1 || nl >= sizeof c->name) { fclose(f); return -1; }
        if (fread(c->name, 1, nl, f) != nl) { fclose(f); return -1; }
        c->name[nl] = '\0';
        if (fread(&c->gtype, 4, 1, f) != 1) { fclose(f); return -1; }
        if (fread(&c->row, 8, 1, f) != 1) { fclose(f); return -1; }
        if (fread(&c->n_vals, 4, 1, f) != 1) { fclose(f); return -1; }
        if (fread(&c->n_raw, 4, 1, f) != 1) { fclose(f); return -1; }
        c->vals = (float *)malloc((size_t)c->n_vals * sizeof *c->vals);
        c->raw  = (unsigned char *)malloc(c->n_raw);
        if (!c->vals || !c->raw) { fclose(f); return -1; }
        if (fread(c->vals, sizeof *c->vals, c->n_vals, f) != c->n_vals) { fclose(f); return -1; }
        if (fread(c->raw, 1, c->n_raw, f) != c->n_raw) { fclose(f); return -1; }
        g_n++;
    }
    fclose(f);
    return 0;
}

static void free_golden(void)
{
    for (int i = 0; i < g_n; i++) { free(g_cases[i].vals); free(g_cases[i].raw); }
    free(g_cases);
}

/* -------------------------------------------- the kernels against the golden -- */

static void test_kquant_against_reference(void)
{
    printf("== k-quant kernels vs an independent numpy reference ==\n");

    if (g_n == 0) {
        ok(0, "golden fixture loaded", "tests/fixtures/gguf/golden.bin");
        return;
    }
    printf("  note  %d real rows from Qwen3-8B-Q4_K_M.gguf\n", g_n);

    int q4 = 0, q6 = 0;
    double worst_deq = 0.0, worst_dot = 0.0;
    const char *worst_deq_name = "", *worst_dot_name = "";
    int deq_bad = 0, dot_bad = 0, size_bad = 0;

    float *x = NULL;
    int xcap = 0;

    for (int i = 0; i < g_n; i++) {
        const Golden *c = &g_cases[i];
        const EngDType dt = gguf_dtype(c->gtype);
        const EngQuantOps *q = eng_quant_ops(dt);
        if (!q) { ok(0, "ops for golden case", c->name); continue; }
        if (dt == ENG_DT_Q4_K) q4++; else if (dt == ENG_DT_Q6_K) q6++;

        /* The raw byte count must agree with row_bytes: if it does not, the block-size
         * constant is wrong and everything after is meaningless. */
        if (q->row_bytes((int)c->n_vals) != (int64_t)c->n_raw) size_bad++;

        /* ---- dequant ---- */
        float *out = (float *)malloc((size_t)c->n_vals * sizeof *out);
        q->dequant_row(out, c->raw, NULL, (int)c->n_vals);

        double w = 0.0;
        for (uint32_t k = 0; k < c->n_vals; k++) {
            const double e = fabs((double)out[k] - (double)c->vals[k]);
            if (e > w) w = e;
        }
        if (w > worst_deq) { worst_deq = w; worst_deq_name = c->name; }
        /* Weights are order 0.03; an f32 ULP there is ~2e-9, and the two
         * implementations round their products in a different order, so a few ULPs is
         * expected. 1e-6 is far below anything a LAYOUT error could produce -- those
         * are wrong by the magnitude of the value itself. */
        if (w > 1e-6) deq_bad++;

        /* ---- fused dot, against the reference values ---- */
        if ((int)c->n_vals > xcap) {
            x = (float *)realloc(x, (size_t)c->n_vals * sizeof *x);
            xcap = (int)c->n_vals;
        }
        for (uint32_t k = 0; k < c->n_vals; k++) x[k] = frand();

        double want = 0.0;
        for (uint32_t k = 0; k < c->n_vals; k++) want += (double)c->vals[k] * (double)x[k];
        const double got = q->dot_row(c->raw, NULL, x, (int)c->n_vals);

        const double denom = fabs(want) > 1e-9 ? fabs(want) : 1.0;
        const double rel = fabs(got - want) / denom;
        if (rel > worst_dot) { worst_dot = rel; worst_dot_name = c->name; }
        if (rel > 1e-5) dot_bad++;

        free(out);
    }
    free(x);

    char d[192];
    snprintf(d, sizeof d, "%d q4_k rows, %d q6_k rows", q4, q6);
    ok(q4 > 0 && q6 > 0, "both k-quant formats covered", d);

    snprintf(d, sizeof d, "%d of %d rows differ", size_bad, g_n);
    ok(size_bad == 0, "row_bytes agrees with the reference block sizes", d);

    snprintf(d, sizeof d, "worst |err| %.3e on %s", worst_deq, worst_deq_name);
    ok(deq_bad == 0, "dequant matches the numpy reference", d);

    snprintf(d, sizeof d, "worst rel %.3e on %s", worst_dot, worst_dot_name);
    ok(dot_bad == 0, "fused dot matches a dot over reference values", d);
}

/* A deliberately WRONG decode must fail the same comparison. Without this, a test that
 * passes proves only that something ran -- not that the comparison has any power. */
static void test_reference_has_teeth(void)
{
    printf("\n== the comparison can actually fail ==\n");
    if (g_n == 0) { ok(0, "golden loaded", NULL); return; }

    const Golden *c = NULL;
    for (int i = 0; i < g_n; i++)
        if (gguf_dtype(g_cases[i].gtype) == ENG_DT_Q4_K) { c = &g_cases[i]; break; }
    if (!c) { ok(0, "a q4_k case exists", NULL); return; }

    const EngQuantOps *q = eng_quant_ops(ENG_DT_Q4_K);
    float *out = (float *)malloc((size_t)c->n_vals * sizeof *out);

    /* Swap the nibbles of every quant byte: the classic wrong-order mistake. The values
     * stay finite and in range, which is the point. */
    unsigned char *bad = (unsigned char *)malloc(c->n_raw);
    memcpy(bad, c->raw, c->n_raw);
    for (uint32_t b = 0; b + 144 <= c->n_raw; b += 144)
        for (int k = 16; k < 144; k++)
            bad[b + k] = (unsigned char)((bad[b + k] >> 4) | (bad[b + k] << 4));

    q->dequant_row(out, bad, NULL, (int)c->n_vals);
    double worst = 0.0;
    int finite = 1;
    for (uint32_t k = 0; k < c->n_vals; k++) {
        if (!isfinite(out[k])) finite = 0;
        const double e = fabs((double)out[k] - (double)c->vals[k]);
        if (e > worst) worst = e;
    }

    char d[128];
    snprintf(d, sizeof d, "still finite=%d, worst |err| %.4f", finite, worst);
    ok(finite && worst > 1e-3, "nibble-swapped blocks decode plausibly but WRONG", d);

    free(bad);
    free(out);
}

/* ----------------------------------------------------- the reader, real file -- */

static void test_reader(const char *path)
{
    printf("\n== gguf reader ==\n");

    Gguf g;
    if (gguf_open(&g, path) != 0) {
        ok(0, "open the model", path);
        return;
    }
    gguf_report(&g, "  ");

    /* Against the independent reference's structural facts. */
    eqi(g.version, 3, "gguf version");
    eqi(g.n_tensors, 399, "tensor count");
    eqi(g.n_kv, 28, "metadata key count");
    eqi(g.alignment, 32, "general.alignment");
    eqi(g.data_start, 5956416, "data section offset");
    eqi(g.file_size, 5027783488LL, "file size");

    /* THE STRUCTURAL PROOF. Every block-size constant in the dtype table feeds this
     * sum; if any one is wrong the total misses the file end. It closes exactly. */
    char d[128];
    snprintf(d, sizeof d, "%lld + %lld = %lld",
             (long long)g.data_start, (long long)g.data_bytes,
             (long long)(g.data_start + g.data_bytes));
    eqi(g.data_start + g.data_bytes, g.file_size,
        "data_start + tensor bytes closes the file exactly");
    printf("        %s\n", d);

    ok(gguf_layout_is_sequential(&g), "layout is sequential",
       "a layer is already one contiguous run, so no repacker is needed");

    /* Architecture metadata, typed. */
    const char *arch = NULL;
    int64_t alen = 0;
    ok(gguf_str(&g, "general.architecture", &arch, &alen) == 0 &&
       alen == 5 && !memcmp(arch, "qwen3", 5), "general.architecture is qwen3", NULL);

    int64_t v = 0;
    ok(gguf_i64(&g, "qwen3.block_count", &v) == 0 && v == 36, "block_count 36", NULL);
    ok(gguf_i64(&g, "qwen3.embedding_length", &v) == 0 && v == 4096,
       "embedding_length 4096", NULL);
    ok(gguf_i64(&g, "qwen3.attention.head_count", &v) == 0 && v == 32,
       "head_count 32", NULL);
    ok(gguf_i64(&g, "qwen3.attention.head_count_kv", &v) == 0 && v == 8,
       "head_count_kv 8 (GQA 4:1)", NULL);
    ok(gguf_i64(&g, "qwen3.context_length", &v) == 0 && v == 40960,
       "context_length 40960", NULL);

    float fv = 0.0f;
    ok(gguf_f32(&g, "qwen3.rope.freq_base", &fv) == 0 && fv == 1000000.0f,
       "rope freq_base 1e6", "not the 1e4 default");
    ok(gguf_f32(&g, "qwen3.attention.layer_norm_rms_epsilon", &fv) == 0 &&
       fv > 9e-7f && fv < 1.1e-6f, "rms eps 1e-6", "not K3's 1e-5");

    /* Missing keys must FAIL rather than default: a silently defaulted architecture
     * field is how a loader builds the wrong model. */
    ok(gguf_i64(&g, "qwen3.does_not_exist", &v) != 0, "absent key is an error", NULL);
    ok(gguf_str(&g, "qwen3.block_count", &arch, &alen) != 0,
       "wrong-type read is an error", "an int is not a string");

    /* Named tensors, against the reference's probe table. */
    const GgufTensor *t = gguf_tensor(&g, "token_embd.weight");
    ok(t != NULL, "token_embd.weight present", NULL);
    if (t) {
        eqi(t->file_off, 516477760, "token_embd file offset");
        eqi(t->nbytes, 350060544, "token_embd byte size");
        ok(t->dtype == ENG_DT_Q4_K, "token_embd is q4_k", eng_dtype_name(t->dtype));
        eqi(t->shape[0], 4096, "token_embd shape[0]");
        eqi(t->shape[1], 151936, "token_embd shape[1]");
    }

    t = gguf_tensor(&g, "output.weight");
    ok(t && t->dtype == ENG_DT_Q6_K, "output.weight is q6_k",
       "untied from token_embd, and a different quant");
    if (t) eqi(t->file_off, 5956416, "output.weight starts the data section");

    t = gguf_tensor(&g, "blk.0.attn_v.weight");
    ok(t && t->dtype == ENG_DT_Q6_K, "attn_v is q6_k",
       "the Q4_K_M mixture gives V six bits");

    t = gguf_tensor(&g, "blk.0.attn_q_norm.weight");
    ok(t != NULL, "attn_q_norm present", "the Qwen3 QK-norm signature");
    if (t) {
        ok(t->dtype == ENG_DT_F32, "attn_q_norm is f32", eng_dtype_name(t->dtype));
        eqi(t->shape[0], 128, "attn_q_norm is head_dim wide");
    }

    ok(gguf_tensor(&g, "nonexistent.weight") == NULL, "unknown tensor returns NULL", NULL);

    /* Read a real row through the reader and decode it, cross-checking the golden --
     * this is the end-to-end path: file offset, byte range, dtype, kernel. */
    t = gguf_tensor(&g, "token_embd.weight");
    if (t && g_n > 0) {
        const EngQuantOps *q = eng_quant_ops(ENG_DT_Q4_K);
        const int cols = (int)t->shape[0];
        const int64_t rb = q->row_bytes(cols);
        unsigned char *buf = (unsigned char *)malloc((size_t)rb);
        float *out = (float *)malloc((size_t)cols * sizeof *out);

        /* Golden row 100 of token_embd. */
        const Golden *c = NULL;
        for (int i = 0; i < g_n; i++)
            if (!strcmp(g_cases[i].name, "token_embd.weight") && g_cases[i].row == 100)
                c = &g_cases[i];

        if (c && g.store->read(g.store, t->file_off + 100 * rb, rb, buf) == rb) {
            ok(memcmp(buf, c->raw, (size_t)rb) == 0,
               "bytes read at the computed offset match the reference",
               "offset arithmetic is correct end to end");
            q->dequant_row(out, buf, NULL, cols);
            double w = 0.0;
            for (int k = 0; k < cols; k++)
                w = fmax(w, fabs((double)out[k] - (double)c->vals[k]));
            snprintf(d, sizeof d, "worst |err| %.3e", w);
            ok(w < 1e-6, "decoded row matches the reference", d);
        } else {
            ok(0, "read token_embd row 100", NULL);
        }
        free(buf);
        free(out);
    }

    gguf_close(&g);
}

/* ---------------------------------------------------------------- refusals -- */

static void test_refusals(const char *dir)
{
    printf("\n== malformed containers are refused ==\n");
    char path[512];
    Gguf g;

    snprintf(path, sizeof path, "%s/bad_magic.gguf", dir);
    FILE *f = fopen(path, "wb");
    if (f) {
        fwrite("NOPE\3\0\0\0", 1, 8, f);
        for (int i = 0; i < 16; i++) fputc(0, f);
        fclose(f);
        ok(gguf_open(&g, path) != 0, "bad magic refused", NULL);
        remove(path);
    }

    snprintf(path, sizeof path, "%s/bad_ver.gguf", dir);
    f = fopen(path, "wb");
    if (f) {
        fwrite("GGUF\x63\0\0\0", 1, 8, f);
        for (int i = 0; i < 16; i++) fputc(0, f);
        fclose(f);
        ok(gguf_open(&g, path) != 0, "unsupported version refused", "v99");
        remove(path);
    }

    snprintf(path, sizeof path, "%s/truncated.gguf", dir);
    f = fopen(path, "wb");
    if (f) {
        /* Claims 1000 tensors and 1000 kv, then ends. */
        fwrite("GGUF\3\0\0\0", 1, 8, f);
        const uint64_t n = 1000;
        fwrite(&n, 8, 1, f);
        fwrite(&n, 8, 1, f);
        fclose(f);
        ok(gguf_open(&g, path) != 0, "truncated header refused",
           "counts are treated as untrusted");
        remove(path);
    }

    ok(gguf_open(&g, "/definitely/not/here.gguf") != 0, "missing file refused", NULL);
}

int main(int argc, char **argv)
{
    const char *golden = argc > 1 ? argv[1] : "tests/fixtures/gguf/golden.bin";
    const char *tmpdir = argc > 2 ? argv[2] : "build";
    const char *model  = getenv("GGUF_MODEL");
    if (!model) model = "/mnt/c/Users/SHAMI/HAKIM/AI/Qwen3-8B-Q4_K_M.gguf";

    printf("GGUF container and k-quant kernels\n\n");

    if (load_golden(golden) != 0)
        printf("  WARNING: no golden fixture at %s\n", golden);

    test_kquant_against_reference();
    test_reference_has_teeth();
    test_refusals(tmpdir);

    FILE *probe = fopen(model, "rb");
    if (probe) {
        fclose(probe);
        test_reader(model);
    } else {
        printf("\n== gguf reader ==\n");
        printf("  SKIP  %s is not present.\n", model);
        printf("        The kernels above were still checked against real blocks;\n"
               "        set GGUF_MODEL to also exercise the reader.\n");
    }

    free_golden();
    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("GGUF TESTS FAILED\n"); return 1; }
    printf("GGUF TESTS PASSED\n");
    return 0;
}
