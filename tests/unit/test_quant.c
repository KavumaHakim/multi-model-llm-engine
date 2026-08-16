/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_quant.c - the quantization vtable and its registered formats.
 *
 * THREE KINDS OF CHECK HERE, and the split matters:
 *
 *   1. REGISTRY-DRIVEN. The accuracy contract is checked by ITERATING the registry, so
 *      a format added later is covered automatically rather than by a hand-maintained
 *      list that drifts out of date. Each format declares its own tolerance and is held
 *      to that figure, not to a shared constant that would have to be loosened to the
 *      worst format.
 *
 *   2. FORMAT-SPECIFIC BIT LAYOUT. Generic checks cannot catch a wrong nibble order:
 *      swapping the two halves of every byte still decodes to plausible weights, still
 *      runs, and still emits fluent text. It is the same silent-wrongness class as
 *      RoPE's pairing convention. So MXFP4's byte layout is pinned against hand-computed
 *      values.
 *
 *   3. CROSS-CHECK AGAINST THE DTYPE TABLE. row_bytes() and the dtype registry's size
 *      arithmetic are two independent statements of the same fact. If they disagree, one
 *      of them is wrong and every offset computed from it is wrong too.
 *
 * Constructing VALID packed bytes needs format knowledge -- random bytes would produce
 * NaN scales for an interleaved format -- so the data builders switch on dtype while the
 * contract checks stay generic. That asymmetry is deliberate and is the honest split.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dtype.h"
#include "quant.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-48s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

static uint32_t rs = 987654321u;
static float frand(void)
{
    rs = rs * 1664525u + 1013904223u;
    return (float)((double)(rs >> 8) / (double)(1u << 24) * 2.0 - 1.0);
}
static unsigned char brand(void)
{
    rs = rs * 1664525u + 1013904223u;
    return (unsigned char)(rs >> 16);
}

/* Minimal f32 -> f16, good enough for the normal range q8_0 scales live in. */
static uint16_t f32_to_f16(float f)
{
    union { float f; uint32_t u; } v; v.f = f;
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    int exp = (int)((v.u >> 23) & 0xFF) - 127 + 15;
    uint32_t man = v.u & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

/* ------------------------------------------------------------------ registry -- */

static void test_registry(void)
{
    printf("== registry ==\n");

    const int n = eng_quant_count();
    char d[96];
    snprintf(d, sizeof d, "%d format(s) registered", n);
    ok(n >= 2, "built-in formats self-register on first lookup", d);

    for (int i = 0; i < n; i++) {
        const EngQuantOps *q = eng_quant_at(i);
        if (!q) { ok(0, "eng_quant_at returned NULL inside range", NULL); continue; }
        snprintf(d, sizeof d, "%s (group %u, tol %.0e)", q->name, q->group,
                 q->dequant_tolerance);
        ok(q->dot_row && q->dequant_row && q->row_bytes && q->scale_bytes,
           "registered format is complete", d);
    }

    ok(eng_quant_ops(ENG_DT_MXFP4) != NULL, "mxfp4 is registered", NULL);
    ok(eng_quant_ops(ENG_DT_Q8_0)  != NULL, "q8_0 is registered", NULL);

    /* A dtype with no implementation must return NULL, not a default. There is no
     * sensible way to "decode these bytes somehow". */
    ok(eng_quant_ops(ENG_DT_F32)     == NULL, "unquantized dtype has no ops", NULL);
    ok(eng_quant_ops(ENG_DT_Q4_K)    == NULL, "unimplemented quant has no ops",
       "q4_k arrives with the GGUF loader");
    ok(eng_quant_ops(ENG_DT_INVALID) == NULL, "invalid dtype has no ops", NULL);

    /* Registering over an existing format must be refused rather than silently
     * shadowing it: two implementations of one dtype is a build mistake. */
    const EngQuantOps *mx = eng_quant_ops(ENG_DT_MXFP4);
    ok(eng_quant_register(mx) == -1, "re-registering a dtype is refused", NULL);

    EngQuantOps bad;
    memset(&bad, 0, sizeof bad);
    bad.dtype = ENG_DT_Q6_K;
    bad.name = "broken";
    ok(eng_quant_register(&bad) == -1, "incomplete ops are refused", "no dot_row");
}

/* -------------------------------------------------- MXFP4 bit layout, pinned -- */

static void test_mxfp4_layout(void)
{
    printf("\n== mxfp4 bit layout ==\n");
    const EngQuantOps *q = eng_quant_ops(ENG_DT_MXFP4);
    if (!q) { ok(0, "mxfp4 ops present", NULL); return; }

    /* THE NIBBLE ORDER. Low nibble is the EVEN element, high nibble the ODD one.
     * Swapping them decodes to equally plausible weights and the model still produces
     * fluent output, so nothing downstream would catch it. Pinned here against values
     * computed by hand from the OCP MX E2M1 table:
     *
     *   code 0..7  = +0, 0.5, 1, 1.5, 2, 3, 4, 6
     *   code 8..15 = the same with the sign bit set
     */
    {
        unsigned char w[16];
        memset(w, 0, sizeof w);
        w[0] = 0x21;   /* low = 1 -> 0.5 (elem 0);  high = 2 -> 1.0 (elem 1) */
        w[1] = 0x9F;   /* low = 15 -> -6.0 (elem 2); high = 9 -> -0.5 (elem 3) */
        unsigned char s = 127;      /* E8M0 127 -> 2^0 = 1.0 */

        float out[32];
        q->dequant_row(out, w, &s, 32);

        char d[128];
        snprintf(d, sizeof d, "[%.2f %.2f %.2f %.2f]", out[0], out[1], out[2], out[3]);
        ok(out[0] == 0.5f && out[1] == 1.0f && out[2] == -6.0f && out[3] == -0.5f,
           "low nibble is the EVEN element", d);
    }

    /* The full E2M1 table, both signs, at scale 1. */
    {
        unsigned char w[16];
        for (int i = 0; i < 8; i++) w[i] = (unsigned char)((2 * i + 1) << 4 | (2 * i));
        unsigned char s = 127;
        float out[32];
        q->dequant_row(out, w, &s, 32);
        static const float WANT[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
        };
        int bad = 0;
        for (int i = 0; i < 16; i++) if (out[i] != WANT[i]) bad++;
        char d[64]; snprintf(d, sizeof d, "%d of 16 codes wrong", bad);
        ok(bad == 0, "E2M1 code table matches the OCP MX spec", d);
    }

    /* E8M0 255 IS NaN by the spec, not exponent 128. Mapping it to a huge power of two
     * lets one bad byte poison a whole group with Inf, and the failure then surfaces
     * hundreds of layers away with nothing pointing back here. */
    {
        unsigned char w[16];
        memset(w, 0x77, sizeof w);   /* every element 6.0 */
        unsigned char s = 255;
        float out[32];
        q->dequant_row(out, w, &s, 32);
        int allzero = 1, finite = 1;
        for (int i = 0; i < 32; i++) {
            if (out[i] != 0.0f) allzero = 0;
            if (!isfinite(out[i])) finite = 0;
        }
        ok(allzero && finite, "E8M0 255 (NaN by spec) zeroes its group",
           "not exponent 128");
    }

    /* Scale really scales: byte 128 is 2^1. */
    {
        unsigned char w[16];
        memset(w, 0, sizeof w);
        w[0] = 0x02;                    /* elem 0 = code 2 = 1.0 */
        unsigned char s = 128;          /* 2^(128-127) = 2 */
        float out[32];
        q->dequant_row(out, w, &s, 32);
        char d[64]; snprintf(d, sizeof d, "got %.3f want 2.000", out[0]);
        ok(out[0] == 2.0f, "E8M0 exponent applies to the group", d);
    }

    /* Two groups must use their OWN scales, not the first one for everything. */
    {
        unsigned char w[32];
        memset(w, 0, sizeof w);
        w[0]  = 0x02;                   /* group 0, elem 0 = 1.0 */
        w[16] = 0x02;                   /* group 1, elem 0 = 1.0 */
        unsigned char s[2] = { 127, 129 };   /* 2^0 and 2^2 */
        float out[64];
        q->dequant_row(out, w, s, 64);
        char d[80]; snprintf(d, sizeof d, "g0=%.2f g1=%.2f", out[0], out[32]);
        ok(out[0] == 1.0f && out[32] == 4.0f, "each group uses its own scale", d);
    }
}

/* ------------------------------------------- the accuracy contract, generic -- */

/* Build valid packed bytes for one row. Format-specific by necessity: random bytes
 * would give an interleaved format a NaN scale and make the comparison meaningless. */
static int build_row(EngDType dt, int n, unsigned char **w, unsigned char **s)
{
    const EngQuantOps *q = eng_quant_ops(dt);
    if (!q) return -1;
    const int64_t rb = q->row_bytes(n);
    const int64_t sb = q->scale_bytes(n);
    if (rb <= 0) return -1;

    *w = (unsigned char *)calloc((size_t)rb, 1);
    *s = sb > 0 ? (unsigned char *)calloc((size_t)sb, 1) : NULL;
    if (!*w || (sb > 0 && !*s)) return -1;

    if (dt == ENG_DT_MXFP4) {
        for (int64_t i = 0; i < rb; i++) (*w)[i] = brand();
        /* Exponents near the middle of the range, and never 255 (NaN by spec). */
        for (int64_t i = 0; i < sb; i++) (*s)[i] = (unsigned char)(120 + (brand() % 12));
    } else if (dt == ENG_DT_Q8_0) {
        /* 34-byte blocks: f16 scale then 32 int8. Write a sane scale per block. */
        const int64_t nblk = rb / 34;
        for (int64_t b = 0; b < nblk; b++) {
            unsigned char *blk = *w + b * 34;
            const uint16_t h = f32_to_f16(0.01f + 0.001f * (float)(b % 7));
            memcpy(blk, &h, 2);
            for (int j = 0; j < 32; j++) blk[2 + j] = brand();
        }
    } else {
        /* Unknown format: bytes are opaque, so this test cannot build valid data for
         * it. Say so rather than fabricate. */
        free(*w); free(*s);
        *w = *s = NULL;
        return -1;
    }
    return 0;
}

static void test_accuracy_contract(void)
{
    printf("\n== accuracy contract (registry-driven) ==\n");

    const int n = eng_quant_count();
    for (int i = 0; i < n; i++) {
        const EngQuantOps *q = eng_quant_at(i);
        if (!q) continue;

        /* A whole number of groups, and more than one, so the group loop and the
         * per-group scale factoring are both exercised. */
        const int N = (int)q->group * 8;

        unsigned char *w = NULL, *s = NULL;
        if (build_row(q->dtype, N, &w, &s) != 0) {
            char d[96];
            snprintf(d, sizeof d, "%s: no data builder in this test", q->name);
            ok(0, "accuracy contract checked", d);
            continue;
        }

        float *x = (float *)malloc((size_t)N * sizeof *x);
        float *ref = (float *)malloc((size_t)N * sizeof *ref);
        for (int j = 0; j < N; j++) x[j] = frand();

        /* The two paths: fused, and materialise-then-dot. They are NOT required to be
         * bit-identical -- the fused kernel factors each group's scale out and applies
         * it once, so the summation order differs -- only to agree within the format's
         * declared tolerance. */
        const double fused = q->dot_row(w, s, x, N);

        q->dequant_row(ref, w, s, N);
        double materialised = 0.0;
        for (int j = 0; j < N; j++) materialised += (double)ref[j] * (double)x[j];

        const double denom = fabs(materialised) > 1e-12 ? fabs(materialised) : 1.0;
        const double rel = fabs(fused - materialised) / denom;

        char d[160];
        snprintf(d, sizeof d, "%s: rel %.2e vs declared %.0e (n=%d)",
                 q->name, rel, q->dequant_tolerance, N);
        ok(rel <= q->dequant_tolerance, "fused dot matches materialised within tolerance", d);

        /* And it must not be trivially zero, which would make the check vacuous. */
        snprintf(d, sizeof d, "%s: |dot| = %.4g", q->name, fabs(fused));
        ok(fabs(fused) > 1e-9, "the test row produces a non-trivial dot", d);

        free(w); free(s); free(x); free(ref);
    }
}

/* ------------------------------------------ cross-check against the dtype table -- */

static void test_size_agreement(void)
{
    printf("\n== row_bytes agrees with the dtype registry ==\n");

    const int n = eng_quant_count();
    for (int i = 0; i < n; i++) {
        const EngQuantOps *q = eng_quant_at(i);
        if (!q) continue;
        const int N = (int)q->group * 4;

        /* Two independent statements of the same fact. A disagreement means every
         * tensor offset derived from one of them is wrong. */
        const int64_t from_ops   = q->row_bytes(N);
        const int64_t from_dtype = eng_dtype_row_bytes(q->dtype, N);

        char d[128];
        snprintf(d, sizeof d, "%s: ops %lld, dtype table %lld (n=%d)",
                 q->name, (long long)from_ops, (long long)from_dtype, N);
        ok(from_ops == from_dtype, "row_bytes matches the dtype table", d);

        /* The declared group must match the dtype's external-scale group where the
         * dtype has one. */
        if (eng_dtype_has_ext_scales(q->dtype)) {
            const uint32_t g = eng_dtype_scale_group(q->dtype);
            snprintf(d, sizeof d, "%s: ops %u, dtype %u", q->name, q->group, g);
            ok(q->group == g, "scale group matches the dtype table", d);
            snprintf(d, sizeof d, "%s: %lld bytes for %d elems",
                     q->name, (long long)q->scale_bytes(N), N);
            ok(q->scale_bytes(N) == (N + (int)q->group - 1) / (int)q->group,
               "scale_bytes is one byte per group", d);
        } else {
            snprintf(d, sizeof d, "%s: scales are interleaved", q->name);
            ok(q->scale_bytes(N) == 0, "interleaved format reports 0 scale bytes", d);
        }
    }
}

/* --------------------------------------------------------------- the matmul -- */

static void test_matmul(void)
{
    printf("\n== eng_matmul_quant ==\n");
    const EngQuantOps *q = eng_quant_ops(ENG_DT_MXFP4);
    if (!q) { ok(0, "mxfp4 ops present", NULL); return; }

    const int in = 128, rows = 24;
    const int64_t rb = q->row_bytes(in), sb = q->scale_bytes(in);
    unsigned char *W = (unsigned char *)malloc((size_t)rb * rows);
    unsigned char *S = (unsigned char *)malloc((size_t)sb * rows);
    float *x = (float *)malloc((size_t)in * sizeof *x);
    float *y = (float *)malloc((size_t)rows * sizeof *y);

    for (int64_t i = 0; i < rb * rows; i++) W[i] = brand();
    for (int64_t i = 0; i < sb * rows; i++) S[i] = (unsigned char)(120 + (brand() % 12));
    for (int i = 0; i < in; i++) x[i] = frand();

    ok(eng_matmul_quant(y, x, W, S, in, rows, ENG_DT_MXFP4) == 0,
       "matmul over packed weights succeeds", NULL);

    /* Every row must equal the single-row entry point exactly: the matmul is a loop
     * over dot_row and must not reassociate anything. */
    int bad = 0;
    for (int r = 0; r < rows; r++) {
        const float want = (float)q->dot_row(W + (size_t)r * rb, S + (size_t)r * sb, x, in);
        if (memcmp(&y[r], &want, sizeof want) != 0) bad++;
    }
    char d[64]; snprintf(d, sizeof d, "%d of %d rows differ", bad, rows);
    ok(bad == 0, "matmul rows are bit-identical to dot_row", d);

    /* An external-scale format handed NULL scales would otherwise read from NULL+offset
     * and produce plausible garbage. */
    ok(eng_matmul_quant(y, x, W, NULL, in, rows, ENG_DT_MXFP4) == -1,
       "refuses NULL scales for an external-scale format", NULL);

    ok(eng_matmul_quant(y, x, W, S, in, rows, ENG_DT_Q4_K) == -1,
       "refuses a dtype with no implementation", "q4_k");

    free(W); free(S); free(x); free(y);
}

int main(void)
{
    printf("quantization: registry, layout, and the accuracy contract\n\n");
    test_registry();
    test_mxfp4_layout();
    test_accuracy_contract();
    test_size_agreement();
    test_matmul();

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("QUANT TESTS FAILED\n"); return 1; }
    printf("QUANT TESTS PASSED\n");
    return 0;
}
