/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_kernels.c - runtime dispatch and the two numerical policies.
 *
 * THE CENTRAL CASE COMPARES RAW BITS. Under ENG_NUM_EXACT the AVX2 path must produce
 * the reference's exact bit pattern, because that is what backs K3's claim that its
 * output does not depend on how much memory it was given: a streamed layer and a
 * resident layer take different code paths through the cache but must agree exactly.
 *
 * A test that compared values to a tolerance would pass on an implementation that is
 * merely close -- and merely close is precisely the failure. So memcmp on the float,
 * not fabs on the difference.
 *
 * Lengths are chosen to exercise the tail: the exact kernels consume 16 elements per
 * vector step, so 16 and 32 have no remainder while 17, 31 and 4097 do, and the tail
 * loop has to fold in the same order as the reference or the bits diverge.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel.h"
#include "kernel_impl.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-46s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

/* Deterministic pseudo-random, so a failure is reproducible. */
static uint32_t rng_state = 12345u;
static float frand(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((double)(rng_state >> 8) / (double)(1u << 24) * 2.0 - 1.0);
}

static uint16_t f32_to_bf16(float f)
{
    union { float f; uint32_t u; } v; v.f = f;
    /* Round-to-nearest-even, which is what a real bf16 checkpoint contains. */
    const uint32_t r = v.u + 0x7fffu + ((v.u >> 16) & 1u);
    return (uint16_t)(r >> 16);
}

static int bits_equal(float a, float b)
{
    return memcmp(&a, &b, sizeof a) == 0;
}

/* ------------------------------------------------------------------ dispatch -- */

static void test_dispatch(void)
{
    printf("== dispatch ==\n");
    eng_kernel_report("  ");

    ok(eng_kernel_available(ENG_IMPL_REF), "reference always available", NULL);

    const EngImpl a = eng_kernel_select(ENG_IMPL_AUTO);
    printf("  note  auto selected %s (avx2 compiled=%d, cpu=%d)\n",
           eng_kernel_name(a), eng_avx2_compiled(), eng_cpu_has_avx2());

    if (eng_avx2_compiled() && eng_cpu_has_avx2())
        ok(a == ENG_IMPL_AVX2, "auto picks AVX2 when the CPU has it", NULL);
    else
        ok(a == ENG_IMPL_REF, "auto falls back to the reference", NULL);

    /* Forcing the reference must be honoured: it is how a bug is bisected. */
    ok(eng_kernel_select(ENG_IMPL_REF) == ENG_IMPL_REF, "reference can be forced", NULL);
    ok(eng_kernel_active() == ENG_IMPL_REF, "active reflects the choice", NULL);

    eng_kernel_select(ENG_IMPL_AUTO);
}

/* ------------------------------------------------------- the bit-identity gate -- */

static void test_exact_bit_identity(void)
{
    printf("\n== ENG_NUM_EXACT: AVX2 must equal the reference BIT FOR BIT ==\n");

    if (!(eng_avx2_compiled() && eng_cpu_has_avx2())) {
        printf("  SKIP  no AVX2 on this host; nothing to compare against\n");
        return;
    }

    /* 16 and 32 divide the vector step exactly; 17, 31, 63 and 4097 leave a tail that
     * must be folded in the reference's order. 4096 is Qwen3's hidden width. */
    static const int LENS[] = { 1, 7, 15, 16, 17, 31, 32, 63, 64, 127, 4096, 4097 };
    int f32_bad = 0, bf16_bad = 0, worst_len = 0;

    for (size_t k = 0; k < sizeof LENS / sizeof *LENS; k++) {
        const int n = LENS[k];
        float *a = (float *)malloc((size_t)n * sizeof *a);
        float *b = (float *)malloc((size_t)n * sizeof *b);
        uint16_t *w = (uint16_t *)malloc((size_t)n * sizeof *w);
        for (int i = 0; i < n; i++) {
            a[i] = frand() * 3.0f;
            b[i] = frand() * 3.0f;
            w[i] = f32_to_bf16(frand() * 2.0f);
        }

        const float r1 = (float)eng_ref_dot_f32_exact(a, b, n);
        const float v1 = (float)eng_avx2_dot_f32_exact(a, b, n);
        if (!bits_equal(r1, v1)) { f32_bad++; worst_len = n; }

        const float r2 = (float)eng_ref_dot_bf16_exact(a, w, n);
        const float v2 = (float)eng_avx2_dot_bf16_exact(a, w, n);
        if (!bits_equal(r2, v2)) { bf16_bad++; worst_len = n; }

        free(a); free(b); free(w);
    }

    char d[128];
    snprintf(d, sizeof d, "%zu lengths incl. tails", sizeof LENS / sizeof *LENS);
    ok(f32_bad == 0, "f32 exact: every length bit-identical", d);
    ok(bf16_bad == 0, "bf16 exact: every length bit-identical", d);
    if (f32_bad || bf16_bad) {
        snprintf(d, sizeof d, "first divergence at n=%d", worst_len);
        printf("        %s\n", d);
    }

    /* And the double accumulators must be identical too, not just the float they
     * narrow to -- a float result can hide a difference in the last few bits. */
    {
        const int n = 4096;
        float *a = (float *)malloc((size_t)n * sizeof *a);
        float *b = (float *)malloc((size_t)n * sizeof *b);
        for (int i = 0; i < n; i++) { a[i] = frand(); b[i] = frand(); }
        const double r = eng_ref_dot_f32_exact(a, b, n);
        const double v = eng_avx2_dot_f32_exact(a, b, n);
        ok(memcmp(&r, &v, sizeof r) == 0,
           "f32 exact: the DOUBLE accumulator matches too",
           "narrowing to float could mask a low-bit difference");
        free(a); free(b);
    }
}

/* ------------------------------------------------------------ the fast policy -- */

static void test_fast_policy(void)
{
    printf("\n== ENG_NUM_FAST: within tolerance, not bit-identical ==\n");

    const int n = 4096;
    float *a = (float *)malloc((size_t)n * sizeof *a);
    float *b = (float *)malloc((size_t)n * sizeof *b);
    for (int i = 0; i < n; i++) { a[i] = frand(); b[i] = frand(); }

    const double exact = eng_ref_dot_f32_exact(a, b, n);
    const float  fast  = eng_ref_dot_f32_fast(a, b, n);

    /* fp32 accumulation over 4096 terms. Relative error grows roughly with sqrt(n) for
     * random signs, so 1e-4 is generous but comfortably tight enough to catch a wrong
     * kernel rather than a rounding difference. */
    const double rel = fabs((double)fast - exact) / (fabs(exact) + 1e-9);
    char d[96];
    snprintf(d, sizeof d, "rel err %.2e over n=%d", rel, n);
    ok(rel < 1e-4, "fast agrees with exact within tolerance", d);

    if (eng_avx2_compiled() && eng_cpu_has_avx2()) {
        const float vfast = eng_avx2_dot_f32_fast(a, b, n);
        const double rel2 = fabs((double)vfast - exact) / (fabs(exact) + 1e-9);
        snprintf(d, sizeof d, "rel err %.2e", rel2);
        ok(rel2 < 1e-4, "avx2 fast agrees with exact within tolerance", d);
        /* Not required to be bit-identical -- that is the whole point of the policy. */
    }
    free(a); free(b);
}

/* -------------------------------------------------------------------- matmul -- */

static void test_matmul(void)
{
    printf("\n== matmul ==\n");
    const int in = 128, out = 96;
    float *W = (float *)malloc((size_t)in * out * sizeof *W);
    float *x = (float *)malloc((size_t)in * sizeof *x);
    float *y = (float *)malloc((size_t)out * sizeof *y);
    uint16_t *Wb = (uint16_t *)malloc((size_t)in * out * sizeof *Wb);

    for (int i = 0; i < in * out; i++) { W[i] = frand(); Wb[i] = f32_to_bf16(W[i]); }
    for (int i = 0; i < in; i++) x[i] = frand();

    eng_matmul_f32(y, x, W, in, out, ENG_NUM_EXACT);
    int bad = 0;
    for (int o = 0; o < out; o++) {
        const float want = (float)eng_ref_dot_f32_exact(W + (size_t)o * in, x, in);
        if (!bits_equal(y[o], want)) bad++;
    }
    ok(bad == 0, "matmul_f32 rows equal the reference dot", "bit for bit");

    eng_matmul_bf16(y, x, Wb, in, out, ENG_NUM_EXACT);
    bad = 0;
    for (int o = 0; o < out; o++) {
        const float want = (float)eng_ref_dot_bf16_exact(x, Wb + (size_t)o * in, in);
        if (!bits_equal(y[o], want)) bad++;
    }
    ok(bad == 0, "matmul_bf16 rows equal the reference dot", "bit for bit");

    /* bf16 keeps 8 mantissa bits, so a bf16 matmul must track the f32 one loosely but
     * not exactly -- if it matched exactly, the weights were not actually rounded.
     *
     * THE ERROR IS NORMALISED BY THE RMS OF THE OUTPUT, not taken per row. A dot
     * product of random-signed terms can land arbitrarily close to zero through
     * cancellation, and the RELATIVE error on such a row is unbounded no matter how
     * accurate the kernel is -- it says something about the input, not the arithmetic.
     * The first version of this check used per-row relative error and failed at 6.35e-2
     * on a correct kernel for exactly that reason. Normalising by the output's own
     * scale is the measure that actually tracks precision. */
    float *yf = (float *)malloc((size_t)out * sizeof *yf);
    eng_matmul_f32(yf, x, W, in, out, ENG_NUM_EXACT);
    eng_matmul_bf16(y, x, Wb, in, out, ENG_NUM_EXACT);

    double worst_abs = 0.0, ss = 0.0;
    int identical = 1;
    for (int o = 0; o < out; o++) {
        const double e = fabs((double)y[o] - (double)yf[o]);
        if (e > worst_abs) worst_abs = e;
        ss += (double)yf[o] * (double)yf[o];
        if (!bits_equal(y[o], yf[o])) identical = 0;
    }
    const double rms = sqrt(ss / (double)out);
    const double norm_err = worst_abs / (rms + 1e-12);

    char d[128];
    snprintf(d, sizeof d, "worst |err| %.2e vs output rms %.2e -> %.2e", worst_abs, rms, norm_err);
    /* bf16 carries ~2^-8 = 3.9e-3 relative per weight; over `in` terms with random
     * signs the error grows as sqrt(in), so 1e-2 of the output scale is the right order
     * and tight enough to catch a genuinely wrong kernel. */
    ok(norm_err < 1e-2, "bf16 tracks f32 within bf16 precision", d);
    ok(!identical, "bf16 is NOT identical to f32", "otherwise the rounding did nothing");

    free(W); free(x); free(y); free(Wb); free(yf);
}

/* --------------------------------------------------------------- elementwise -- */

static void test_elementwise(void)
{
    printf("\n== elementwise ==\n");

    /* RMSNorm against a hand-computed value. */
    {
        float x[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
        float w[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        float y[4];
        eng_rmsnorm(y, x, w, 4, 0.0f);
        /* mean(x^2) = (1+4+9+16)/4 = 7.5; 1/sqrt(7.5) = 0.3651483717 */
        const double s = 1.0 / sqrt(7.5);
        double worst = 0.0;
        for (int i = 0; i < 4; i++)
            worst = fmax(worst, fabs((double)y[i] - x[i] * s));
        char d[64]; snprintf(d, sizeof d, "max err %.2e", worst);
        ok(worst < 1e-6, "rmsnorm matches the hand computation", d);
    }

    /* eps is INSIDE the rsqrt: with an all-zero input and eps>0 the result must be
     * finite zero, not NaN. */
    {
        float x[4] = { 0, 0, 0, 0 }, w[4] = { 1, 1, 1, 1 }, y[4];
        eng_rmsnorm(y, x, w, 4, 1e-6f);
        int finite = 1;
        for (int i = 0; i < 4; i++) if (!isfinite(y[i])) finite = 0;
        ok(finite, "rmsnorm of zeros is finite", "eps inside the rsqrt");
    }

    /* Softmax: sums to 1, and survives values that would overflow expf without the
     * max subtraction. */
    {
        float x[5] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        eng_softmax(x, 5);
        double sum = 0.0;
        for (int i = 0; i < 5; i++) sum += x[i];
        ok(fabs(sum - 1.0) < 1e-6, "softmax sums to 1", NULL);
        ok(x[4] > x[3] && x[3] > x[2], "softmax preserves order", NULL);

        float big[3] = { 1000.0f, 1001.0f, 1002.0f };
        eng_softmax(big, 3);
        int finite = 1; double s2 = 0.0;
        for (int i = 0; i < 3; i++) { if (!isfinite(big[i])) finite = 0; s2 += big[i]; }
        ok(finite && fabs(s2 - 1.0) < 1e-6, "softmax survives large inputs",
           "max subtraction, not naive expf");
    }

    /* SiLU at known points: silu(0)=0, silu(x)->x for large x, ->0 for very negative. */
    {
        float x[4] = { 0.0f, 20.0f, -20.0f, 1.0f }, y[4];
        eng_silu(y, x, 4);
        ok(fabs(y[0]) < 1e-9, "silu(0) = 0", NULL);
        ok(fabs(y[1] - 20.0f) < 1e-4, "silu(20) ~ 20", NULL);
        ok(fabs(y[2]) < 1e-6, "silu(-20) ~ 0", NULL);
        ok(fabs(y[3] - 0.7310586f) < 1e-6, "silu(1) = 0.7310586", NULL);
    }

    /* SwiGLU reads [gate | up] and must not confuse the halves: with up all ones the
     * result is exactly silu(gate), which pins the layout. */
    {
        float x[8] = { 1.0f, 2.0f, -1.0f, 0.0f,   1.0f, 1.0f, 1.0f, 1.0f };
        float y[4], s[4];
        eng_swiglu(y, x, 4);
        eng_silu(s, x, 4);
        int same = 1;
        for (int i = 0; i < 4; i++) if (fabs(y[i] - s[i]) > 1e-6f) same = 0;
        ok(same, "swiglu([g|1]) == silu(g)", "pins which half is the gate");

        /* And with gate all ones it is silu(1) * up, which pins the other half. */
        float x2[8] = { 1.0f, 1.0f, 1.0f, 1.0f,   2.0f, 3.0f, 4.0f, 5.0f };
        eng_swiglu(y, x2, 4);
        const float k = 0.7310586f;
        int scaled = 1;
        for (int i = 0; i < 4; i++)
            if (fabs(y[i] - k * x2[4 + i]) > 1e-5f) scaled = 0;
        ok(scaled, "swiglu([1|u]) == silu(1) * u", NULL);
    }
}

/* ---------------------------------------------------------------------- rope -- */

static void test_rope(void)
{
    printf("\n== rope ==\n");
    const int dim = 128;

    /* Position 0 is the identity: every angle is zero. */
    {
        float v[128], orig[128];
        for (int i = 0; i < dim; i++) v[i] = orig[i] = frand();
        eng_rope(v, dim, 0, 1000000.0f, ENG_ROPE_HALVED);
        double worst = 0.0;
        for (int i = 0; i < dim; i++) worst = fmax(worst, fabs(v[i] - orig[i]));
        char d[64]; snprintf(d, sizeof d, "max err %.2e", worst);
        ok(worst < 1e-6, "rope at position 0 is the identity", d);
    }

    /* A rotation preserves the norm of each pair, whichever pairing is used. That is
     * the invariant a wrong angle or a wrong sign would break. */
    {
        float v[128], orig[128];
        for (int i = 0; i < dim; i++) v[i] = orig[i] = frand();
        eng_rope(v, dim, 137, 1000000.0f, ENG_ROPE_HALVED);
        const int half = dim / 2;
        double worst = 0.0;
        for (int i = 0; i < half; i++) {
            const double n0 = (double)orig[i] * orig[i] + (double)orig[i + half] * orig[i + half];
            const double n1 = (double)v[i] * v[i] + (double)v[i + half] * v[i + half];
            worst = fmax(worst, fabs(n1 - n0));
        }
        char d[64]; snprintf(d, sizeof d, "max pair-norm drift %.2e", worst);
        ok(worst < 1e-5, "rope preserves each pair's norm", d);
    }

    /* THE PAIRING CONVENTION MATTERS. Both styles are rotations, so both preserve
     * norms -- but they produce DIFFERENT vectors from the same weights. Qwen3 needs
     * HALVED (what GGUF and the HF implementation use); picking the other gives a model
     * that runs and is wrong, exactly like K3's MXFP4 nibble order. */
    {
        float a[128], b[128];
        for (int i = 0; i < dim; i++) a[i] = b[i] = frand();
        eng_rope(a, dim, 42, 1000000.0f, ENG_ROPE_HALVED);
        eng_rope(b, dim, 42, 1000000.0f, ENG_ROPE_ADJACENT);
        int differ = 0;
        for (int i = 0; i < dim; i++) if (fabs(a[i] - b[i]) > 1e-6f) { differ = 1; break; }
        ok(differ, "the two pairing conventions give different results",
           "so choosing wrong is detectable, not silent");
    }

    /* theta_base matters too: 1e6 (Qwen3) must not equal 1e4 (the common default). */
    {
        float a[128], b[128];
        for (int i = 0; i < dim; i++) a[i] = b[i] = frand();
        eng_rope(a, dim, 42, 1000000.0f, ENG_ROPE_HALVED);
        eng_rope(b, dim, 42, 10000.0f, ENG_ROPE_HALVED);
        int differ = 0;
        for (int i = 0; i < dim; i++) if (fabs(a[i] - b[i]) > 1e-6f) { differ = 1; break; }
        ok(differ, "theta base 1e6 differs from 1e4", "Qwen3 uses 1e6");
    }
}

int main(void)
{
    printf("CPU kernels: runtime dispatch and numerical policy\n\n");
    test_dispatch();
    test_exact_bit_identity();
    test_fast_policy();
    test_matmul();
    test_elementwise();
    test_rope();

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("KERNEL TESTS FAILED\n"); return 1; }
    printf("KERNEL TESTS PASSED\n");
    return 0;
}
