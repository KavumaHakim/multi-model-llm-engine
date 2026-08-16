/* SPDX-License-Identifier: Apache-2.0 */
/*
 * mxfp4.c - OCP MX FP4: E2M1 nibbles with a shared E8M0 scale per 32 elements.
 *
 * MOVED FROM src/core/k3_ops.c, numerics unchanged. MXFP4 is an OCP standard rather
 * than anything of K3's -- gpt-oss ships in it too -- so it belongs in the quantization
 * layer and not in a model backend. The arithmetic below is the same operation in the
 * same order as the original; the determinism gate in scripts/verify.sh (the mxfp4
 * FNV1a hash a231061237b5579d) is what holds that claim to account.
 *
 * The one change is that the AVX2 path is now selected at RUNTIME through a target
 * attribute rather than at compile time through #if defined(__AVX2__), matching the
 * rest of the kernels after M5. That is safe here precisely because the two paths were
 * already written to be bit-identical to each other -- same lane partition, same
 * reduction order, same IEEE fused multiply-add -- so which one runs cannot change the
 * result, only the speed.
 *
 * TWO THINGS ARE EASY TO GET WRONG AND SILENT WHEN YOU DO
 *
 *   NIBBLE ORDER. The LOW nibble is the EVEN element and the high nibble is the odd
 *   one. Reversing it yields the right values in the wrong places, which preserves every
 *   statistic a sanity check might look at -- mean, variance, histogram, norm -- while
 *   producing a model that is simply wrong.
 *
 *   E8M0 255 IS NaN by the spec, not exponent 128. Mapping it to a huge power of two
 *   instead of zero lets one bad byte poison a whole group, and the failure surfaces
 *   thousands of tokens later as an occasional garbage logit.
 */
#include "quant.h"

#include <math.h>
#include <pthread.h>
#include <stddef.h>

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define MXFP4_X86 1
#else
#  define MXFP4_X86 0
#endif

int eng_cpu_has_avx2(void);   /* kernels/kernel_impl.h; declared to avoid the dependency */

#define MXFP4_GROUP 32

/* OCP MX E2M1: index by the 4-bit code; bit 3 is the sign. */
static const float E2M1[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* A whole BYTE to its two E2M1 values, so the inner loop does one 8-byte load instead
 * of masking, shifting and two separate lookups. 2 KB, built once. */
static float E2M1_PAIR[256][2];
/* E8M0 byte to its power of two. Precomputed because ldexpf in the group loop is a
 * function call the compiler will not inline into a vectorised body. */
static float E8M0[256];

static pthread_once_t tables_once = PTHREAD_ONCE_INIT;

static void build_tables(void)
{
    for (int b = 0; b < 256; b++) {
        E2M1_PAIR[b][0] = E2M1[b & 0x0F];   /* low nibble  = EVEN element */
        E2M1_PAIR[b][1] = E2M1[b >> 4];     /* high nibble = ODD element  */
        E8M0[b] = (b == 255) ? 0.0f : ldexpf(1.0f, b - 127);
    }
}

/* Public so the K3 backend can warm the tables before entering a parallel region.
 * pthread_once makes that unnecessary for correctness; it is here so the first token
 * does not pay for it inside a timed section. */
void eng_mxfp4_init(void) { pthread_once(&tables_once, build_tables); }

/* ------------------------------------------------------- the group inner loop -- */
/*
 * Four double lanes partitioned by i%4, reduced as (s0+s1)+(s2+s3), in BOTH paths so
 * they agree bit for bit on every machine. The split is written out rather than left to
 * the compiler because a sequential floating-point reduction may not be reassociated
 * without -ffast-math, which this build does not set: expressed as one serial
 * accumulator, the hottest loop in the engine compiles to scalar adds regardless of
 * what surrounds it.
 */
static double group_dot_ref(const float *wf, const float *xg, int n)
{
    double s[8] = { 0 };
    int i = 0;
    for (; i + 7 < n; i += 8)
        for (int l = 0; l < 8; l++)
            s[l] = fma((double)wf[i + l], (double)xg[i + l], s[l]);
    const double b0 = s[0] + s[4], b1 = s[1] + s[5];
    const double b2 = s[2] + s[6], b3 = s[3] + s[7];
    double sub = (b0 + b1) + (b2 + b3);
    for (; i < n; i++) sub = fma((double)wf[i], (double)xg[i], sub);
    return sub;
}

#if MXFP4_X86
__attribute__((target("avx2,fma")))
static double group_dot_avx2(const float *wf, const float *xg, int n)
{
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(wf + i)),
                             _mm256_cvtps_pd(_mm_loadu_ps(xg + i)), v0);
        v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(wf + i + 4)),
                             _mm256_cvtps_pd(_mm_loadu_ps(xg + i + 4)), v1);
    }
    /* v0 lane j holds elements i where (i/4)%2 == 0, matching s[j] and s[j+4] of the
     * reference once they are paired; (v0+v1) lane j IS the reference's b_j. */
    double a[4];
    _mm256_storeu_pd(a, _mm256_add_pd(v0, v1));
    double sub = (a[0] + a[1]) + (a[2] + a[3]);
    for (; i < n; i++) sub = fma((double)wf[i], (double)xg[i], sub);
    return sub;
}
#endif

static double group_dot(const float *wf, const float *xg, int n)
{
#if MXFP4_X86
    if (eng_cpu_has_avx2()) return group_dot_avx2(wf, xg, n);
#endif
    return group_dot_ref(wf, xg, n);
}

/* ---------------------------------------------------------------- row dot -- */

double eng_mxfp4_dot_row_g(const void *w, const void *scales, const float *x,
                           int n, int group)
{
    eng_mxfp4_init();

    const unsigned char *pr = (const unsigned char *)w;
    const unsigned char *sr = (const unsigned char *)scales;
    const int ngrp  = (n + group - 1) / group;
    const int gbyte = group / 2;

    double acc = 0.0;
    for (int g = 0; g < ngrp; g++) {
        const unsigned char sb = sr[g];
        if (sb == 255) continue;                  /* NaN scale: contribute nothing */
        const unsigned char *pb = pr + (size_t)g * gbyte;
        const float *xg = x + (size_t)g * group;

        int m = n - g * group;
        if (m > group) m = group;

        /* Expand the group to floats first, then take a plain dot product. The split
         * exists so the second loop can vectorise, which it cannot while a table lookup
         * sits in the middle of the accumulation. */
        float wf[64];                             /* group is 32 here; 64 is headroom */
        const int half = m >> 1;
        for (int j = 0; j < half; j++) {
            const float *pv = E2M1_PAIR[pb[j]];
            wf[2 * j]     = pv[0];
            wf[2 * j + 1] = pv[1];
        }
        if (m & 1) wf[m - 1] = E2M1_PAIR[pb[half]][0];

        /* The scale is constant within a group, so it factors out of the inner sum and
         * is applied once per group instead of once per element. This is also the
         * reassociation the accuracy contract in quant.h describes. */
        acc += group_dot(wf, xg, m) * (double)E8M0[sb];
    }
    return acc;
}

static double mxfp4_dot_row(const void *w, const void *scales, const float *x, int n)
{
    return eng_mxfp4_dot_row_g(w, scales, x, n, MXFP4_GROUP);
}

/* -------------------------------------------------------------- dequantise -- */

void eng_mxfp4_dequant_row_g(float *out, const void *w, const void *scales,
                             int n, int group)
{
    const unsigned char *pr = (const unsigned char *)w;
    const unsigned char *sr = (const unsigned char *)scales;
    const int ngrp = (n + group - 1) / group;

    for (int g = 0; g < ngrp; g++) {
        /* E8M0 is a bare biased exponent. 255 is NaN by spec; map it to zero so one bad
         * byte cannot poison the row. ldexpf is exact for powers of two. */
        const unsigned char sb = sr[g];
        const float mult = (sb == 255) ? 0.0f : ldexpf(1.0f, (int)sb - 127);

        const int lo = g * group;
        int hi = lo + group;
        if (hi > n) hi = n;

        for (int i = lo; i < hi; i++) {
            const unsigned char byte = pr[i >> 1];
            /* low nibble = EVEN element. Reversing this gives right values in wrong
             * places, which every statistical check would pass. */
            const unsigned char nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
            out[i] = E2M1[nib] * mult;
        }
    }
}

static void mxfp4_dequant_row(float *out, const void *w, const void *scales, int n)
{
    eng_mxfp4_dequant_row_g(out, w, scales, n, MXFP4_GROUP);
}

static int64_t mxfp4_row_bytes(int n)
{
    if (n < 0 || (n & 1)) return -1;              /* two elements per byte */
    return n / 2;
}

static int64_t mxfp4_scale_bytes(int n)
{
    if (n < 0) return -1;
    return (n + MXFP4_GROUP - 1) / MXFP4_GROUP;   /* one E8M0 byte per group */
}

const EngQuantOps eng_quant_mxfp4 = {
    .dtype       = ENG_DT_MXFP4,
    .name        = "mxfp4",
    .group       = MXFP4_GROUP,
    .dot_row     = mxfp4_dot_row,
    .dequant_row = mxfp4_dequant_row,
    .row_bytes   = mxfp4_row_bytes,
    .scale_bytes = mxfp4_scale_bytes,
    /* Every product is exact in double (3 mantissa bits from E2M1 plus 24 from the
     * activation needs 27 of 53), so only the additions round and the group-wise
     * reassociation moves the result by about 1 ULP of double. Measured against
     * dequantise-then-dot on real checkpoint weights, the gap is ~1e-16 relative; 1e-9
     * is a generous ceiling that still catches a genuinely wrong kernel. */
    .dequant_tolerance = 1e-9
};
