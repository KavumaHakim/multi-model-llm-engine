/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kernel.c - reference implementations and the runtime dispatch.
 *
 * THE REFERENCE IS NORMATIVE. Under ENG_NUM_EXACT every other implementation must
 * reproduce these bits exactly, so the reduction order here is part of the contract and
 * not an implementation detail. It is the same 16-accumulator tree K3 uses, restated
 * rather than borrowed so that the two can be compared:
 *
 *     a[0..15] accumulate elements i, i+1, ... i+15 of each 16-wide step
 *     b0 = (a0+a4)+(a8+a12), b1 = (a1+a5)+(a9+a13), ...
 *     acc = (b0+b1)+(b2+b3)
 *
 * Sixteen accumulators rather than one because a single accumulator serialises on
 * floating-point add latency -- roughly 4 cycles on the target part -- which caps the
 * loop at one element per 4 cycles regardless of how wide the loads are.
 */
#include "kernel.h"
#include "kernel_impl.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ dispatch -- */

static EngImpl g_impl = ENG_IMPL_REF;
static int     g_selected = 0;

int eng_kernel_available(EngImpl impl)
{
    switch (impl) {
        case ENG_IMPL_REF:  return 1;
        case ENG_IMPL_AVX2: return eng_avx2_compiled() && eng_cpu_has_avx2();
        default:            return 0;
    }
}

const char *eng_kernel_name(EngImpl impl)
{
    switch (impl) {
        case ENG_IMPL_REF:  return "reference";
        case ENG_IMPL_AVX2: return "avx2+fma";
        case ENG_IMPL_AUTO: return "auto";
        default:            return "?";
    }
}

EngImpl eng_kernel_select(EngImpl want)
{
    if (want == ENG_IMPL_AUTO) {
        g_impl = eng_kernel_available(ENG_IMPL_AVX2) ? ENG_IMPL_AVX2 : ENG_IMPL_REF;
    } else if (!eng_kernel_available(want)) {
        /* Say so rather than crashing on the first unsupported instruction. */
        fprintf(stderr, "kernels: %s is not available here (%s); using the reference\n",
                eng_kernel_name(want),
                eng_avx2_compiled() ? "CPU lacks it" : "not compiled in");
        g_impl = ENG_IMPL_REF;
    } else {
        g_impl = want;
    }
    g_selected = 1;
    return g_impl;
}

EngImpl eng_kernel_active(void)
{
    if (!g_selected) eng_kernel_select(ENG_IMPL_AUTO);
    return g_impl;
}

void eng_kernel_report(const char *label)
{
    const EngImpl a = eng_kernel_active();
    printf("%s%skernels: %s", label ? label : "", label ? " " : "", eng_kernel_name(a));
    if (a != ENG_IMPL_AVX2) {
        if (!eng_avx2_compiled())      printf("  (AVX2 not compiled into this binary)");
        else if (!eng_cpu_has_avx2())  printf("  (this CPU has no AVX2)");
    }
    printf("\n");
}

/* --------------------------------------------------------------- reference -- */

double eng_ref_dot_f32_exact(const float *a, const float *b, int n)
{
    double acc[16] = { 0 };
    int i = 0;
    for (; i + 15 < n; i += 16)
        for (int l = 0; l < 16; l++)
            acc[l] = fma((double)a[i + l], (double)b[i + l], acc[l]);

    const double b0 = (acc[0] + acc[4]) + (acc[8]  + acc[12]);
    const double b1 = (acc[1] + acc[5]) + (acc[9]  + acc[13]);
    const double b2 = (acc[2] + acc[6]) + (acc[10] + acc[14]);
    const double b3 = (acc[3] + acc[7]) + (acc[11] + acc[15]);
    double s = (b0 + b1) + (b2 + b3);

    for (; i < n; i++) s = fma((double)a[i], (double)b[i], s);
    return s;
}

float eng_ref_dot_f32_fast(const float *a, const float *b, int n)
{
    float acc[8] = { 0 };
    int i = 0;
    for (; i + 7 < n; i += 8)
        for (int l = 0; l < 8; l++) acc[l] += a[i + l] * b[i + l];

    float s = ((acc[0] + acc[4]) + (acc[1] + acc[5]))
            + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

double eng_ref_dot_bf16_exact(const float *x, const uint16_t *w, int n)
{
    double acc[16] = { 0 };
    int i = 0;
    for (; i + 15 < n; i += 16)
        for (int l = 0; l < 16; l++)
            acc[l] = fma((double)eng_bf16f(w[i + l]), (double)x[i + l], acc[l]);

    const double b0 = (acc[0] + acc[4]) + (acc[8]  + acc[12]);
    const double b1 = (acc[1] + acc[5]) + (acc[9]  + acc[13]);
    const double b2 = (acc[2] + acc[6]) + (acc[10] + acc[14]);
    const double b3 = (acc[3] + acc[7]) + (acc[11] + acc[15]);
    double s = (b0 + b1) + (b2 + b3);

    for (; i < n; i++) s = fma((double)eng_bf16f(w[i]), (double)x[i], s);
    return s;
}

float eng_ref_dot_bf16_fast(const float *x, const uint16_t *w, int n)
{
    float acc[8] = { 0 };
    int i = 0;
    for (; i + 7 < n; i += 8)
        for (int l = 0; l < 8; l++) acc[l] += eng_bf16f(w[i + l]) * x[i + l];

    float s = ((acc[0] + acc[4]) + (acc[1] + acc[5]))
            + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (; i < n; i++) s += eng_bf16f(w[i]) * x[i];
    return s;
}

/* ------------------------------------------------------------ public entry -- */

float eng_dot_f32(const float *a, const float *b, int n, EngNumPolicy pol)
{
    if (eng_kernel_active() == ENG_IMPL_AVX2)
        return pol == ENG_NUM_EXACT ? (float)eng_avx2_dot_f32_exact(a, b, n)
                                    : eng_avx2_dot_f32_fast(a, b, n);
    return pol == ENG_NUM_EXACT ? (float)eng_ref_dot_f32_exact(a, b, n)
                                : eng_ref_dot_f32_fast(a, b, n);
}

void eng_matmul_f32(float *y, const float *x, const float *W,
                    int in, int out, EngNumPolicy pol)
{
    const int avx = eng_kernel_active() == ENG_IMPL_AVX2;
    /* Parallel over output rows, and only when there are enough of them to pay for the
     * fork: a 128-row projection costs more in barrier than it saves. K3 uses the same
     * threshold, measured. */
#ifdef _OPENMP
#   pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        if (pol == ENG_NUM_EXACT)
            y[o] = (float)(avx ? eng_avx2_dot_f32_exact(row, x, in)
                               : eng_ref_dot_f32_exact(row, x, in));
        else
            y[o] = avx ? eng_avx2_dot_f32_fast(row, x, in)
                       : eng_ref_dot_f32_fast(row, x, in);
    }
}

void eng_matmul_bf16(float *y, const float *x, const uint16_t *W,
                     int in, int out, EngNumPolicy pol)
{
    const int avx = eng_kernel_active() == ENG_IMPL_AVX2;
#ifdef _OPENMP
#   pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        if (pol == ENG_NUM_EXACT)
            y[o] = (float)(avx ? eng_avx2_dot_bf16_exact(x, row, in)
                               : eng_ref_dot_bf16_exact(x, row, in));
        else
            y[o] = avx ? eng_avx2_dot_bf16_fast(x, row, in)
                       : eng_ref_dot_bf16_fast(x, row, in);
    }
}

/* ------------------------------------------------------------ elementwise -- */

void eng_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    /* Double accumulation regardless of policy: see kernel.h. A sum of squares has no
     * cancellation, so the error grows monotonically with n. */
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++) y[i] = w[i] * (x[i] * scale);
}

void eng_softmax(float *x, int n)
{
    if (n <= 0) return;
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];

    /* Subtracting the max costs one pass and removes the overflow that makes a naive
     * softmax produce NaN on a confident distribution. */
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const float e = expf(x[i] - m);
        x[i] = e;
        sum += (double)e;
    }
    const float inv = (float)(1.0 / (sum > 0.0 ? sum : 1.0));
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void eng_silu(float *y, const float *x, int n)
{
    for (int i = 0; i < n; i++) y[i] = x[i] / (1.0f + expf(-x[i]));
}

void eng_swiglu(float *y, const float *x, int n)
{
    const float *gate = x;
    const float *up   = x + n;
    for (int i = 0; i < n; i++)
        y[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
}

void eng_rope(float *v, int dim, int pos, float theta_base, EngRopeStyle style)
{
    const int half = dim / 2;
    for (int i = 0; i < half; i++) {
        /* theta_i = pos * base^(-2i/dim). Computed in double: at base 1e6 and dim 128
         * the exponent range is wide enough that float powf loses low-order bits, and
         * the angle feeds a sin/cos whose error then shows up in every score. */
        const double freq = 1.0 / pow((double)theta_base, (2.0 * (double)i) / (double)dim);
        const double ang  = (double)pos * freq;
        const float c = (float)cos(ang), s = (float)sin(ang);

        int ia, ib;
        if (style == ENG_ROPE_HALVED) { ia = i;         ib = i + half; }
        else                          { ia = 2 * i;     ib = 2 * i + 1; }

        const float a = v[ia], b = v[ib];
        v[ia] = a * c - b * s;
        v[ib] = a * s + b * c;
    }
}
