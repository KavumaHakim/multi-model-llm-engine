/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kernel_avx2.c - AVX2 + FMA implementations.
 *
 * COMPILED UNCONDITIONALLY, via __attribute__((target("avx2,fma"))) on each function
 * rather than -mavx2 on the whole translation unit. That is what lets ONE binary carry
 * both this and the reference: the compiler emits AVX2 for these functions and nothing
 * else, so a portable build still contains them and the dispatcher decides at runtime
 * whether to call them. A file-level -mavx2 would let the compiler hoist AVX2 into any
 * function here, including one reachable before the CPU check.
 *
 * THE EXACT PATH IS BIT-IDENTICAL TO THE REFERENCE, not merely close, and the structure
 * below is what makes that true rather than a hope:
 *
 *   The reference keeps 16 double accumulators over each 16-element step and reduces
 *       b_j = (a[j] + a[j+4]) + (a[j+8] + a[j+12]),  acc = (b0+b1) + (b2+b3)
 *
 *   Four __m256d hold exactly those 16 lanes, in that assignment:
 *       v0 lane j <-> a[j]      v1 lane j <-> a[j+4]
 *       v2 lane j <-> a[j+8]    v3 lane j <-> a[j+12]
 *
 *   so (v0+v1)+(v2+v3) computes b_j lane-wise, and the final cross-lane
 *   (t0+t1)+(t2+t3) is the reference's last line. Every product is an IEEE fused
 *   multiply-add in double in both paths, and both convert float to double exactly.
 *
 *   The tail loop is the reference's, verbatim, for the same reason: the remainder must
 *   be folded in the same order.
 *
 * This is asserted by test_kernels comparing RAW BITS, not values within a tolerance.
 * A test that compared values would pass on an implementation that is merely close, and
 * "merely close" is what breaks K3's claim that its output does not depend on how much
 * memory it was given.
 */
#include "kernel_impl.h"

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define ENG_HAS_AVX2_SOURCE 1
#else
#  define ENG_HAS_AVX2_SOURCE 0
#endif

#include <math.h>

int eng_avx2_compiled(void) { return ENG_HAS_AVX2_SOURCE; }

int eng_cpu_has_avx2(void)
{
#if ENG_HAS_AVX2_SOURCE
    __builtin_cpu_init();
    /* Both, and checked together: the FMA intrinsics below are useless without AVX2 and
     * the AVX2 path uses FMA in its inner loop, so a CPU with one and not the other must
     * take the reference path. */
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}

#if ENG_HAS_AVX2_SOURCE

#define AVX2_FN __attribute__((target("avx2,fma")))

AVX2_FN double eng_avx2_dot_f32_exact(const float *a, const float *b, int n)
{
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        /* cvtps_pd widens 4 floats to 4 doubles exactly: a float is representable in a
         * double with room to spare, so this conversion introduces no error. */
        v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i)), v0);
        v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 4)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i + 4)), v1);
        v2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 8)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i + 8)), v2);
        v3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(a + i + 12)),
                             _mm256_cvtps_pd(_mm_loadu_ps(b + i + 12)), v3);
    }

    /* (v0+v1)+(v2+v3) lane-wise IS the reference's b_j. */
    const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
    double t[4];
    _mm256_storeu_pd(t, vt);
    double s = (t[0] + t[1]) + (t[2] + t[3]);

    for (; i < n; i++) s = fma((double)a[i], (double)b[i], s);
    return s;
}

AVX2_FN float eng_avx2_dot_f32_fast(const float *a, const float *b, int n)
{
    __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        v0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     v0);
        v1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), v1);
    }
    __m256 vs = _mm256_add_ps(v0, v1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float s = _mm_cvtss_f32(lo);

    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

/* bf16 -> f32 in vector form: zero-extend u16 to u32, shift left 16, reinterpret. The
 * same pure bit operation the scalar path does, so the widened values are identical. */
AVX2_FN static inline __m256d bf16_to_pd4(const uint16_t *w)
{
    const __m128i h = _mm_loadl_epi64((const __m128i *)w);            /* 4 x u16 */
    const __m128i u = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);      /* 4 x u32 */
    return _mm256_cvtps_pd(_mm_castsi128_ps(u));
}

AVX2_FN double eng_avx2_dot_bf16_exact(const float *x, const uint16_t *w, int n)
{
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();

    int i = 0;
    for (; i + 15 < n; i += 16) {
        /* Operand order matches the reference: fma(weight, activation, acc). FMA is
         * commutative in exact arithmetic and the hardware rounds once either way, so
         * this is belt and braces -- but the reference is normative and matching it
         * literally costs nothing. */
        v0 = _mm256_fmadd_pd(bf16_to_pd4(w + i),
                             _mm256_cvtps_pd(_mm_loadu_ps(x + i)), v0);
        v1 = _mm256_fmadd_pd(bf16_to_pd4(w + i + 4),
                             _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4)), v1);
        v2 = _mm256_fmadd_pd(bf16_to_pd4(w + i + 8),
                             _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8)), v2);
        v3 = _mm256_fmadd_pd(bf16_to_pd4(w + i + 12),
                             _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12)), v3);
    }

    const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
    double t[4];
    _mm256_storeu_pd(t, vt);
    double s = (t[0] + t[1]) + (t[2] + t[3]);

    for (; i < n; i++) s = fma((double)eng_bf16f(w[i]), (double)x[i], s);
    return s;
}

AVX2_FN float eng_avx2_dot_bf16_fast(const float *x, const uint16_t *w, int n)
{
    __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        const __m256i h0 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + i)));
        const __m256i h1 = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(w + i + 8)));
        v0 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h0, 16)),
                             _mm256_loadu_ps(x + i), v0);
        v1 = _mm256_fmadd_ps(_mm256_castsi256_ps(_mm256_slli_epi32(h1, 16)),
                             _mm256_loadu_ps(x + i + 8), v1);
    }
    __m256 vs = _mm256_add_ps(v0, v1);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    float s = _mm_cvtss_f32(lo);

    for (; i < n; i++) s += eng_bf16f(w[i]) * x[i];
    return s;
}

#else /* not x86: the dispatcher never calls these, but they must link */

double eng_avx2_dot_f32_exact (const float *a, const float *b, int n)
{ (void)a; (void)b; (void)n; return 0.0; }
float  eng_avx2_dot_f32_fast  (const float *a, const float *b, int n)
{ (void)a; (void)b; (void)n; return 0.0f; }
double eng_avx2_dot_bf16_exact(const float *x, const uint16_t *w, int n)
{ (void)x; (void)w; (void)n; return 0.0; }
float  eng_avx2_dot_bf16_fast (const float *x, const uint16_t *w, int n)
{ (void)x; (void)w; (void)n; return 0.0f; }

#endif
