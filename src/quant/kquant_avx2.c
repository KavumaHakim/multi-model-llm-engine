/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kquant_avx2.c - AVX2 fused dot products for GGUF Q4_K and Q6_K.
 *
 * These two carry the whole Qwen3 compute path: 217 of its tensors are Q4_K and 37 are
 * Q6_K, and until this file existed both were plain scalar C while bf16, f32 and MXFP4
 * already had vector paths.
 *
 * BIT-IDENTICAL TO THE SCALAR REFERENCE, not merely close. quant.h requires it, and the
 * requirement is what shaped the scalar code: both q4_k.c and q6_k.c were restructured
 * into a four-lane accumulator tree specifically so a 4-wide double register could
 * reproduce them. Lane k takes the elements at l = 4i + k; the lanes fold as
 * (0+1)+(2+3). Every product is an IEEE fused multiply-add in double on both sides, and
 * every widening conversion below is exact:
 *
 *   _mm_cvtepu8_epi32   u8  -> i32   exact by construction
 *   _mm256_cvtepi32_pd  i32 -> f64   exact, 32 bits into a 53-bit mantissa
 *   _mm256_cvtps_pd     f32 -> f64   exact, a float is a subset of a double
 *
 * So the two paths differ in no operation that can round differently, which is why
 * tests/unit/test_quant.c can compare them on raw bits rather than to a tolerance. A
 * tolerance-based check would pass on an implementation that is merely close, and merely
 * close is what breaks the guarantee that output does not depend on the host.
 *
 * COMPILED UNCONDITIONALLY via per-function target attributes, exactly as
 * kernels/kernel_avx2.c is: one binary carries both paths and the dispatcher chooses
 * from what the CPU reports. A file-level -mavx2 would let the compiler hoist AVX2 into
 * a function reachable before the check.
 */
#include "quant.h"

#if defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define ENG_KQ_AVX2 1
#else
#  define ENG_KQ_AVX2 0
#endif

#include <math.h>
#include <stdint.h>
#include <string.h>

int eng_kquant_avx2_compiled(void) { return ENG_KQ_AVX2; }

#if ENG_KQ_AVX2

#define AVX2_FN __attribute__((target("avx2,fma")))

#define QK_K        256
#define Q4_K_BYTES  144
#define Q6_K_BYTES  210

/* Fold four lanes as (0+1)+(2+3), matching the scalar RED4. */
AVX2_FN static inline double red4(__m256d v)
{
    double t[4];
    _mm256_storeu_pd(t, v);
    return (t[0] + t[1]) + (t[2] + t[3]);
}

/* Load 4 bytes and zero-extend to four int32 lanes. */
AVX2_FN static inline __m128i load4_u8(const uint8_t *p)
{
    int32_t packed;
    memcpy(&packed, p, 4);
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed));
}

/* ---------------------------------------------------------------------- Q4_K -- */

/* Same 6-bit unpacking as q4_k.c. Restated rather than shared because it is four lines
 * and a cross-file dependency here would mean the scalar file could not be read on its
 * own -- which is the property that makes the block layout checkable. */
AVX2_FN static inline void get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m  = (uint8_t)((q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4));
    }
}

AVX2_FN double eng_q4_k_dot_row_avx2(const void *w, const float *x, int n)
{
    const uint8_t *blk = (const uint8_t *)w;
    const int nblk = n / QK_K;
    const __m128i mask_lo = _mm_set1_epi32(0x0F);
    double acc = 0.0;

    for (int b = 0; b < nblk; b++) {
        const uint8_t *p = blk + (size_t)b * Q4_K_BYTES;
        uint16_t hd, hm;
        memcpy(&hd, p + 0, 2);
        memcpy(&hm, p + 2, 2);
        const double d    = (double)eng_f16_to_f32(hd);
        const double dmin = (double)eng_f16_to_f32(hm);
        const uint8_t *sc12 = p + 4;
        const uint8_t *qs   = p + 16;
        const float *xb = x + (size_t)b * QK_K;

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min(is + 0, sc12, &sc, &m);
            const double d1 = d * sc, m1 = dmin * m;
            get_scale_min(is + 1, sc12, &sc, &m);
            const double d2 = d * sc, m2 = dmin * m;

            const uint8_t *q = qs + (size_t)(j / 64) * 32;
            __m256d s1 = _mm256_setzero_pd(), t1 = _mm256_setzero_pd();
            __m256d s2 = _mm256_setzero_pd(), t2 = _mm256_setzero_pd();

            for (int l = 0; l < 32; l += 4) {
                const __m128i q32 = load4_u8(q + l);
                const __m256d qlo = _mm256_cvtepi32_pd(_mm_and_si128(q32, mask_lo));
                const __m256d qhi = _mm256_cvtepi32_pd(_mm_srli_epi32(q32, 4));
                const __m256d xa  = _mm256_cvtps_pd(_mm_loadu_ps(xb + j + l));
                const __m256d xc  = _mm256_cvtps_pd(_mm_loadu_ps(xb + j + 32 + l));

                /* Operand order matches the scalar fma(q, x, acc). */
                s1 = _mm256_fmadd_pd(qlo, xa, s1);
                t1 = _mm256_add_pd(t1, xa);
                s2 = _mm256_fmadd_pd(qhi, xc, s2);
                t2 = _mm256_add_pd(t2, xc);
            }

            acc += d1 * red4(s1) - m1 * red4(t1);
            acc += d2 * red4(s2) - m2 * red4(t2);
            is += 2;
        }
    }
    return acc;
}

/* ---------------------------------------------------------------------- Q6_K -- */

AVX2_FN double eng_q6_k_dot_row_avx2(const void *w, const float *x, int n)
{
    const uint8_t *blk = (const uint8_t *)w;
    const int nblk = n / QK_K;
    const __m128i mask_lo = _mm_set1_epi32(0x0F);
    const __m128i mask_2  = _mm_set1_epi32(3);
    const __m128i bias    = _mm_set1_epi32(32);
    double acc = 0.0;

    for (int b = 0; b < nblk; b++) {
        const uint8_t *p  = blk + (size_t)b * Q6_K_BYTES;
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t  *sc = (const int8_t *)(p + 192);
        uint16_t hd;
        memcpy(&hd, p + 208, 2);
        const double d = (double)eng_f16_to_f32(hd);
        const float *xb = x + (size_t)b * QK_K;

        for (int half = 0; half < 2; half++) {
            __m256d part[8];
            for (int k = 0; k < 8; k++) part[k] = _mm256_setzero_pd();

            for (int l = 0; l < 32; l += 4) {
                const int is = l / 16;      /* constant across these four */
                const __m128i lo0 = load4_u8(ql + l);
                const __m128i lo1 = load4_u8(ql + l + 32);
                const __m128i hi  = load4_u8(qh + l);

                /* Six bits: four low from ql, two high from the packed qh byte. The
                 * shift selects which pair, exactly as the scalar does. */
                const __m128i q1 = _mm_sub_epi32(
                    _mm_or_si128(_mm_and_si128(lo0, mask_lo),
                                 _mm_slli_epi32(_mm_and_si128(hi, mask_2), 4)), bias);
                const __m128i q2 = _mm_sub_epi32(
                    _mm_or_si128(_mm_and_si128(lo1, mask_lo),
                                 _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(hi, 2), mask_2), 4)), bias);
                const __m128i q3 = _mm_sub_epi32(
                    _mm_or_si128(_mm_srli_epi32(lo0, 4),
                                 _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(hi, 4), mask_2), 4)), bias);
                const __m128i q4 = _mm_sub_epi32(
                    _mm_or_si128(_mm_srli_epi32(lo1, 4),
                                 _mm_slli_epi32(_mm_and_si128(_mm_srli_epi32(hi, 6), mask_2), 4)), bias);

                part[is + 0] = _mm256_fmadd_pd(_mm256_cvtepi32_pd(q1),
                                   _mm256_cvtps_pd(_mm_loadu_ps(xb + l +  0)), part[is + 0]);
                part[is + 2] = _mm256_fmadd_pd(_mm256_cvtepi32_pd(q2),
                                   _mm256_cvtps_pd(_mm_loadu_ps(xb + l + 32)), part[is + 2]);
                part[is + 4] = _mm256_fmadd_pd(_mm256_cvtepi32_pd(q3),
                                   _mm256_cvtps_pd(_mm_loadu_ps(xb + l + 64)), part[is + 4]);
                part[is + 6] = _mm256_fmadd_pd(_mm256_cvtepi32_pd(q4),
                                   _mm256_cvtps_pd(_mm_loadu_ps(xb + l + 96)), part[is + 6]);
            }

            for (int k = 0; k < 8; k++) acc += d * (double)sc[k] * red4(part[k]);
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

#else /* not x86: never called, but must link */

double eng_q4_k_dot_row_avx2(const void *w, const float *x, int n)
{ (void)w; (void)x; (void)n; return 0.0; }
double eng_q6_k_dot_row_avx2(const void *w, const float *x, int n)
{ (void)w; (void)x; (void)n; return 0.0; }

#endif
