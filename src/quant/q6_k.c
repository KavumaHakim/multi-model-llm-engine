/* SPDX-License-Identifier: Apache-2.0 */
/*
 * q6_k.c - GGUF Q6_K. 256 elements in 210 bytes, 6.5625 bits per weight.
 *
 * THE BLOCK, byte for byte:
 *
 *     offset  size  field
 *          0   128  ql      low 4 bits of each of 256 quants
 *        128    64  qh      high 2 bits of each, four quants per byte
 *        192    16  scales  int8, SIGNED, one per 16 elements
 *        208     2  d       f16 super-block scale
 *       total 210
 *
 * SYMMETRIC, WITH A BIAS OF 32. Unlike Q4_K there is no per-sub-block min: the value is
 *
 *     w = d * scale * (q - 32)
 *
 * where q is the reassembled 6-bit code in 0..63, so q-32 lands in -32..31. That bias is
 * the whole reason the scales are SIGNED int8 here while Q4_K's are unsigned 6-bit --
 * reading them as unsigned flips the sign of roughly half the weights, which produces a
 * model that still emits fluent text.
 *
 * THE 6 BITS ARE SPLIT ACROSS TWO ARRAYS and reassembled as
 *
 *     q = (ql_nibble) | (qh_2bits << 4)
 *
 * WHY THE LOOP LOOKS LIKE THAT. The 256 elements are processed in two halves of 128. In
 * each half, one iteration of l in 0..31 emits FOUR outputs, 32 apart:
 *
 *     y[l+ 0] from ql[l+ 0] low nibble  + qh[l] bits 0-1, scale sc[(l/16) + 0]
 *     y[l+32] from ql[l+32] low nibble  + qh[l] bits 2-3, scale sc[(l/16) + 2]
 *     y[l+64] from ql[l+ 0] high nibble + qh[l] bits 4-5, scale sc[(l/16) + 4]
 *     y[l+96] from ql[l+32] high nibble + qh[l] bits 6-7, scale sc[(l/16) + 6]
 *
 * so one qh byte feeds four different outputs, each with a different scale index. This
 * is not an arbitrary interleave -- it is what lets a SIMD implementation load 32 ql
 * bytes and 32 qh bytes once and produce 128 outputs -- but it means the scale index is
 * NOT simply element/16 in output order, and treating it as such silently applies the
 * wrong scale to three quarters of the weights.
 */
#include "quant.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QK_K        256
#define Q6_K_BYTES  210

static void q6_k_dequant_row(float *out, const void *w, const void *scales, int n)
{
    (void)scales;
    const uint8_t *blk = (const uint8_t *)w;
    const int nblk = n / QK_K;

    for (int b = 0; b < nblk; b++) {
        const uint8_t *p  = blk + (size_t)b * Q6_K_BYTES;
        const uint8_t *ql = p;
        const uint8_t *qh = p + 128;
        const int8_t  *sc = (const int8_t *)(p + 192);   /* SIGNED */
        uint16_t hd;
        memcpy(&hd, p + 208, 2);
        const float d = eng_f16_to_f32(hd);

        float *y = out + (size_t)b * QK_K;
        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* Fused dot. Symmetric, so each scale factors cleanly out of its 16-element group --
 * no offset term, unlike Q4_K. The eight scale groups per half are accumulated
 * separately and combined at the end so each multiply happens once rather than per
 * element. */
double eng_q6_k_dot_row_ref(const void *w, const void *scales, const float *x, int n)
{
    (void)scales;
    const uint8_t *blk = (const uint8_t *)w;
    const int nblk = n / QK_K;
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
            /* One partial sum per (scale index, quarter) pair: 2 scale indices x 4
             * quarters = 8, matching the 8 scales this half consumes. */
            /* FOUR LANES PER PARTIAL, for the reason given in q4_k.c: a serial sum
             * cannot be vectorised without changing the summation order, and quant.h
             * requires implementations to agree exactly. Lane k of part[p] takes the
             * elements at l = 4i + k, and the lanes fold as (0+1)+(2+3).
             *
             * `is` is constant across each run of 16 elements, so a 4-wide step never
             * straddles two scale indices -- which is what makes the lane assignment
             * well defined here at all. */
            double part[8][4];
            memset(part, 0, sizeof part);
            for (int l = 0; l < 32; l += 4) {
                const int is = l / 16;          /* constant over these 4 */
                for (int k = 0; k < 4; k++) {
                    const int i = l + k;
                    const int q1 = (int)((ql[i +  0] & 0x0F) | (((qh[i] >> 0) & 3) << 4)) - 32;
                    const int q2 = (int)((ql[i + 32] & 0x0F) | (((qh[i] >> 2) & 3) << 4)) - 32;
                    const int q3 = (int)((ql[i +  0] >>   4) | (((qh[i] >> 4) & 3) << 4)) - 32;
                    const int q4 = (int)((ql[i + 32] >>   4) | (((qh[i] >> 6) & 3) << 4)) - 32;
                    part[is + 0][k] = fma((double)q1, (double)xb[i +  0], part[is + 0][k]);
                    part[is + 2][k] = fma((double)q2, (double)xb[i + 32], part[is + 2][k]);
                    part[is + 4][k] = fma((double)q3, (double)xb[i + 64], part[is + 4][k]);
                    part[is + 6][k] = fma((double)q4, (double)xb[i + 96], part[is + 6][k]);
                }
            }
            for (int k = 0; k < 8; k++) {
                const double psum = ((part[k][0] + part[k][1]) + (part[k][2] + part[k][3]));
                acc += d * (double)sc[k] * psum;
            }
            xb += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

/* Runtime dispatch. The AVX2 path is bit-identical to the reference above -- see
 * kquant_avx2.c and the raw-bit comparison in tests/unit/test_quant.c -- so which one
 * runs is purely a speed decision and can never change an answer. */
int    eng_kquant_avx2_compiled(void);
int    eng_cpu_has_avx2(void);         /* kernels/kernel_impl.h */
double eng_q6_k_dot_row_avx2(const void *w, const float *x, int n);
double eng_q6_k_dot_row_ref (const void *w, const void *scales, const float *x, int n);

static double q6_k_dot_row(const void *w, const void *scales, const float *x, int n)
{
    (void)scales;
    /* ENG_KQ_SCALAR=1 forces the reference. The vector path is bit-identical, so
     * this changes speed and nothing else -- which is what lets the A/B run on one
     * binary rather than two builds that differ in more than the kernel. */
    static int scalar_only = -1;
    if (scalar_only < 0) scalar_only = getenv("ENG_KQ_SCALAR") ? 1 : 0;
    if (!scalar_only && eng_kquant_avx2_compiled() && eng_cpu_has_avx2())
        return eng_q6_k_dot_row_avx2(w, x, n);
    return eng_q6_k_dot_row_ref(w, scales, x, n);
}

static int64_t q6_k_row_bytes(int n)
{
    if (n < 0 || n % QK_K) return -1;
    return (int64_t)(n / QK_K) * Q6_K_BYTES;
}

static int64_t q6_k_scale_bytes(int n) { (void)n; return 0; }

const EngQuantOps eng_quant_q6_k = {
    .dtype       = ENG_DT_Q6_K,
    .name        = "q6_k",
    .group       = QK_K,
    .dot_row     = q6_k_dot_row,
    .dequant_row = q6_k_dequant_row,
    .row_bytes   = q6_k_row_bytes,
    .scale_bytes = q6_k_scale_bytes,
    /* Symmetric and exact per product in double (a 6-bit code times an f16-derived
     * scale needs far fewer than 53 bits), so only the group reassociation moves the
     * result -- the same situation as MXFP4 and the same tolerance. */
    .dequant_tolerance = 1e-9
};
