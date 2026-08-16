/* SPDX-License-Identifier: Apache-2.0 */
/*
 * q4_k.c - GGUF Q4_K. 256 elements in 144 bytes, 4.5 bits per weight.
 *
 * THE BLOCK, byte for byte. Restated here rather than referenced, because every offset
 * below depends on it and a reader who has to go and find ggml's struct cannot check
 * this file against anything:
 *
 *     offset  size  field
 *          0     2  d      f16, the scale APPLIED TO THE 6-BIT SCALES
 *          2     2  dmin   f16, the scale APPLIED TO THE 6-BIT MINS
 *          4    12  scales 8 six-bit scales and 8 six-bit mins, bit-packed
 *         16   128  qs     256 four-bit quants, two per byte
 *       total 144
 *
 * THIS IS AN AFFINE FORMAT, NOT A SYMMETRIC ONE. The value is
 *
 *     w = (d * sc) * q - (dmin * m)
 *
 * with a per-sub-block MIN subtracted, not a symmetric scale around zero. MXFP4 and
 * Q8_0 are symmetric and have no min at all, which is why the vtable's dot_row cannot
 * assume the group scale factors cleanly out of the sum: it does not. The offset term
 * has to be accumulated separately -- see the dot below -- and getting that wrong
 * produces weights that are systematically shifted, which still trains a plausible
 * output distribution and is exactly the kind of wrongness that survives a smoke test.
 *
 * THE 6-BIT PACKING is the fiddly part and is where a wrong implementation lands. Twelve
 * bytes hold eight scales and eight mins at six bits each (8*6 + 8*6 = 96 bits = 12
 * bytes). The first four of each are stored plainly in the low 6 bits of bytes 0..7; the
 * last four are split, taking their low 4 bits from bytes 8..11 and their high 2 bits
 * from the top of bytes 0..7. get_scale_min() below is that layout written out.
 *
 * SUB-BLOCK STRUCTURE. The 256 elements are eight sub-blocks of 32, each with its own
 * (scale, min) pair. The quants are stored so that one 32-byte stretch of qs holds the
 * LOW nibbles of sub-block 2k and the HIGH nibbles of sub-block 2k+1 -- not two
 * consecutive sub-blocks packed one after the other. Reading it the obvious way
 * interleaves two sub-blocks' worth of weights with each other's scales.
 */
#include "quant.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define QK_K        256
#define Q4_K_BYTES  144
#define Q4_K_SUB     32     /* elements per (scale, min) pair */

/* The 6-bit scale/min unpacking. j selects one of the eight sub-blocks. */
static inline void get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m)
{
    if (j < 4) {
        /* Plain: low 6 bits of two separate bytes. */
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        /* Split: low 4 bits from bytes 8..11, high 2 bits borrowed from the tops of
         * bytes 0..7, which the first four sub-blocks left unused. */
        *sc = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m  = (uint8_t)((q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4));
    }
}

static void q4_k_dequant_row(float *out, const void *w, const void *scales, int n)
{
    (void)scales;                    /* interleaved: the block carries its own */
    const uint8_t *blk = (const uint8_t *)w;
    const int nblk = n / QK_K;

    for (int b = 0; b < nblk; b++) {
        const uint8_t *p = blk + (size_t)b * Q4_K_BYTES;
        uint16_t hd, hm;
        memcpy(&hd, p + 0, 2);
        memcpy(&hm, p + 2, 2);
        const float d    = eng_f16_to_f32(hd);
        const float dmin = eng_f16_to_f32(hm);
        const uint8_t *sc12 = p + 4;
        const uint8_t *qs   = p + 16;

        float *y = out + (size_t)b * QK_K;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min(is + 0, sc12, &sc, &m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min(is + 1, sc12, &sc, &m);
            const float d2 = d * sc, m2 = dmin * m;

            /* One 32-byte stretch, two sub-blocks: LOW nibbles are sub-block is+0,
             * HIGH nibbles are is+1. */
            const uint8_t *q = qs + (size_t)(j / 64) * 32;
            for (int l = 0; l < 32; l++) y[j + l]      = d1 * (float)(q[l] & 0x0F) - m1;
            for (int l = 0; l < 32; l++) y[j + 32 + l] = d2 * (float)(q[l] >>   4) - m2;
            is += 2;
        }
    }
}

/* Fused dot.
 *
 * The affine term is what makes this different from a symmetric format. For one
 * sub-block,
 *
 *     sum_l w_l * x_l  =  (d*sc) * sum_l q_l * x_l  -  (dmin*m) * sum_l x_l
 *
 * so BOTH a weighted sum of the quants and a plain sum of the activations are needed
 * per sub-block. Dropping the second term is the classic Q4_K bug: outputs stay
 * finite and plausibly scaled, and only a numerical comparison catches it.
 *
 * Accumulating in double throughout, and per sub-block rather than per element, so the
 * two terms combine at the same magnitude. */
static double q4_k_dot_row(const void *w, const void *scales, const float *x, int n)
{
    (void)scales;
    const uint8_t *blk = (const uint8_t *)w;
    const int nblk = n / QK_K;
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
            double s1 = 0.0, s2 = 0.0;   /* sum q*x   */
            double t1 = 0.0, t2 = 0.0;   /* sum x     */
            for (int l = 0; l < 32; l++) {
                const double xa = xb[j + l], xb2 = xb[j + 32 + l];
                s1 += (double)(q[l] & 0x0F) * xa;
                t1 += xa;
                s2 += (double)(q[l] >> 4) * xb2;
                t2 += xb2;
            }
            acc += d1 * s1 - m1 * t1;
            acc += d2 * s2 - m2 * t2;
            is += 2;
        }
    }
    return acc;
}

static int64_t q4_k_row_bytes(int n)
{
    if (n < 0 || n % QK_K) return -1;
    return (int64_t)(n / QK_K) * Q4_K_BYTES;
}

static int64_t q4_k_scale_bytes(int n) { (void)n; return 0; }

const EngQuantOps eng_quant_q4_k = {
    .dtype       = ENG_DT_Q4_K,
    .name        = "q4_k",
    .group       = QK_K,
    .dot_row     = q4_k_dot_row,
    .dequant_row = q4_k_dequant_row,
    .row_bytes   = q4_k_row_bytes,
    .scale_bytes = q4_k_scale_bytes,
    /* Looser than MXFP4's 1e-9, and for a structural reason rather than sloppiness. A
     * Q4_K product is NOT exact in double: the scale d*sc is an f16 times a 6-bit
     * integer, and the affine term subtracts a second such product, so the fused form
     * and the materialised form group their roundings differently. The gap is a few
     * ULPs of the sub-block magnitude; 1e-6 relative is comfortably above that and far
     * below anything a layout error would produce. */
    .dequant_tolerance = 1e-6
};
