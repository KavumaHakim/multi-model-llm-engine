/* SPDX-License-Identifier: Apache-2.0 */
/*
 * q8_0.c - GGUF Q8_0: 32 int8 weights per block with one f16 scale, INTERLEAVED.
 *
 * WHY THIS FORMAT IS HERE AND NOT LEFT UNTIL A MODEL NEEDS IT
 *   MXFP4 alone cannot validate the quantization interface, because MXFP4 is the family
 *   the interface was extracted from. Both of its distinguishing traits -- external
 *   scale tensors and a 2-elements-per-byte packing -- would go untested as VARIABLES:
 *   an interface hardcoded to external scales passes every MXFP4 test.
 *
 *   Q8_0 differs on exactly the axes that matter. Its scale lives INSIDE the block
 *   rather than in a sibling tensor, so scale_bytes() returns 0 and dot_row is handed a
 *   NULL scales pointer; and it is one byte per element rather than two elements per
 *   byte, so row_bytes() has a different shape. Registering it is what demonstrates the
 *   abstraction generalises rather than merely compiling.
 *
 *   It is also the format the GGUF Q4_K/Q6_K work at M8-M9 will lean on: llama.cpp
 *   quantizes activations to Q8 before a k-quant matmul, so this block layout is
 *   already the one the next milestone needs.
 *
 * LAYOUT, which is sizeof(block_q8_0) in ggml and not a derivation from "8 bits":
 *   offset 0   f16  d      the scale
 *   offset 2   i8   qs[32] the weights
 *   total      34 bytes for 32 elements
 *
 * Value of element i is d * qs[i]. No zero point -- Q8_0 is symmetric, which is the
 * "_0" in the name (Q8_1 adds one, and is a different dtype).
 */
#include "quant.h"

#include <math.h>      /* fma: without this the implicit declaration returns int */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define Q8_0_GROUP 32
#define Q8_0_BYTES 34

/* f16 -> f32. Unlike bf16 this is NOT a shift: f16 has 5 exponent bits against f32's 8,
 * so the exponent must be rebiased (15 -> 127), and the subnormal and infinity/NaN
 * ranges need their own handling. Written out rather than using _cvtsh_ss so it works
 * without F16C, and rather than a 64 KB table because this runs once per 32 weights,
 * not once per weight. */
static float f16_to_f32(uint16_t h)
{
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    union { uint32_t u; float f; } v;

    if (exp == 0) {
        if (man == 0) { v.u = sign; return v.f; }        /* signed zero */
        /* Subnormal: normalise by shifting the mantissa up until its leading bit falls
         * out, decrementing the exponent to match. */
        uint32_t e = 0, m = man;
        while (!(m & 0x400u)) { m <<= 1; e++; }
        m &= 0x3FFu;
        v.u = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
        return v.f;
    }
    if (exp == 0x1F) {                                    /* inf or NaN */
        v.u = sign | 0x7F800000u | (man << 13);
        return v.f;
    }
    v.u = sign | ((exp - 15 + 127) << 23) | (man << 13);
    return v.f;
}

/* The block's scale and weights, from the interleaved layout. */
static inline float blk_scale(const unsigned char *b)
{
    uint16_t h;
    memcpy(&h, b, sizeof h);      /* memcpy, not a cast: the block is not 2-aligned */
    return f16_to_f32(h);
}

static double q8_0_dot_row(const void *w, const void *scales, const float *x, int n)
{
    (void)scales;                 /* interleaved: there is no external scale array */
    const unsigned char *p = (const unsigned char *)w;

    double acc = 0.0;
    for (int off = 0; off < n; off += Q8_0_GROUP) {
        const unsigned char *blk = p + (size_t)(off / Q8_0_GROUP) * Q8_0_BYTES;
        const double d = (double)blk_scale(blk);
        const signed char *q = (const signed char *)(blk + 2);

        int m = n - off;
        if (m > Q8_0_GROUP) m = Q8_0_GROUP;

        /* Same four-lane partition and (s0+s1)+(s2+s3) reduction as the other kernels,
         * so scalar and any future vector path stay bit-identical to each other. */
        double s[4] = { 0, 0, 0, 0 };
        int i = 0;
        for (; i + 3 < m; i += 4)
            for (int l = 0; l < 4; l++)
                s[l] = fma((double)q[i + l], (double)x[off + i + l], s[l]);
        double sub = (s[0] + s[1]) + (s[2] + s[3]);
        for (; i < m; i++) sub = fma((double)q[i], (double)x[off + i], sub);

        /* Scale factored out of the group, exactly as MXFP4 does. */
        acc += sub * d;
    }
    return acc;
}

static void q8_0_dequant_row(float *out, const void *w, const void *scales, int n)
{
    (void)scales;
    const unsigned char *p = (const unsigned char *)w;

    for (int off = 0; off < n; off += Q8_0_GROUP) {
        const unsigned char *blk = p + (size_t)(off / Q8_0_GROUP) * Q8_0_BYTES;
        const float d = blk_scale(blk);
        const signed char *q = (const signed char *)(blk + 2);

        int m = n - off;
        if (m > Q8_0_GROUP) m = Q8_0_GROUP;
        for (int i = 0; i < m; i++) out[off + i] = d * (float)q[i];
    }
}

static int64_t q8_0_row_bytes(int n)
{
    /* Whole blocks only. A partial block has no defined layout, and rounding up would
     * read past the row into the next one -- which decodes to plausible noise. */
    if (n < 0 || (n % Q8_0_GROUP)) return -1;
    return (int64_t)(n / Q8_0_GROUP) * Q8_0_BYTES;
}

static int64_t q8_0_scale_bytes(int n)
{
    (void)n;
    return 0;                     /* interleaved */
}

const EngQuantOps eng_quant_q8_0 = {
    .dtype       = ENG_DT_Q8_0,
    .name        = "q8_0",
    .group       = Q8_0_GROUP,
    .dot_row     = q8_0_dot_row,
    .dequant_row = q8_0_dequant_row,
    .row_bytes   = q8_0_row_bytes,
    .scale_bytes = q8_0_scale_bytes,
    /* int8 x f32 is exact in double (8 bits + 24 needs 32 of 53), so as with MXFP4 only
     * the additions round and the group reassociation costs ~1 ULP of double. */
    .dequant_tolerance = 1e-9
};
