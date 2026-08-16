/* SPDX-License-Identifier: Apache-2.0 */
/*
 * quant.h - quantized formats behind one interface.
 *
 * THE PRIMARY OPERATION IS A FUSED DOT PRODUCT, NOT "DEQUANTIZE A TENSOR", and that
 * choice is the whole design.
 *
 *   Materialising is what streaming exists to avoid. One K3 routed expert is 33,030,144
 *   parameters: 132 MB as fp32, 17.55 MB as packed nibbles. A single token touches 1,472
 *   experts, so dequantise-then-multiply would move 194 GB of materialised weights per
 *   token on a machine with 8 GB of RAM.
 *
 *   And it is not even a speed/memory trade. A matrix-vector product is memory bound, so
 *   reading 7.5x fewer bytes makes the fused kernel FASTER than dequantising first, not
 *   slower. The packed path wins on both axes at once, which is why it is the interface
 *   and bulk dequantisation is the side door.
 *
 *   eng_quant_dequant_row exists anyway, for two honest reasons: tests need a
 *   independent path to compare the fused kernel against, and a weight small enough to
 *   stay resident (a norm vector, an embedding row) is sometimes cheaper materialised
 *   once than decoded on every use.
 *
 * TWO FAMILIES OF SCALE STORAGE, and the interface has to carry both
 *
 *   EXTERNAL   the packed weights and their scales are separate tensors. K3's MXFP4
 *              does this: `w1.weight_packed` holds nibbles, `w1.weight_scale` holds one
 *              E8M0 byte per 32 elements, and the loader pairs them by name.
 *   INTERLEAVED the scale metadata sits INSIDE each block. Every GGUF k-quant does
 *              this: a Q4_K superblock is 256 elements in 144 bytes with its scales and
 *              mins packed in among the weights.
 *
 *   So every entry point takes both a `w` pointer and a `scales` pointer, and `scales`
 *   is NULL for interleaved formats. Designing for only one family would have meant
 *   reworking the interface the moment the second model arrived -- which is precisely
 *   the situation this refactor exists to end.
 *
 * ACCURACY CONTRACT, per format and stated rather than assumed
 *
 *   A fused quantized dot is NOT bit-identical to dequantise-then-dot, and no caller
 *   should assume it is. The fused kernel factors each group's scale out of that group's
 *   sum and applies it once; the materialised path sums every term of the row under one
 *   set of accumulators. The orders differ, so the results differ.
 *
 *   The difference is bounded and tiny where it matters. For MXFP4 every individual
 *   product is EXACT in double -- an E2M1 value carries 3 mantissa bits and the
 *   activation carries 24, so the product needs 27 of the 53 available -- so only the
 *   additions round, and reassociating exact terms moves the result by about 1 ULP of
 *   double, order 1e-16 relative. Each registered format declares its own tolerance in
 *   `dequant_tolerance` and tests/unit/test_quant.c holds it to that figure.
 *
 *   This is a different guarantee from the kernels' ENG_NUM_EXACT, which requires
 *   bit-identity between IMPLEMENTATIONS of the same operation. That one still holds:
 *   the scalar and vector paths of a given quantized kernel must agree exactly, because
 *   K3's output must not depend on which machine ran it.
 */
#ifndef ENG_QUANT_H
#define ENG_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    EngDType    dtype;
    const char *name;

    /* Elements sharing one scale. 32 for MXFP4 and the legacy GGUF quants, 256 for the
     * k-quants' superblocks. */
    uint32_t    group;

    /* Dot one row of `n` elements against x, dequantising as it goes and never
     * materialising the row.
     *
     * `w`      packed weight bytes for this row
     * `scales` external scale bytes for this row, or NULL when interleaved
     *
     * Returns a double so the caller controls the final narrowing: a matmul over many
     * rows may want to keep the wider value while it accumulates something else. */
    double (*dot_row)(const void *w, const void *scales, const float *x, int n);

    /* Materialise one row. The independent path tests compare dot_row against, and the
     * right choice for a weight small enough to stay resident. */
    void (*dequant_row)(float *out, const void *w, const void *scales, int n);

    /* Bytes of packed weight for one row of n elements, EXCLUDING external scales. */
    int64_t (*row_bytes)(int n);

    /* Bytes of external scale data for one row, or 0 when scales are interleaved. */
    int64_t (*scale_bytes)(int n);

    /* Largest acceptable relative difference between dot_row and
     * dequant_row-then-dot on this format. Declared per format because it depends on
     * the mantissa width: see the accuracy contract above. */
    double  dequant_tolerance;
} EngQuantOps;

/* Look up a format. NULL when the dtype is not quantized or has no implementation --
 * callers must check rather than proceeding, because there is no sensible default for
 * "decode these bytes somehow". */
const EngQuantOps *eng_quant_ops(EngDType dt);

/* Register an implementation. Returns 0 on success, -1 if the dtype already has one or
 * the ops are incomplete. Built-in formats register themselves on first lookup. */
int eng_quant_register(const EngQuantOps *ops);

/* How many formats are implemented, and the i'th, for `inspect` and for tests that must
 * cover every registered format rather than a list that drifts out of date. */
int                eng_quant_count(void);
const EngQuantOps *eng_quant_at(int i);

/* y[rows] = W[rows][in] . x[in], with W packed in `dt` and never materialised.
 *
 * `scales` is the external scale array for the whole matrix (rows * scale_bytes(in)),
 * or NULL for interleaved formats. Returns -1 when the dtype has no implementation. */
int eng_matmul_quant(float *y, const float *x, const void *W, const void *scales,
                     int in, int rows, EngDType dt);

#ifdef __cplusplus
}
#endif

#endif /* ENG_QUANT_H */
