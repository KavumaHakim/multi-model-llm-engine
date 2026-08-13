/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dtype.h - the engine's data-type registry.
 *
 * WHY THIS EXISTS
 *   The K3 engine expressed dtype three different ways at once: a K3Dtype enum in the
 *   safetensors reader, an `int wdt` tag beside every weight pointer (K3_WF32 /
 *   K3_WBF16 / K3_WI8), and implicit knowledge inside each kernel. That worked because
 *   K3 has exactly three storage formats. Qwen3 GGUF adds Q4_K and Q6_K, and the brief
 *   calls for Q5/Q8/MXFP4 beyond that, so the tag has to become a descriptor.
 *
 * THE THREE LAYOUT FAMILIES, which is the whole reason this is a table and not an enum
 *   Quantized formats do not share a layout, and the differences are not cosmetic:
 *
 *   1. DENSE            one element, one fixed size. F32, BF16, I8.
 *
 *   2. BLOCKED          a fixed number of elements share scale metadata that is stored
 *                       INSIDE the block. GGUF's k-quants: Q4_K is 256 elements in 144
 *                       bytes, scales and mins packed in with the weights. Size is
 *                       (nelem / block_elems) * block_bytes.
 *
 *   3. EXTERNAL SCALES  the packed weights and their scales are SEPARATE TENSORS.
 *                       K3's MXFP4 does this: `w1.weight_packed` holds nibbles and
 *                       `w1.weight_scale` holds one E8M0 byte per 32 elements, and the
 *                       loader resolves them as a pair. So the packed tensor's byte
 *                       count must be computed from the nibbles ALONE (2 elements per
 *                       byte), and anything that wants the values needs the sibling.
 *
 *   4. ROW SCALE        each ROW carries its own scale inline at its start:
 *                       [f32 scale][int8 x cols]. K3's draft trunk (I8R). Byte size
 *                       cannot be computed from the element count at all -- it depends
 *                       on how the elements are divided into rows -- which is why
 *                       eng_dtype_bytes() refuses it and eng_tensor_nbytes() exists.
 *
 *   A single "bytes per element" number describes only family 1. Trying to stretch it
 *   over the other three is how a loader silently reads the wrong span.
 *
 * WHAT THIS FILE DOES NOT DO
 *   No conversion, no dequantisation, no kernels. It answers "how many bytes" and "what
 *   shape of thing is this", and nothing else. Dequantisation lives behind the quant
 *   vtable (milestone M6) so that adding a format does not mean editing this file's
 *   consumers.
 */
#ifndef ENG_DTYPE_H
#define ENG_DTYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Values are stable and may be serialised into a packed model file. Append only. */
typedef enum {
    ENG_DT_INVALID = 0,

    /* dense */
    ENG_DT_F32     = 1,
    ENG_DT_F16     = 2,
    ENG_DT_BF16    = 3,
    ENG_DT_F64     = 4,
    ENG_DT_I8      = 5,
    ENG_DT_U8      = 6,
    ENG_DT_I16     = 7,
    ENG_DT_I32     = 8,
    ENG_DT_I64     = 9,

    /* GGUF legacy quants: block metadata interleaved */
    ENG_DT_Q4_0    = 16,
    ENG_DT_Q4_1    = 17,
    ENG_DT_Q5_0    = 18,
    ENG_DT_Q5_1    = 19,
    ENG_DT_Q8_0    = 20,
    ENG_DT_Q8_1    = 21,

    /* GGUF k-quants: 256-element superblocks */
    ENG_DT_Q2_K    = 32,
    ENG_DT_Q3_K    = 33,
    ENG_DT_Q4_K    = 34,
    ENG_DT_Q5_K    = 35,
    ENG_DT_Q6_K    = 36,
    ENG_DT_Q8_K    = 37,

    /* OCP MX FP4, scales in a sibling tensor (Kimi K3 routed experts) */
    ENG_DT_MXFP4   = 48,

    /* per-row inline f32 scale then int8 weights (Kimi K3 draft trunk) */
    ENG_DT_I8R     = 49,

    ENG_DT_MAX     = 64
} EngDType;

enum {
    ENG_DTF_FLOAT      = 1u << 0,  /* real-valued */
    ENG_DTF_INT        = 1u << 1,  /* integer-valued */
    ENG_DTF_QUANTIZED  = 1u << 2,  /* needs a quant backend to produce floats */
    ENG_DTF_BLOCKED    = 1u << 3,  /* block_elems > 1 */
    ENG_DTF_EXT_SCALES = 1u << 4,  /* scales live in a SIBLING tensor */
    ENG_DTF_ROW_SCALE  = 1u << 5   /* scale is inline at the head of each row */
};

typedef struct {
    const char *name;         /* canonical, lowercase; matches GGUF naming where one exists */
    uint32_t    flags;
    uint32_t    block_elems;  /* elements per block; 1 when dense */
    uint32_t    block_bytes;  /* bytes per block, EXCLUDING any external scale tensor */
    uint32_t    row_extra;    /* extra bytes at the head of each row (ROW_SCALE only) */
} EngDTypeInfo;

/* NULL for an unknown or invalid id. Never returns a placeholder: a caller that gets
 * NULL must fail rather than proceed with a guessed size. */
const EngDTypeInfo *eng_dtype_info(EngDType dt);

const char *eng_dtype_name(EngDType dt);         /* "?" when unknown */
EngDType    eng_dtype_by_name(const char *name); /* ENG_DT_INVALID when unknown */

int eng_dtype_is_quantized(EngDType dt);
int eng_dtype_is_blocked(EngDType dt);
int eng_dtype_has_ext_scales(EngDType dt);

/* Bytes occupied by `nelem` elements.
 *
 * Returns -1, and does NOT guess, when:
 *   - dt is unknown;
 *   - nelem is negative;
 *   - nelem is not a whole multiple of block_elems (a partial block has no defined
 *     size, and rounding one up is how a reader walks off the end of a tensor);
 *   - dt carries a per-row scale, whose size depends on the row split rather than on
 *     the element count. Use eng_dtype_row_bytes() or eng_tensor_nbytes() for those.
 *
 * Check the return value. Every caller of this in the engine treats -1 as fatal. */
int64_t eng_dtype_bytes(EngDType dt, int64_t nelem);

/* Bytes for ONE row of `cols` elements, including any row-inline scale. Same -1
 * contract as above, minus the ROW_SCALE exclusion. */
int64_t eng_dtype_row_bytes(EngDType dt, int64_t cols);

/* Elements per external scale group, or 0 when the dtype has no external scales.
 * MXFP4's E8M0 exponents cover 32 elements each. */
uint32_t eng_dtype_scale_group(EngDType dt);

#ifdef __cplusplus
}
#endif

#endif /* ENG_DTYPE_H */
