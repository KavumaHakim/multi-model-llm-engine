/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dtype.c - the dtype table.
 *
 * BLOCK SIZES ARE NOT GUESSES. Each k-quant figure below is the sizeof() of the
 * corresponding ggml block struct, which is what a GGUF file on disk actually
 * contains. They are restated here rather than derived from a bits-per-weight figure
 * because the padding is real: Q4_K is 144 bytes for 256 elements (4.5 bits/weight),
 * but Q3_K is 110 bytes for 256 (3.4375), and neither is what "3 bits" or "4 bits"
 * would predict. A derived number would be plausible and wrong.
 *
 * These are validated against the target file by tests/unit/test_dtype.c, which sums
 * the computed sizes of all 399 tensors in the Qwen3 GGUF and checks the total against
 * the file's own tensor-data region. That is the check that matters: a wrong constant
 * here shifts every subsequent tensor offset and the model reads as noise.
 */
#include "dtype.h"

#include <string.h>

#define F   ENG_DTF_FLOAT
#define I   ENG_DTF_INT
#define Q   ENG_DTF_QUANTIZED
#define B   ENG_DTF_BLOCKED
#define X   ENG_DTF_EXT_SCALES
#define R   ENG_DTF_ROW_SCALE

/* Indexed by EngDType. Gaps are zeroed and read as unknown, which is why
 * eng_dtype_info() tests `name` rather than trusting the index. */
static const EngDTypeInfo TBL[ENG_DT_MAX] = {
    [ENG_DT_F32]   = { "f32",   F,         1,   4, 0 },
    [ENG_DT_F16]   = { "f16",   F,         1,   2, 0 },
    [ENG_DT_BF16]  = { "bf16",  F,         1,   2, 0 },
    [ENG_DT_F64]   = { "f64",   F,         1,   8, 0 },
    [ENG_DT_I8]    = { "i8",    I,         1,   1, 0 },
    [ENG_DT_U8]    = { "u8",    I,         1,   1, 0 },
    [ENG_DT_I16]   = { "i16",   I,         1,   2, 0 },
    [ENG_DT_I32]   = { "i32",   I,         1,   4, 0 },
    [ENG_DT_I64]   = { "i64",   I,         1,   8, 0 },

    /* legacy GGUF quants, 32-element blocks */
    [ENG_DT_Q4_0]  = { "q4_0",  F|Q|B,    32,  18, 0 },  /* d(f16) + 16 nibbles      */
    [ENG_DT_Q4_1]  = { "q4_1",  F|Q|B,    32,  20, 0 },  /* d,m(f16) + 16            */
    [ENG_DT_Q5_0]  = { "q5_0",  F|Q|B,    32,  22, 0 },  /* d + qh(4) + 16           */
    [ENG_DT_Q5_1]  = { "q5_1",  F|Q|B,    32,  24, 0 },  /* d,m + qh(4) + 16         */
    [ENG_DT_Q8_0]  = { "q8_0",  F|Q|B,    32,  34, 0 },  /* d(f16) + 32 int8         */
    [ENG_DT_Q8_1]  = { "q8_1",  F|Q|B,    32,  36, 0 },  /* d,s(f16) + 32 int8       */

    /* k-quants, 256-element superblocks */
    [ENG_DT_Q2_K]  = { "q2_k",  F|Q|B,   256,  84, 0 },
    [ENG_DT_Q3_K]  = { "q3_k",  F|Q|B,   256, 110, 0 },
    [ENG_DT_Q4_K]  = { "q4_k",  F|Q|B,   256, 144, 0 },
    [ENG_DT_Q5_K]  = { "q5_k",  F|Q|B,   256, 176, 0 },
    [ENG_DT_Q6_K]  = { "q6_k",  F|Q|B,   256, 210, 0 },
    [ENG_DT_Q8_K]  = { "q8_k",  F|Q|B,   256, 292, 0 },

    /* OCP MX FP4: two E2M1 nibbles per byte. The E8M0 exponents are a SEPARATE tensor,
     * one byte per 32 elements, so they are deliberately not counted here -- the packed
     * tensor's size is the nibbles alone. See k3_load.h for the pairing. */
    [ENG_DT_MXFP4] = { "mxfp4", F|Q|B|X,   2,   1, 0 },

    /* Draft-only: [f32 scale][int8 x cols] per row. block_bytes covers the int8 body;
     * row_extra covers the scale, which is why the size depends on the row split. */
    [ENG_DT_I8R]   = { "i8r",   F|Q|R,     1,   1, 4 },
};

const EngDTypeInfo *eng_dtype_info(EngDType dt)
{
    if ((int)dt <= 0 || (int)dt >= ENG_DT_MAX) return NULL;
    return TBL[dt].name ? &TBL[dt] : NULL;
}

const char *eng_dtype_name(EngDType dt)
{
    const EngDTypeInfo *i = eng_dtype_info(dt);
    return i ? i->name : "?";
}

EngDType eng_dtype_by_name(const char *name)
{
    if (!name) return ENG_DT_INVALID;
    for (int d = 1; d < ENG_DT_MAX; d++)
        if (TBL[d].name && !strcmp(TBL[d].name, name)) return (EngDType)d;
    return ENG_DT_INVALID;
}

int eng_dtype_is_quantized(EngDType dt)
{
    const EngDTypeInfo *i = eng_dtype_info(dt);
    return i && (i->flags & ENG_DTF_QUANTIZED) ? 1 : 0;
}

int eng_dtype_is_blocked(EngDType dt)
{
    const EngDTypeInfo *i = eng_dtype_info(dt);
    return i && (i->flags & ENG_DTF_BLOCKED) ? 1 : 0;
}

int eng_dtype_has_ext_scales(EngDType dt)
{
    const EngDTypeInfo *i = eng_dtype_info(dt);
    return i && (i->flags & ENG_DTF_EXT_SCALES) ? 1 : 0;
}

int64_t eng_dtype_bytes(EngDType dt, int64_t nelem)
{
    const EngDTypeInfo *i = eng_dtype_info(dt);
    if (!i || nelem < 0) return -1;
    /* A row-scaled type's size is a function of the row split, not the element count.
     * Returning nelem*1 here would be off by 4 bytes per row and would look right in a
     * single-row test. Refuse instead. */
    if (i->flags & ENG_DTF_ROW_SCALE) return -1;
    if (nelem % i->block_elems) return -1;
    return (nelem / i->block_elems) * (int64_t)i->block_bytes;
}

int64_t eng_dtype_row_bytes(EngDType dt, int64_t cols)
{
    const EngDTypeInfo *i = eng_dtype_info(dt);
    if (!i || cols < 0) return -1;
    if (cols % i->block_elems) return -1;
    return (cols / i->block_elems) * (int64_t)i->block_bytes + (int64_t)i->row_extra;
}

uint32_t eng_dtype_scale_group(EngDType dt)
{
    const EngDTypeInfo *i = eng_dtype_info(dt);
    if (!i || !(i->flags & ENG_DTF_EXT_SCALES)) return 0;
    /* MXFP4 is the only external-scale type today and its group is fixed by the OCP MX
     * spec at 32. Named here rather than spelled at call sites so a future format with
     * a different group changes one line. */
    return 32;
}
