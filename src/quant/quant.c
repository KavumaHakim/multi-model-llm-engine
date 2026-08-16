/* SPDX-License-Identifier: Apache-2.0 */
/* quant.c - the registry and the generic quantized matmul. See quant.h. */
#include "quant.h"

#include <stdio.h>
#include <string.h>

/* Built-in formats, defined in their own files and pulled in here. A format is a table,
 * not a case in a switch, so adding one touches this list and nothing else. */
extern const EngQuantOps eng_quant_mxfp4;
extern const EngQuantOps eng_quant_q8_0;
extern const EngQuantOps eng_quant_q4_k;
extern const EngQuantOps eng_quant_q6_k;

static const EngQuantOps *REG[ENG_DT_MAX];
static int reg_ready = 0;
static int reg_count = 0;

static void reg_init(void)
{
    if (reg_ready) return;
    reg_ready = 1;      /* set FIRST: eng_quant_register calls back into lookup */
    eng_quant_register(&eng_quant_mxfp4);
    eng_quant_register(&eng_quant_q8_0);
    eng_quant_register(&eng_quant_q4_k);
    eng_quant_register(&eng_quant_q6_k);
}

int eng_quant_register(const EngQuantOps *ops)
{
    if (!ops || !ops->dot_row || !ops->dequant_row || !ops->row_bytes || !ops->scale_bytes)
        return -1;
    if ((int)ops->dtype <= 0 || (int)ops->dtype >= ENG_DT_MAX) return -1;
    reg_init();
    if (REG[ops->dtype]) return -1;         /* refuse to shadow silently */
    REG[ops->dtype] = ops;
    reg_count++;
    return 0;
}

const EngQuantOps *eng_quant_ops(EngDType dt)
{
    if ((int)dt <= 0 || (int)dt >= ENG_DT_MAX) return NULL;
    reg_init();
    return REG[dt];
}

int eng_quant_count(void)
{
    reg_init();
    return reg_count;
}

const EngQuantOps *eng_quant_at(int i)
{
    reg_init();
    int seen = 0;
    for (int d = 1; d < ENG_DT_MAX; d++)
        if (REG[d] && seen++ == i) return REG[d];
    return NULL;
}

int eng_matmul_quant(float *y, const float *x, const void *W, const void *scales,
                     int in, int rows, EngDType dt)
{
    const EngQuantOps *q = eng_quant_ops(dt);
    if (!q) {
        fprintf(stderr, "quant: no implementation for %s\n", eng_dtype_name(dt));
        return -1;
    }

    const int64_t rb = q->row_bytes(in);
    const int64_t sb = q->scale_bytes(in);
    if (rb < 0 || sb < 0) {
        fprintf(stderr, "quant: %s cannot express a row of %d elements\n", q->name, in);
        return -1;
    }
    /* An external-scale format with no scales given would silently read whatever
     * happens to be at NULL+offset. Refuse. */
    if (sb > 0 && !scales) {
        fprintf(stderr, "quant: %s needs an external scale array\n", q->name);
        return -1;
    }

#ifdef _OPENMP
#   pragma omp parallel for schedule(static) if (rows > 64)
#endif
    for (int r = 0; r < rows; r++) {
        const unsigned char *wr = (const unsigned char *)W + (size_t)r * (size_t)rb;
        const unsigned char *sr = scales
            ? (const unsigned char *)scales + (size_t)r * (size_t)sb : NULL;
        y[r] = (float)q->dot_row(wr, sr, x, in);
    }
    return 0;
}
