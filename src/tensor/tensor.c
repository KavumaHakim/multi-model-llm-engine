/* SPDX-License-Identifier: Apache-2.0 */
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void eng_tensor_zero(EngTensor *t)
{
    if (t) memset(t, 0, sizeof *t);
}

/* RANK ZERO IS A REAL SHAPE, NOT AN ERROR.
 * safetensors writes genuine scalars with "shape": [] -- the fixture set has one
 * (scalar.f32), and the released K3 checkpoint's config-adjacent tensors can too. A
 * rank-0 tensor holds exactly one element. Treating ndim <= 0 as invalid rejected a
 * legitimate tensor and took the whole container down with it, which is how this was
 * found. Everywhere below, rank 0 behaves as a 1x1: numel 1, one row of one column. */
int64_t eng_tensor_numel(const EngTensor *t)
{
    if (!t || t->ndim < 0) return 0;
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) {
        if (t->shape[i] < 0) return 0;
        n *= t->shape[i];
    }
    return n;
}

int64_t eng_tensor_cols(const EngTensor *t)
{
    if (!t) return 0;
    return t->ndim > 0 ? t->shape[0] : 1;
}

int64_t eng_tensor_rows(const EngTensor *t)
{
    if (!t || t->ndim < 0) return 0;
    int64_t r = 1;
    for (int i = 1; i < t->ndim; i++) r *= t->shape[i];
    return r;
}

int64_t eng_tensor_nbytes(EngDType dt, const int64_t *shape, int ndim)
{
    const EngDTypeInfo *info = eng_dtype_info(dt);
    if (!info || ndim < 0 || ndim > ENG_MAX_DIMS) return -1;
    if (ndim > 0 && !shape) return -1;

    /* Rank 0 is a scalar: one element, one row, one column. See eng_tensor_numel. */
    int64_t cols = ndim > 0 ? shape[0] : 1;
    int64_t rows = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return -1;
        if (i > 0) rows *= shape[i];
    }

    /* A row-scaled dtype pays its scale once per row, so the total depends on the row
     * split and cannot come from the element count. This is the case eng_dtype_bytes()
     * refuses; here the shape supplies the missing information. */
    if (info->flags & ENG_DTF_ROW_SCALE) {
        const int64_t rb = eng_dtype_row_bytes(dt, cols);
        if (rb < 0) return -1;
        return rows * rb;
    }

    /* Everything else is uniform, but the BLOCK boundary still has to fall inside a
     * row: a matrix whose row is not a whole number of blocks has no layout, and
     * computing it from the total element count would silently accept one. */
    if (info->block_elems > 1 && cols % info->block_elems) return -1;

    int64_t n = 1;
    for (int i = 0; i < ndim; i++) n *= shape[i];
    return eng_dtype_bytes(dt, n);   /* n == 1 when rank 0 */
}

int eng_tensor_init_file(EngTensor *t, const char *name, EngDType dt,
                         const int64_t *shape, int ndim,
                         struct EngStorage *store, int64_t file_off)
{
    if (!t) return -1;
    const int64_t nb = eng_tensor_nbytes(dt, shape, ndim);
    eng_tensor_zero(t);
    if (nb < 0 || file_off < 0) return -1;

    t->name     = name;
    t->dtype    = dt;
    t->ndim     = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->nbytes   = nb;
    t->store    = store;
    t->file_off = file_off;
    t->data     = NULL;
    t->flags    = 0;                       /* not resident: that is the point */
    return 0;
}

int eng_tensor_init_mem(EngTensor *t, const char *name, EngDType dt,
                        const int64_t *shape, int ndim, void *data, int owned)
{
    if (!t) return -1;
    const int64_t nb = eng_tensor_nbytes(dt, shape, ndim);
    eng_tensor_zero(t);
    if (nb < 0) return -1;

    t->name   = name;
    t->dtype  = dt;
    t->ndim   = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->nbytes = nb;
    t->data   = data;
    t->flags  = ENG_TF_RESIDENT | (owned ? ENG_TF_OWNED : 0u);
    return 0;
}

void eng_tensor_free(EngTensor *t)
{
    if (!t) return;
    /* Only free what this tensor actually owns. A view, a mapping and a cache slot all
     * have data != NULL and all belong to someone else. */
    if ((t->flags & ENG_TF_OWNED) && t->data) free(t->data);
    eng_tensor_zero(t);
}

int eng_tensor_is_resident(const EngTensor *t)
{
    return t && t->data && (t->flags & ENG_TF_RESIDENT) ? 1 : 0;
}

int eng_tensor_is_dense(const EngTensor *t)
{
    if (!t || t->ndim <= 0) return 0;
    for (int i = 0; i < t->ndim; i++)
        if (t->stride[i] != 0) return 0;
    return 1;
}

int eng_tensor_view_rows(EngTensor *out, const EngTensor *src,
                         int64_t first, int64_t count)
{
    if (!out || !src) return -1;
    if (!eng_tensor_is_dense(src)) return -1;          /* strided slicing not supported yet */

    const int64_t rows = eng_tensor_rows(src);
    if (first < 0 || count < 0 || first + count > rows) return -1;

    const int64_t cols = eng_tensor_cols(src);
    const int64_t rb   = eng_dtype_row_bytes(src->dtype, cols);
    if (rb < 0) return -1;    /* row is not a whole number of blocks: not addressable */

    eng_tensor_zero(out);
    out->name   = src->name;
    out->dtype  = src->dtype;
    out->ndim   = 2;
    out->shape[0] = cols;
    out->shape[1] = count;
    out->nbytes = count * rb;
    out->scales = src->scales;

    const int64_t byte_off = first * rb;
    if (src->data) {
        out->data  = (unsigned char *)src->data + byte_off;
        out->flags = ENG_TF_RESIDENT | ENG_TF_VIEW
                   | (src->flags & (ENG_TF_MAPPED | ENG_TF_CACHED));
    } else {
        out->store    = src->store;
        out->file_off = src->file_off + byte_off;
        out->flags    = ENG_TF_VIEW;
    }
    return 0;
}

void eng_tensor_describe(const EngTensor *t, char *buf, size_t cap)
{
    if (!buf || cap == 0) return;
    if (!t) { snprintf(buf, cap, "(null)"); return; }

    char dims[96];
    size_t n = 0;
    dims[0] = '\0';
    for (int i = 0; i < t->ndim && n < sizeof dims - 1; i++) {
        const int w = snprintf(dims + n, sizeof dims - n, "%s%lld",
                               i ? "," : "", (long long)t->shape[i]);
        if (w < 0) break;
        n += (size_t)w;
    }

    const double mb = (double)t->nbytes / 1e6;
    if (eng_tensor_is_resident(t))
        snprintf(buf, cap, "%s %s [%s] %.2f MB ram%s",
                 t->name ? t->name : "(anon)", eng_dtype_name(t->dtype), dims, mb,
                 (t->flags & ENG_TF_VIEW) ? " view" : "");
    else if (t->store)
        snprintf(buf, cap, "%s %s [%s] %.2f MB disk@0x%llx",
                 t->name ? t->name : "(anon)", eng_dtype_name(t->dtype), dims, mb,
                 (unsigned long long)t->file_off);
    else
        snprintf(buf, cap, "%s %s [%s] %.2f MB unbound",
                 t->name ? t->name : "(anon)", eng_dtype_name(t->dtype), dims, mb);
}
