/* SPDX-License-Identifier: Apache-2.0 */
/*
 * tensor.h - one type that describes a weight whether or not it is in RAM.
 *
 * THE PROBLEM THIS SOLVES
 *   K3 had no tensor type. It had K3Tensor (file metadata: name, shard, dtype, offset,
 *   nbytes -- no data pointer, no residency) and, separately, tagged raw pointers
 *   (`const void *W` plus `int wdt`) with shapes passed as loose `int in, int out`
 *   arguments. Neither can express "this weight exists, here is its shape and dtype,
 *   and it is currently on disk at offset X" -- which is exactly the statement a
 *   streaming engine needs to make about most of its weights most of the time.
 *
 *   So a model could not say "I need tensor X" and let the runtime decide whether X
 *   comes from RAM, cache, mmap or disk. That decision was hardcoded at each call site.
 *
 * THE RESIDENCY MODEL
 *   A tensor is a DESCRIPTOR. `data` is NULL until something materialises it. Where it
 *   comes from is the storage layer's business, not the model's:
 *
 *     data != NULL                     already in RAM; use it
 *     data == NULL && store != NULL    fetch from `store` at `file_off`
 *     data == NULL && store == NULL    unbound; a bug if anything reads it
 *
 *   ENG_TF_OWNED says this tensor's buffer was allocated for it and must be freed with
 *   it. A tensor pointing into a cache slot, an mmap, or a larger layer blob is NOT
 *   owned -- freeing it would take the whole arena with it. Ownership is a property of
 *   the tensor, recorded explicitly, because getting it implicit is how a refactor
 *   introduces a double free.
 *
 * STRIDES
 *   Stored in ELEMENTS, not bytes, and only meaningful for dense dtypes -- a stride
 *   into the middle of a 256-element Q4_K superblock does not address anything. So
 *   eng_tensor_view() refuses to slice a blocked dtype along a non-block boundary
 *   rather than producing a descriptor that looks fine and reads garbage.
 *
 *   stride[i] == 0 for every i means "dense row-major", which is what every weight in
 *   both K3 and Qwen3 actually is. The field exists so that transposed or strided views
 *   are expressible later without changing the struct.
 *
 * SHAPE CONVENTION
 *   shape[0] is the FASTEST-varying axis, matching GGUF (`ne[]`) and ggml. So a
 *   [4096, 151936] embedding is 151,936 rows of 4,096 elements, and the "row" for
 *   row-scaled dtypes and for matmul purposes is shape[0] wide. This is the opposite of
 *   numpy's convention and the same as the file format's, and it is chosen to match the
 *   file because every off-by-a-transpose bug in a loader comes from converting between
 *   the two more times than necessary.
 */
#ifndef ENG_TENSOR_H
#define ENG_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#include "dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENG_MAX_DIMS 4

struct EngStorage;   /* storage/storage.h; a tensor only ever holds a pointer */

enum {
    ENG_TF_OWNED    = 1u << 0,  /* free(data) belongs to this tensor            */
    ENG_TF_RESIDENT = 1u << 1,  /* data is valid right now                      */
    ENG_TF_MAPPED   = 1u << 2,  /* data points into an mmap, do not free        */
    ENG_TF_CACHED   = 1u << 3,  /* data points into a cache slot, may be evicted */
    ENG_TF_PINNED   = 1u << 4,  /* cache must not evict while set               */
    ENG_TF_VIEW     = 1u << 5   /* data points inside another tensor            */
};

typedef struct EngTensor {
    const char       *name;       /* borrowed; the format's string pool owns it */
    EngDType          dtype;
    int               ndim;
    int64_t           shape[ENG_MAX_DIMS];
    int64_t           stride[ENG_MAX_DIMS];  /* ELEMENTS; all zero = dense row-major */
    int64_t           nbytes;

    struct EngStorage *store;     /* NULL when this is a pure-RAM tensor */
    int64_t           file_off;   /* absolute offset within `store`      */

    void             *data;       /* NULL until materialised */
    uint32_t          flags;

    /* External-scale dtypes (MXFP4) need a sibling. NULL otherwise. Held as a plain
     * pointer rather than a name so resolving it is done once, at bind time. */
    struct EngTensor *scales;
} EngTensor;

/* ---------------------------------------------------------------- construction -- */

/* Zero a tensor. Call this before filling one on the stack: several fields select a
 * code path by being NULL, and K3's headers carry the same warning for the same
 * reason. */
void eng_tensor_zero(EngTensor *t);

/* Describe a file-backed tensor. Computes nbytes from dtype and shape; returns -1 and
 * leaves *t zeroed if the shape and dtype are inconsistent (see eng_tensor_nbytes). */
int eng_tensor_init_file(EngTensor *t, const char *name, EngDType dt,
                         const int64_t *shape, int ndim,
                         struct EngStorage *store, int64_t file_off);

/* Describe a tensor that already lives in RAM. Does not copy. `owned` records whether
 * free() should follow this tensor. */
int eng_tensor_init_mem(EngTensor *t, const char *name, EngDType dt,
                        const int64_t *shape, int ndim, void *data, int owned);

/* Free only what this tensor owns, then zero it. Safe on an already-zeroed tensor and
 * safe to call twice. Never frees a view, a mapping or a cache slot. */
void eng_tensor_free(EngTensor *t);

/* ------------------------------------------------------------------- geometry -- */

int64_t eng_tensor_numel(const EngTensor *t);

/* Byte size implied by dtype and shape. Handles the row-scaled case by splitting into
 * rows of shape[0]. Returns -1 when the shape is not expressible in the dtype (a
 * partial block, a negative dimension, an unknown dtype) -- callers must treat that as
 * fatal rather than substituting a computed guess. */
int64_t eng_tensor_nbytes(EngDType dt, const int64_t *shape, int ndim);

int eng_tensor_is_dense(const EngTensor *t);   /* all strides zero, or exactly packed */
int eng_tensor_is_resident(const EngTensor *t);

/* Rows and columns as the matmul kernels see them: cols = shape[0] (fastest axis),
 * rows = product of the remaining dimensions. A 1-D tensor is one row. */
int64_t eng_tensor_cols(const EngTensor *t);
int64_t eng_tensor_rows(const EngTensor *t);

/* --------------------------------------------------------------------- views -- */

/* A view of `count` rows starting at row `first`. The view borrows the parent's data
 * (or file offset when the parent is not resident) and never owns it.
 *
 * Refuses, returning -1, when the slice would not be addressable:
 *   - the parent has non-trivial strides;
 *   - the range is out of bounds;
 *   - the dtype is blocked and a row is not a whole number of blocks.
 * That last case is the one worth having: slicing a Q4_K matrix at an arbitrary row is
 * only valid because its rows happen to be block-aligned, and a dtype where they are
 * not would otherwise yield a descriptor that looks correct and decodes to noise. */
int eng_tensor_view_rows(EngTensor *out, const EngTensor *src,
                         int64_t first, int64_t count);

/* ------------------------------------------------------------------ diagnostic -- */

/* "name f32 [4096,151936] 350.1 MB disk@0x1a2b" into buf. Always NUL-terminates. */
void eng_tensor_describe(const EngTensor *t, char *buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* ENG_TENSOR_H */
