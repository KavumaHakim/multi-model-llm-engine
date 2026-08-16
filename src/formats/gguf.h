/* SPDX-License-Identifier: Apache-2.0 */
/*
 * gguf.h - the GGUF container.
 *
 * WHAT GGUF IS, and why it needs a reader rather than a struct cast
 *   A header of key/value metadata, then a table of tensor descriptors, then the tensor
 *   data. Everything is little-endian and length-prefixed; nothing is fixed-width, so
 *   the whole header must be walked to find its end. In the target file that header is
 *   5,956,416 bytes -- almost 6 MB -- because the tokenizer ships 151,936 token strings
 *   inside it.
 *
 * TWO PROPERTIES OF THE TARGET FILE THAT SHAPE THIS READER
 *
 *   1. TENSOR OFFSETS ARE RELATIVE TO THE DATA SECTION, not to the file. The absolute
 *      position is data_start + offset, and data_start is the header end rounded up to
 *      general.alignment (32 here, NOT the 4096 an O_DIRECT reader would want). Treating
 *      the stored offset as absolute reads 5.9 MB early and produces noise that still
 *      decodes to finite floats.
 *
 *   2. THE LAYOUT IS ALREADY STREAMABLE. Offsets are monotonically non-decreasing in
 *      table order, and each block's tensors are adjacent, so a layer is already ONE
 *      contiguous run -- the property tools/pack_trunk.py had to CREATE for K3. So the
 *      repacker is optional for Qwen3, and this reader is expected to be the whole
 *      storage story for it. That claim is verified rather than assumed: see
 *      gguf_layout_is_sequential().
 *
 * ARRAYS ARE NOT MATERIALISED. The tokenizer's 151,936 token strings and their merges
 * are the bulk of the header. Copying them into a string table at open time would cost
 * megabytes for a caller that only wants block_count. Instead the header blob is kept
 * and array values point into it, with accessors that walk on demand.
 *
 * ENDIANNESS. GGUF is little-endian. Every multi-byte read goes through a byte-assembling
 * helper rather than a struct cast, so the reader is correct on a big-endian host and
 * cannot trip an unaligned load -- the tensor table's u64 dimensions are not aligned to
 * 8 in general, because they follow a variable-length name.
 */
#ifndef ENG_GGUF_H
#define ENG_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "dtype.h"
#include "storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF metadata value types, as stored. */
typedef enum {
    GGUF_T_UINT8 = 0, GGUF_T_INT8 = 1, GGUF_T_UINT16 = 2, GGUF_T_INT16 = 3,
    GGUF_T_UINT32 = 4, GGUF_T_INT32 = 5, GGUF_T_FLOAT32 = 6, GGUF_T_BOOL = 7,
    GGUF_T_STRING = 8, GGUF_T_ARRAY = 9, GGUF_T_UINT64 = 10, GGUF_T_INT64 = 11,
    GGUF_T_FLOAT64 = 12
} GgufType;

#define GGUF_MAX_DIMS 4

typedef struct {
    const char *name;        /* into the header blob; NUL-terminated copy */
    int         n_dims;
    int64_t     shape[GGUF_MAX_DIMS];   /* ne[] order: shape[0] is fastest-varying */
    uint32_t    ggml_type;   /* as stored */
    EngDType    dtype;       /* mapped; ENG_DT_INVALID when unrecognised */
    int64_t     rel_off;     /* as stored: relative to the data section */
    int64_t     file_off;    /* absolute, computed */
    int64_t     nbytes;
} GgufTensor;

typedef struct {
    const char *key;
    GgufType    type;
    /* Scalars. Integers land in `i` regardless of width and signedness, floats in `f`,
     * strings in `str`/`slen`. One union would need the caller to know the width to read
     * it back, which is exactly the mistake this avoids. */
    int64_t     i;
    double      f;
    const char *str;
    int64_t     slen;
    /* Arrays: not materialised. */
    GgufType    arr_type;
    int64_t     arr_len;
    const unsigned char *arr_data;   /* first element, inside the header blob */
} GgufKV;

typedef struct {
    EngStorage *store;
    int         owns_store;

    uint32_t    version;
    int64_t     n_tensors;
    int64_t     n_kv;

    unsigned char *blob;      /* the whole header, kept for array access */
    int64_t        blob_size;

    /* Keys and tensor names, copied out and NUL-terminated. They cannot be terminated
     * in place: a GGUF string is length-prefixed and the byte after it is already the
     * next field. */
    char          *strpool;
    int64_t        strcap, strused;

    GgufKV      *kv;
    GgufTensor  *t;

    int64_t     alignment;    /* general.alignment, default 32 */
    int64_t     data_start;   /* absolute offset of the tensor data section */
    int64_t     data_bytes;   /* sum over tensors, from the table */
    int64_t     file_size;
} Gguf;

/* Open and parse the header. Returns 0 on success. Never reads tensor data.
 *
 * Every length in the header is treated as untrusted: a corrupt or hostile file must be
 * rejected with a message, not walked off the end of the blob. */
int  gguf_open(Gguf *g, const char *path);
void gguf_close(Gguf *g);

/* Metadata lookup. All return 0 on success, non-zero when the key is absent or is not
 * of a compatible type -- callers must check rather than defaulting, because a missing
 * architecture field silently defaulted is how a loader builds the wrong model. */
const GgufKV *gguf_find(const Gguf *g, const char *key);
int gguf_u32(const Gguf *g, const char *key, uint32_t *out);
int gguf_i64(const Gguf *g, const char *key, int64_t *out);
int gguf_f32(const Gguf *g, const char *key, float *out);
int gguf_str(const Gguf *g, const char *key, const char **out, int64_t *len);

/* i'th element of a string array, without materialising the array. Returns NULL when
 * out of range. The pointer is into the header blob and is NOT NUL-terminated, so *len
 * is the only way to know where it ends. */
const char *gguf_arr_str(const Gguf *g, const GgufKV *kv, int64_t i, int64_t *len);

/* Tensor lookup by exact name. NULL when absent. */
const GgufTensor *gguf_tensor(const Gguf *g, const char *name);

/* Is the data laid out so a layer is one contiguous forward run?
 *
 * Checked rather than assumed: the streaming design depends on it, and a container that
 * interleaved layers would still load correctly while streaming pathologically. Returns
 * 1 when every tensor's offset is >= its predecessor's. */
int gguf_layout_is_sequential(const Gguf *g);

/* Map a ggml type id to the engine's dtype. ENG_DT_INVALID when unknown. */
EngDType gguf_dtype(uint32_t ggml_type);
const char *gguf_type_name(uint32_t ggml_type);

void gguf_report(const Gguf *g, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* ENG_GGUF_H */
