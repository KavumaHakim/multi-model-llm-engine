/* SPDX-License-Identifier: Apache-2.0 */
/*
 * format.h - model container files, behind one interface.
 *
 * A format answers three questions and nothing else:
 *   1. what tensors are in here, by name, in O(1);
 *   2. what does the container say about the model (metadata key/value);
 *   3. where do the bytes come from (each tensor carries an EngStorage + offset).
 *
 * It does NOT know what the tensors mean. Mapping "blk.12.attn_q.weight" onto a Q
 * projection is the model backend's job (models/<arch>/), and mapping it onto a cache
 * slot is the runtime's. Keeping that boundary is what lets safetensors and GGUF be
 * siblings rather than two code paths inside one loader.
 *
 * WHY TENSORS CARRY THEIR OWN STORAGE HANDLE
 *   Safetensors checkpoints are SHARDED -- K3's is 96 files -- so "the file" is not a
 *   single thing and a format-level read(off, n) would need a shard argument that GGUF
 *   has no use for. Putting an EngStorage pointer on each tensor makes the sharding
 *   invisible above this layer: the caller reads a tensor, not a shard.
 */
#ifndef ENG_FORMAT_H
#define ENG_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#include "storage.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngFormat EngFormat;

struct EngFormat {
    const char *name;        /* "safetensors", "gguf" */
    const char *path;        /* borrowed */

    int64_t (*ntensors)(EngFormat *f);

    /* O(1). NULL when absent, which callers MUST treat as fatal: a missing weight
     * read as zeros produces a model that runs and is quietly wrong. */
    const EngTensor *(*find)(EngFormat *f, const char *name);

    /* Iteration in the container's own order, which for GGUF is also its layout
     * order -- that is what lets the streamer read a layer sequentially. */
    const EngTensor *(*at)(EngFormat *f, int64_t i);

    /* Metadata. Return 0 on success, non-zero when the key is absent or is a
     * different type. No defaults are substituted: k3_cfg.h's refuse-rather-than-guess
     * policy exists because a config half-understood yields a model that loads, runs
     * and is architecturally wrong. */
    int (*meta_str)(EngFormat *f, const char *key, const char **out);
    int (*meta_i64)(EngFormat *f, const char *key, int64_t *out);
    int (*meta_f64)(EngFormat *f, const char *key, double *out);

    /* Architecture string used to select a backend, or NULL when the container does
     * not declare one (safetensors does not; its config.json is read separately). */
    const char *(*arch)(EngFormat *f);

    void (*close)(EngFormat *f);

    void *ctx;
};

/* Open by sniffing: a GGUF magic, or a directory / file of safetensors. Returns NULL
 * when nothing claims it. */
EngFormat *eng_format_open(const char *path);

/* Explicit openers, for when the caller already knows. */
EngFormat *eng_format_open_safetensors(const char *dir);

/* Materialise a tensor's bytes into dst, which must hold t->nbytes. Returns bytes
 * read. This is the plain path; the streaming tier uses read_aligned via the tensor's
 * storage handle directly. */
int64_t eng_format_read_tensor(const EngTensor *t, void *dst);

/* Sum of every tensor's nbytes. Used by `inspect` and by the planner. */
int64_t eng_format_total_bytes(EngFormat *f);

void eng_format_report(EngFormat *f, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* ENG_FORMAT_H */
