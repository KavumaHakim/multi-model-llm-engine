/* SPDX-License-Identifier: Apache-2.0 */
/*
 * storage.h - byte ranges, from wherever they live.
 *
 * WHAT THIS ABSTRACTS
 *   K3's readers each opened their own descriptors and made their own pread calls:
 *   k3_st.c for the safetensors shards, k3_trunk.c for trunk.bin. Both implemented the
 *   same O_DIRECT alignment dance, and both counted their own statistics. Neither could
 *   be pointed at a different kind of backing store.
 *
 *   A storage backend answers exactly one question: give me bytes [off, off+n). It does
 *   not know what a tensor is, what a layer is, or which model is running. That is the
 *   layering rule from docs/architecture-report.md §12 -- storage never learns what a
 *   layer is -- and it is what lets the same streamer serve K3's packed trunk and a
 *   Qwen3 GGUF without a conditional.
 *
 * THE ALIGNED READ IS NOT read() WITH A FLAG
 *   O_DIRECT requires the file offset, the length AND the buffer address all to be
 *   multiples of ENG_IO_ALIGN. A tensor starts wherever the file put it, so the read is
 *   WIDENED OUTWARD to the enclosing aligned window and the caller is told where its
 *   payload begins inside the buffer. buf must therefore hold nbytes + 2*ENG_IO_ALIGN
 *   and be page-aligned.
 *
 *   This is worth the complication for streamed weights specifically: such a weight is
 *   read once and evicted, so the page cache can only copy it a second time and push
 *   out something useful. K3 measured the buffered path at 1,247 MB/s on a device that
 *   sustains 6,400 under a 32 GB cgroup cap (see k3_st.h).
 *
 *   Every backend must fall back to a buffered read, setting *payload_off = 0, when the
 *   direct path is unavailable. Correctness never depends on which path runs.
 *
 * STATISTICS ARE PART OF THE INTERFACE, not an add-on. The brief requires the engine to
 * be able to say whether it is storage-bound, and that answer is impossible after the
 * fact if the byte and time counters live in whichever caller happened to make the
 * read. They are updated by the backend, on every path.
 */
#ifndef ENG_STORAGE_H
#define ENG_STORAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Direct-I/O alignment. Offset, length and buffer must all be multiples of this.
 * 4096 matches K3_ST_ALIGN and K3_TRUNK_ALIGN, and every mainstream filesystem. */
#define ENG_IO_ALIGN 4096

typedef struct EngStorage EngStorage;

struct EngStorage {
    const char *kind;    /* "file", "mmap", ... for diagnostics */
    const char *path;    /* borrowed, may be NULL */

    /* Plain read of [off, off+nbytes) into dst. Returns bytes read, which is < nbytes
     * only on a short read or error. Loops internally: a partial pread is normal. */
    int64_t (*read)(EngStorage *s, int64_t off, int64_t nbytes, void *dst);

    /* Direct read where available. See the header comment: buf must hold
     * nbytes + 2*ENG_IO_ALIGN and be page-aligned, and *payload_off receives where the
     * requested bytes begin inside buf. Returns payload bytes available.
     *
     * MAY be the same function as read() on a backend with no direct path, in which
     * case it sets *payload_off = 0. Callers must honour *payload_off either way
     * rather than assuming zero. */
    int64_t (*read_aligned)(EngStorage *s, int64_t off, int64_t nbytes,
                            void *buf, int64_t bufcap, int64_t *payload_off);

    /* Borrowed pointer to a mapped range, or NULL when this backend cannot map. The
     * mapping stays valid until close(). Callers must not free it. */
    const void *(*map)(EngStorage *s, int64_t off, int64_t nbytes);

    int64_t (*size)(EngStorage *s);
    void    (*close)(EngStorage *s);

    void *ctx;

    /* stats, maintained by the backend on every read path */
    uint64_t reads;
    uint64_t bytes_read;
    double   read_seconds;
    uint64_t direct_reads;    /* subset of `reads` that used the direct path */
};

/* Open a file backend. want_direct requests a second O_DIRECT descriptor; if the
 * filesystem refuses, the backend still opens and read_aligned() falls back to
 * buffered. Returns NULL only if the file itself cannot be opened. */
EngStorage *eng_storage_open_file(const char *path, int want_direct);

/* 1 when the backend actually obtained a direct descriptor. Diagnostic only. */
int eng_storage_is_direct(const EngStorage *s);

/* Page-aligned allocation sized for a read_aligned() of nbytes, i.e.
 * nbytes + 2*ENG_IO_ALIGN rounded up. Free with eng_storage_free_aligned. */
void *eng_storage_alloc_aligned(int64_t nbytes, int64_t *out_cap);
void  eng_storage_free_aligned(void *p);

void eng_storage_report(const EngStorage *s, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* ENG_STORAGE_H */
