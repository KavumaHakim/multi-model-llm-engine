/* SPDX-License-Identifier: Apache-2.0 */
/*
 * hwinfo.h - what this machine actually is.
 *
 * The planner cannot choose a sensible configuration from arithmetic alone: whether the
 * weights fit, how many threads help, and whether streaming is affordable are all
 * properties of the host. K3 detected exactly one of these (total RAM, by reading
 * /proc/meminfo, Linux only, silently returning 0 elsewhere) and hardcoded the rest.
 *
 * TWO THINGS HERE ARE DELIBERATELY RUNTIME RATHER THAN COMPILE TIME
 *
 *   CPU FEATURES. K3 selected its AVX2 kernels with #if defined(__AVX2__), which bakes
 *   the decision into the binary: a build for this machine crashes on an older one, and
 *   a portable build leaves the vector units idle on a newer one. Detecting at runtime
 *   lets ONE binary carry both paths and pick per host. The compile-time macro still
 *   decides whether the code was EMITTED, so both are reported separately -- a kernel
 *   can be unavailable because the CPU lacks it or because the build omitted it, and
 *   those need different fixes.
 *
 *   CORE COUNTS, physical AND logical. These are not interchangeable and the difference
 *   is measurable. On the reference machine (i5-6300U, 2 physical / 4 logical), 7
 *   interleaved reps quoting the minimum gave:
 *
 *       threads   bf16 matmul        MXFP4 matmul
 *          1      26.12 ms  1.00x     9.05 ms  1.00x
 *          2      15.30 ms  1.71x     4.69 ms  1.93x
 *          3      13.88 ms  1.88x     3.59 ms  2.52x
 *          4      14.09 ms  1.85x     3.18 ms  2.85x
 *
 *   bf16 saturates at 2-3 threads and REGRESSES at 4; MXFP4 keeps scaling. The two
 *   kernels want different thread counts on the same CPU, because bf16 matmul is
 *   memory-bandwidth-bound while MXFP4 reads ~7.5x fewer bytes per multiply and sits
 *   further from the bandwidth roof. A planner that just uses nproc is right only by
 *   accident. See docs/baseline-m0.md.
 */
#ifndef ENG_HWINFO_H
#define ENG_HWINFO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    ENG_ISA_SSE2    = 1u << 0,
    ENG_ISA_AVX     = 1u << 1,
    ENG_ISA_AVX2    = 1u << 2,
    ENG_ISA_FMA     = 1u << 3,
    ENG_ISA_AVX512F = 1u << 4,
    ENG_ISA_AVX512BW= 1u << 5,
    ENG_ISA_NEON    = 1u << 6
};

typedef enum {
    ENG_STORE_UNKNOWN = 0,
    ENG_STORE_ROTATIONAL,     /* seek cost dominates: prefer sequential, deep batches */
    ENG_STORE_SSD             /* includes NVMe; random reads are affordable */
} EngStorageClass;

typedef struct {
    char     cpu_name[64];
    int      cores_physical;
    int      cores_logical;
    uint32_t isa;              /* what the CPU supports */
    uint32_t isa_built;        /* what this binary actually contains */

    int64_t  ram_total;
    int64_t  ram_available;    /* what the OS thinks can be handed out now */

    EngStorageClass storage;
    int64_t  storage_free;     /* on the model's filesystem, when known */

    int      l3_bytes;
} EngHwInfo;

/* Fills every field it can and leaves the rest zero. Never fails: an unknown value is
 * reported as zero and the planner treats zero as "unknown" rather than as "none",
 * because a detection gap must not silently become a budget of nothing. */
void eng_hwinfo_detect(EngHwInfo *hw);

/* Storage class for the filesystem holding `path`, and its free space. Separate from
 * detect() because it needs a path and detect() does not take one. */
void eng_hwinfo_probe_path(EngHwInfo *hw, const char *path);

int  eng_hwinfo_has(const EngHwInfo *hw, uint32_t isa_bit);
void eng_hwinfo_isa_string(uint32_t isa, char *buf, size_t cap);
void eng_hwinfo_report(const EngHwInfo *hw, const char *label);

#ifdef __cplusplus
}
#endif

#endif /* ENG_HWINFO_H */
