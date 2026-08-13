/* k3_trunk.h - K3's packed dense trunk, now a thin binding over the generic streamer.
 *
 * WHAT MOVED, AT M3
 *   The pinned-prefix policy, the ring, the asynchronous reader, the fixed-point budget
 *   sizing and the O_DIRECT read path now live in src/storage/streamer.c. None of that
 *   was K3-specific: it is what any architecture wants when it walks its layers in a
 *   fixed cyclic order, which is all of them.
 *
 *   What remains here is genuinely K3's: the trunk.json layout produced by
 *   tools/pack_trunk.py, the per-layer tensor tables it describes, and binding those
 *   tensors onto the kernels' weight structs by name.
 *
 * WHY STREAM THE TRUNK AT ALL, which is the argument this file exists to serve
 *   The engine holds 108.81 GB of trunk plus 4.70 GB of embed/lm_head. Quantising it
 *   down is the obvious idea and it is the wrong one: K3's technical report section
 *   4.1.4 says the experts are MXFP4 with quantisation-aware training "while all
 *   non-expert components ... remain in higher precision". That list IS this trunk.
 *   Measured on 31 real attention tensors (docs/data/trunk-quantisation.txt), post-hoc
 *   int4 costs 17.4% mean relative weight reconstruction error against 0.96% for int8,
 *   an ~18x gap consistent across every tensor sampled. That rules out 4-bit on a trunk
 *   never trained for it.
 *
 *   Streaming costs ZERO error. The bytes are the checkpoint's own bytes.
 *
 * WHY IT IS AFFORDABLE
 *   The trunk access order is FIXED: layer 0, 1, ... 92, every single token. So the next
 *   read is always known and can be issued while the current layer computes. Measured on
 *   the released checkpoint at the laptop preset this took 71.75 s/token to 42.27
 *   s/token, a 1.70x improvement, and beat running the same model with four times the
 *   memory and no overlap.
 *
 * LAYOUT
 *   tools/pack_trunk.py copies each layer's trunk -- ONE contiguous run in its shard --
 *   into trunk.bin and records offsets in trunk.json, padding runs to 4096 so O_DIRECT
 *   works. Loading a layer is therefore one read from a known offset, and a tensor's
 *   position inside a slot is (its absolute shard offset - the run start).
 */
#ifndef K3_TRUNK_H
#define K3_TRUNK_H

#include "k3.h"
#include "k3_bind.h"

#include "storage.h"
#include "streamer.h"

#define K3_TRUNK_ALIGN 4096   /* pack_trunk.py pads runs to this so O_DIRECT works */

typedef struct {
    char    *name;
    int64_t  off;          /* byte offset WITHIN the layer run */
    int64_t  nbytes;
    int      dtype;        /* K3Dtype */
} K3TrunkTensor;

typedef struct {
    int64_t  file_off;     /* offset in trunk.bin */
    int64_t  nbytes;
    K3TrunkTensor *t;
    int      nt;
} K3TrunkLayer;

typedef struct {
    int           n_layers;
    K3TrunkLayer *lay;

    /* Backs every K3TrunkTensor.name, so it must outlive the whole struct. Owned here
     * and freed by k3_trunk_close; do not free the parser arena separately. */
    char         *json_arena;

    EngStorage   *store;
    EngStreamer  *sm;
    int           direct;       /* 1 when reads bypass the page cache */

    /* Mirrored from the streamer after each bind, so callers reading these directly
     * (src/cli/k3_run.c) keep working. The authority is `sm`. */
    uint64_t      hits, misses;
    uint64_t      bytes_read;
    double        load_seconds;
} K3Trunk;

/* budget_bytes sizes the pinned prefix and the ring. Returns 0 on success. */
int  k3_trunk_open(K3Trunk *tr, const char *dir, const K3Cfg *c, int64_t budget_bytes);
void k3_trunk_close(K3Trunk *tr);

/* Make layer L resident and point b's weight pointers at it. b must already have been
 * prepared by k3_bind_layer_mem, which resolves the layer's tensor shapes once. */
int  k3_trunk_bind(K3Trunk *tr, const K3Cfg *c, int L, K3LayerBind *b);

/* Start an asynchronous read of layer L, if it is not resident. Safe to call for a
 * layer that is pinned or already loaded (it becomes a no-op). */
void k3_trunk_prefetch(K3Trunk *tr, int L);

void k3_trunk_report(const K3Trunk *tr, const char *label);

#endif /* K3_TRUNK_H */
