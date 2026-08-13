/* SPDX-License-Identifier: Apache-2.0 */
/*
 * streamer.h - blocks walked in a fixed cyclic order, with a pinned prefix and an
 * overlapped read.
 *
 * WHY THIS IS NOT THE CACHE
 *   src/storage/cache.c is LRU, and LRU is the WORST POSSIBLE POLICY for this access
 *   pattern. A transformer walks its layers 0, 1, ... N-1 and then does it again for
 *   the next token. With fewer slots than blocks, by the time the walk returns to
 *   block 0 it is exactly the least recently used thing and has just been evicted, so
 *   the hit rate is ZERO no matter how much memory is added. Every extra gigabyte buys
 *   nothing.
 *
 *   So this pins a PREFIX of K blocks and streams the rest through a small ring. The
 *   hit rate is then exactly K/N, deterministically, and each extra gigabyte buys its
 *   fair share. The cache keeps LRU because expert reuse is data-dependent, which is
 *   the opposite situation. Choosing between the two is the planner's job; neither
 *   pretends to cover both cases.
 *
 * WHY PREFETCH IS SAFE HERE AND WOULD NOT BE ELSEWHERE
 *   The walk order is FIXED. The next block is always known, so there is nothing to
 *   predict and a hint can never be wrong. That is what lets one reader thread load
 *   block L+1 while the caller computes on block L. Measured on K3's released
 *   checkpoint this took 71.75 s/token to 42.27 s/token, a 1.70x improvement, and beat
 *   running the same model with four times the memory and no overlap.
 *
 * THE TWO-SLOT RULE IS A CORRECTNESS REQUIREMENT, NOT A TUNING KNOB
 *   The reader claims the next ring slot for the incoming block. With only ONE ring
 *   slot, that is necessarily the slot the caller is computing on right now, so the
 *   worker reads block L+1 straight over block L's bytes mid-computation.
 *
 *   Nothing detects it. The read succeeds, no bound pointer changes, the run completes,
 *   and the model emits fluent wrong output. Measured on K3's real checkpoint, the same
 *   prompt that gives
 *       17374 20829 10 427 414 1008 606 142957
 *   instead produced
 *       32609 2329 146429 2539 11 152834 44449 7569
 *   with no diagnostic of any kind.
 *
 *   So the reader is started ONLY when the ring has at least two slots. Below that the
 *   streamer is synchronous, which is correct and merely slower. eng_streamer_is_async()
 *   reports which path is live, and the constructor says so on stdout, because a user
 *   who budgeted for overlap and silently did not get it would draw wrong conclusions
 *   from the timings.
 *
 * WHY PINNED BLOCKS ARE NOT UNIFORM SLOTS
 *   Blocks differ in size, sometimes a lot: K3's layer 0 carries a dense MLP and is
 *   2.34 GB against 1.27 GB for a normal layer. Sizing every slot for the largest would
 *   waste about half the budget. So pinned blocks get EXACT-size allocations and only
 *   the streaming ring is uniform, sized to the largest block that can still arrive in
 *   it.
 */
#ifndef ENG_STREAMER_H
#define ENG_STREAMER_H

#include <stddef.h>
#include <stdint.h>

#include "storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Where one block lives in the backing store. Blocks are expected to be laid out so
 * that reading one is a single sequential range, which is what makes streaming cheap;
 * the streamer does not require it, but a container that scatters a block will read
 * slowly and there is nothing this layer can do about that. */
typedef struct {
    int64_t off;
    int64_t nbytes;
} EngBlock;

typedef struct EngStreamer EngStreamer;

typedef struct {
    const char *name;         /* diagnostics, e.g. "trunk" or "layer" */
    EngStorage *store;
    const EngBlock *blocks;   /* copied; caller may free after the call */
    int      nblocks;
    int64_t  budget_bytes;

    /* Per-slot scratch appended after the payload, for callers that must expand part of
     * a block in place (K3 widens a handful of bf16 vectors to fp32 at bind time).
     * Zero when unused. */
    int64_t  extra_bytes;

    int      ring_want;       /* desired ring slots; 2 is the minimum that allows async */
    int      async;           /* 0 forces the synchronous path even when the ring allows */
    int      hugepages;       /* 1 on, 0 off, -1 auto (on unless ENG_NOHUGE) */
    int      quiet;           /* suppress the construction banner */
} EngStreamerCfg;

EngStreamer *eng_streamer_create(const EngStreamerCfg *cfg);
void         eng_streamer_destroy(EngStreamer *s);

/* Make `block` resident and return a pointer to its bytes, or NULL on failure.
 *
 * The pointer stays valid until the next eng_streamer_get() for a block that is not
 * pinned -- a ring slot can be recycled. Pinned blocks are valid for the streamer's
 * lifetime. Callers that need a block to outlive the next get() must pin it by
 * arranging for it to fall in the prefix, which is a budget decision, not a runtime one.
 *
 * When extra_bytes was non-zero, *extra receives the scratch area for this slot; pass
 * NULL if unused. */
unsigned char *eng_streamer_get(EngStreamer *s, int block, unsigned char **extra);

/* Start an asynchronous read of `block` if it is not already resident, pinned, or
 * in flight. A no-op when the synchronous path is live. Never blocks. */
void eng_streamer_prefetch(EngStreamer *s, int block);

/* ---------------------------------------------------------------- diagnostics -- */

typedef struct {
    uint64_t hits, misses;
    uint64_t bytes_read;
    double   load_seconds;    /* time inside the read loop only, i.e. a DEVICE rate */
    int      npin;            /* blocks 0..npin-1 are pinned */
    int      nslot;           /* ring slots */
    int      nblocks;
    int64_t  slot_bytes;      /* ring slot stride, payload + extra, aligned */
    int64_t  pinned_bytes;
    int      async;           /* 1 when the reader thread is running */
} EngStreamerStats;

void eng_streamer_stats(const EngStreamer *s, EngStreamerStats *out);
int  eng_streamer_is_async(const EngStreamer *s);
void eng_streamer_report(const EngStreamer *s, const char *label);

/* Hit rate this configuration will achieve on a full cyclic walk, known in advance
 * because the policy is deterministic: npin/nblocks. The planner uses this to decide
 * how to split a budget without having to run anything. */
double eng_streamer_predicted_hit_rate(const EngStreamer *s);

#ifdef __cplusplus
}
#endif

#endif /* ENG_STREAMER_H */
