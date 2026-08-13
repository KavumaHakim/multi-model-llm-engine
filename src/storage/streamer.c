/* SPDX-License-Identifier: Apache-2.0 */
/* streamer.c - see streamer.h. Extracted from src/io/k3_trunk.c. */
#define _POSIX_C_SOURCE 200809L

#include "streamer.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__) || defined(__APPLE__)
#  include <sys/mman.h>
#endif

typedef struct {
    pthread_t       thread;
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    EngStreamer    *s;
    int stop, busy, done;
    int block, slot, result;
} StreamIO;

struct EngStreamer {
    char       *name;
    EngStorage *store;
    EngBlock   *blk;
    int         nblocks;

    unsigned char **pin;       /* [npin] exact-size allocations */
    int             npin;
    int64_t         pinned_bytes;

    unsigned char  *arena;     /* [nslot] uniform ring slots */
    int64_t         slot_bytes;
    int64_t         extra_bytes;
    int             nslot;
    int            *block_of;  /* [nslot] which block occupies each slot, -1 empty */
    int32_t        *slot_of;   /* [nblocks] -1 when not resident */
    int             ring;      /* next ring slot to reuse */

    /* Where the payload begins inside its buffer. Non-zero only when a direct read was
     * widened outward to alignment boundaries: the block's own offset is not required to
     * be aligned, so the read covers the enclosing window and the payload sits somewhere
     * inside it. K3's packed trunk pads every run to 4096, so these are all zero there --
     * but a GGUF layer run is not padded, and assuming zero would return a pointer up to
     * 4095 bytes before the actual data. */
    int32_t        *slot_pad;  /* [nslot]   */
    int32_t        *pin_pad;   /* [npin]    */

    StreamIO   *io;            /* NULL when synchronous */

    uint64_t hits, misses, bytes_read;
    double   load_seconds;
};

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int64_t align_up(int64_t v, int64_t a)
{
    return (v + a - 1) & ~(a - 1);
}

/* 2 MB aligned and hugepage-advised where possible: a direct read pins its destination
 * pages, and a multi-gigabyte slot on 4 KB pages is hundreds of thousands of pins per
 * read. The LENGTH is rounded too, because madvise only covers whole pages and a 2 MB
 * start with a ragged tail leaves the last stretch on small pages. */
static int alloc_direct(void **out, size_t bytes, int huge)
{
    const size_t align = huge ? (2u << 20) : (size_t)ENG_IO_ALIGN;
    const size_t len   = huge ? ((bytes + align - 1) & ~(align - 1)) : bytes;
    if (posix_memalign(out, align, len) != 0) return -1;
#if defined(MADV_HUGEPAGE)
    if (huge) madvise(*out, len, MADV_HUGEPAGE);   /* advisory; failure is not an error */
#endif
    return 0;
}

/* The read itself. Bracketed timing here and nowhere else, so load_seconds is a DEVICE
 * rate: whatever the caller does with the bytes afterwards is deliberately not counted,
 * and a caller that wants the total must measure around eng_streamer_get().
 *
 * Goes through read_aligned rather than read, because the page cache is the one thing
 * that CANNOT help here: each streamed block is read once per pass and never reused
 * before eviction, so buffering copies every byte twice and evicts whatever else was
 * being held. K3 measured its trunk at 1,878 MB/s buffered under a 32 GB cap against
 * 6,553 MB/s unconstrained.
 *
 * *pad receives where the payload starts. The backend falls back to a buffered read
 * (pad 0) whenever the direct path is unavailable, so correctness never depends on it. */
static int load_block(EngStreamer *s, int b, unsigned char *dst, int64_t cap, int32_t *pad)
{
    const EngBlock *bk = &s->blk[b];
    int64_t poff = 0;
    const double t0 = now_s();
    const int64_t got = s->store->read_aligned(s->store, bk->off, bk->nbytes,
                                               dst, cap, &poff);
    s->load_seconds += now_s() - t0;
    if (got != bk->nbytes) {
        fprintf(stderr, "%s streamer: short read on block %d (%lld of %lld)\n",
                s->name, b, (long long)got, (long long)bk->nbytes);
        return -1;
    }
    s->bytes_read += (uint64_t)got;
    *pad = (int32_t)poff;
    return 0;
}

/* ------------------------------------------------------------ the reader thread -- */

static void *io_main(void *arg)
{
    StreamIO *io = (StreamIO *)arg;
    for (;;) {
        pthread_mutex_lock(&io->mu);
        while (!io->busy && !io->stop) pthread_cond_wait(&io->cv, &io->mu);
        if (io->stop) { pthread_mutex_unlock(&io->mu); return NULL; }
        const int b = io->block, slot = io->slot;
        EngStreamer *s = io->s;
        pthread_mutex_unlock(&io->mu);

        /* Outside the lock: the destination slot was reserved before the worker was
         * woken, and nothing else may claim it while busy is set. */
        int32_t pad = 0;
        const int rc = load_block(s, b, s->arena + (size_t)slot * s->slot_bytes,
                                  s->slot_bytes, &pad);
        if (rc == 0) s->slot_pad[slot] = pad;

        pthread_mutex_lock(&io->mu);
        io->result = rc;
        io->done = 1;
        io->busy = 0;
        pthread_cond_broadcast(&io->cv);
        pthread_mutex_unlock(&io->mu);
    }
}

/* Collect a prefetch of block b, if one is outstanding for it.
 * Returns 1 when it landed, 0 when there was nothing to collect, -1 on a failed read.
 *
 * PUBLISHING HAPPENS HERE, after the read succeeded -- never when the request was
 * issued. Registering the mapping up front would let a failed read leave a slot
 * claiming a block it does not hold, and the next walk would count it a hit. */
static int io_collect(EngStreamer *s, int b)
{
    StreamIO *io = s->io;
    if (!io) return 0;
    pthread_mutex_lock(&io->mu);
    if ((io->busy || io->done) && io->block == b) {
        while (!io->done && !io->stop) pthread_cond_wait(&io->cv, &io->mu);
        const int rc = io->result;
        const int slot = io->slot;
        if (!io->stop && rc == 0) {
            s->block_of[slot] = b;
            s->slot_of[b] = (int32_t)slot;
            s->misses++;              /* the bytes still moved: not a hit */
        }
        io->done = 0;
        pthread_mutex_unlock(&io->mu);
        return rc == 0 ? 1 : -1;
    }
    pthread_mutex_unlock(&io->mu);
    return 0;
}

/* ----------------------------------------------------------------------- get -- */

unsigned char *eng_streamer_get(EngStreamer *s, int block, unsigned char **extra)
{
    if (!s || block < 0 || block >= s->nblocks) return NULL;
    unsigned char *base;

    if (block < s->npin) {
        unsigned char *buf = s->pin[block];
        if (s->slot_of[block] < 0) {           /* first touch: load once, keep forever */
            int32_t pad = 0;
            const int64_t cap = align_up(s->blk[block].nbytes, ENG_IO_ALIGN)
                              + ENG_IO_ALIGN + s->extra_bytes;
            if (load_block(s, block, buf, cap, &pad) != 0) return NULL;
            s->pin_pad[block] = pad;
            s->slot_of[block] = (int32_t)block;
            s->misses++;
        } else {
            s->hits++;
        }
        base = buf + s->pin_pad[block];
        if (extra) *extra = base + align_up(s->blk[block].nbytes, ENG_IO_ALIGN);
        return base;
    }

    int slot = -1;
    const int landed = io_collect(s, block);
    if (landed < 0) return NULL;
    if (landed > 0) {
        slot = s->slot_of[block];
    } else {
        for (int i = 0; i < s->nslot; i++)
            if (s->block_of[i] == block) { slot = i; break; }
        if (slot >= 0) {
            s->hits++;
        } else {
            slot = s->ring;
            s->ring = (s->ring + 1) % s->nslot;
            if (s->block_of[slot] >= 0) s->slot_of[s->block_of[slot]] = -1;
            /* EMPTY before the read, not after: a failed read must not leave the slot
             * claiming a block it does not hold. */
            s->block_of[slot] = -1;
            int32_t pad = 0;
            if (load_block(s, block, s->arena + (size_t)slot * s->slot_bytes,
                           s->slot_bytes, &pad) != 0)
                return NULL;
            s->slot_pad[slot] = pad;
            s->block_of[slot] = block;
            s->slot_of[block] = (int32_t)slot;
            s->misses++;
        }
    }

    base = s->arena + (size_t)slot * s->slot_bytes + s->slot_pad[slot];
    if (extra) *extra = base + align_up(s->blk[block].nbytes, ENG_IO_ALIGN);
    return base;
}

void eng_streamer_prefetch(EngStreamer *s, int block)
{
    if (!s || block < 0 || block >= s->nblocks) return;
    if (block < s->npin) return;                    /* pinned: nothing to do */
    if (!s->io) return;                             /* synchronous path */
    for (int i = 0; i < s->nslot; i++) if (s->block_of[i] == block) return;

    StreamIO *io = s->io;
    pthread_mutex_lock(&io->mu);
    if (io->busy || io->done || s->slot_of[block] >= 0) {
        pthread_mutex_unlock(&io->mu);
        return;
    }
    const int slot = s->ring;
    s->ring = (s->ring + 1) % s->nslot;
    if (s->block_of[slot] >= 0) s->slot_of[s->block_of[slot]] = -1;
    s->block_of[slot] = -1;
    io->block = block;
    io->slot  = slot;
    io->done  = 0;
    io->busy  = 1;
    pthread_cond_signal(&io->cv);
    pthread_mutex_unlock(&io->mu);
}

/* --------------------------------------------------------------- construction -- */

EngStreamer *eng_streamer_create(const EngStreamerCfg *cfg)
{
    if (!cfg || !cfg->store || !cfg->blocks || cfg->nblocks <= 0) return NULL;

    EngStreamer *s = (EngStreamer *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->name    = strdup(cfg->name ? cfg->name : "block");
    s->store   = cfg->store;
    s->nblocks = cfg->nblocks;
    s->extra_bytes = cfg->extra_bytes > 0 ? cfg->extra_bytes : 0;
    s->blk = (EngBlock *)malloc((size_t)cfg->nblocks * sizeof *s->blk);
    s->slot_of = (int32_t *)malloc((size_t)cfg->nblocks * sizeof *s->slot_of);
    if (!s->name || !s->blk || !s->slot_of) { eng_streamer_destroy(s); return NULL; }
    memcpy(s->blk, cfg->blocks, (size_t)cfg->nblocks * sizeof *s->blk);
    for (int i = 0; i < cfg->nblocks; i++) s->slot_of[i] = -1;

    const int huge = cfg->hugepages < 0 ? (getenv("ENG_NOHUGE") ? 0 : 1)
                                        : (cfg->hugepages ? 1 : 0);
    const int ring_want = cfg->ring_want > 0 ? cfg->ring_want : 2;

    /* SIZING, to a fixed point.
     *
     * The ring slot must fit the largest block that will ever stream through it. Pinning
     * a block removes it from that set, which can shrink the slot, which frees budget,
     * which pins more blocks, which can shrink it again. Iterate. It is monotone and
     * converges in two or three passes, so the loop is bounded. Where npin ends at 0
     * this correctly changes nothing: every block streams and the ring must still hold
     * the biggest of them. */
    int64_t ring_slot = 0, spent = 0;
    int npin = 0, ring = ring_want;
    for (int pass = 0; pass < 4; pass++) {
        int64_t big = 0;
        for (int i = npin; i < s->nblocks; i++)
            if (s->blk[i].nbytes > big) big = s->blk[i].nbytes;
        if (big == 0) big = s->blk[s->nblocks - 1].nbytes;    /* everything pinned */

        /* One extra alignment unit for the widening: a direct read of an unaligned
         * block covers the enclosing window, so the payload can start up to
         * ENG_IO_ALIGN-1 bytes into the slot. Omitting it makes the last block of an
         * unaligned container overrun its slot. */
        int64_t rs = align_up(big, ENG_IO_ALIGN) + ENG_IO_ALIGN + s->extra_bytes;
        rs = align_up(rs, ENG_IO_ALIGN);

        /* The ring itself must fit before anything is pinned. Drop slots rather than
         * overshoot the budget. */
        ring = ring_want;
        while (ring > 1 && (int64_t)ring * rs > cfg->budget_bytes) ring--;

        int64_t sp = (int64_t)ring * rs;
        int np = 0;
        while (np < s->nblocks) {
            const int64_t need = align_up(s->blk[np].nbytes, ENG_IO_ALIGN)
                               + ENG_IO_ALIGN + s->extra_bytes;
            if (sp + need > cfg->budget_bytes) break;
            sp += need;
            np++;
        }
        if (rs == ring_slot && np == npin) { ring_slot = rs; spent = sp; break; }
        ring_slot = rs; npin = np; spent = sp;
    }

    if (ring_slot <= 0 || (int64_t)ring * ring_slot > cfg->budget_bytes) {
        fprintf(stderr, "%s streamer: budget %.2f GB cannot hold even one ring slot "
                        "of %.2f GB\n", s->name,
                (double)cfg->budget_bytes / 1e9, (double)ring_slot / 1e9);
        eng_streamer_destroy(s);
        return NULL;
    }

    s->npin       = npin;
    s->nslot      = ring;
    s->slot_bytes = ring_slot;

    s->pin     = (unsigned char **)calloc((size_t)(npin ? npin : 1), sizeof *s->pin);
    s->pin_pad = (int32_t *)calloc((size_t)(npin ? npin : 1), sizeof *s->pin_pad);
    if (!s->pin || !s->pin_pad) { eng_streamer_destroy(s); return NULL; }
    for (int i = 0; i < npin; i++) {
        const size_t need = (size_t)(align_up(s->blk[i].nbytes, ENG_IO_ALIGN)
                                     + ENG_IO_ALIGN + s->extra_bytes);
        if (alloc_direct((void **)&s->pin[i], need, huge) != 0) {
            fprintf(stderr, "%s streamer: cannot allocate %.2f GB for pinned block %d\n",
                    s->name, (double)need / 1e9, i);
            eng_streamer_destroy(s);
            return NULL;
        }
        s->pinned_bytes += (int64_t)need;
    }

    if (alloc_direct((void **)&s->arena, (size_t)ring * (size_t)ring_slot, huge) != 0) {
        fprintf(stderr, "%s streamer: cannot allocate the %.2f GB ring\n",
                s->name, (double)ring * (double)ring_slot / 1e9);
        eng_streamer_destroy(s);
        return NULL;
    }
    s->block_of = (int *)malloc((size_t)ring * sizeof *s->block_of);
    s->slot_pad = (int32_t *)calloc((size_t)ring, sizeof *s->slot_pad);
    if (!s->block_of || !s->slot_pad) { eng_streamer_destroy(s); return NULL; }
    for (int i = 0; i < ring; i++) s->block_of[i] = -1;

    /* THE TWO-SLOT RULE. See streamer.h: with one ring slot the reader would overwrite
     * the block the caller is computing on, silently. This is a correctness gate, not a
     * performance one. */
    if (cfg->async && ring >= 2) {
        StreamIO *io = (StreamIO *)calloc(1, sizeof *io);
        if (!io) { eng_streamer_destroy(s); return NULL; }
        io->s = s;
        pthread_mutex_init(&io->mu, NULL);
        pthread_cond_init(&io->cv, NULL);
        s->io = io;
        if (pthread_create(&io->thread, NULL, io_main, io) != 0) {
            fprintf(stderr, "%s streamer: cannot start the asynchronous reader\n", s->name);
            pthread_cond_destroy(&io->cv);
            pthread_mutex_destroy(&io->mu);
            free(io);
            s->io = NULL;
        }
    }

    if (!cfg->quiet) {
        printf("%s streamer: %d/%d blocks PINNED (%.2f GB), ring %d x %.2f GB\n",
               s->name, npin, s->nblocks,
               (double)(spent - (int64_t)ring * ring_slot) / 1e9,
               ring, (double)ring_slot / 1e9);
        printf("            deterministic hit rate %.1f%% "
               "(a cyclic scan defeats LRU, so a pinned prefix is used instead)\n",
               100.0 * npin / s->nblocks);
        if (!s->io)
            printf("            reads are NOT overlapped with compute: the ring has %d "
                   "slot%s and overlap needs 2 (a second slot costs %.2f GB). "
                   "Correctness is unaffected.\n",
                   ring, ring == 1 ? "" : "s", (double)ring_slot / 1e9);
    }
    return s;
}

void eng_streamer_destroy(EngStreamer *s)
{
    if (!s) return;
    if (s->io) {
        pthread_mutex_lock(&s->io->mu);
        s->io->stop = 1;
        pthread_cond_broadcast(&s->io->cv);
        pthread_mutex_unlock(&s->io->mu);
        pthread_join(s->io->thread, NULL);
        pthread_cond_destroy(&s->io->cv);
        pthread_mutex_destroy(&s->io->mu);
        free(s->io);
    }
    if (s->pin) {
        for (int i = 0; i < s->npin; i++) free(s->pin[i]);
        free(s->pin);
    }
    free(s->pin_pad);
    free(s->slot_pad);
    free(s->arena);
    free(s->block_of);
    free(s->slot_of);
    free(s->blk);
    free(s->name);
    free(s);
}

/* --------------------------------------------------------------- diagnostics -- */

void eng_streamer_stats(const EngStreamer *s, EngStreamerStats *out)
{
    if (!s || !out) return;
    out->hits         = s->hits;
    out->misses       = s->misses;
    out->bytes_read   = s->bytes_read;
    out->load_seconds = s->load_seconds;
    out->npin         = s->npin;
    out->nslot        = s->nslot;
    out->nblocks      = s->nblocks;
    out->slot_bytes   = s->slot_bytes;
    out->pinned_bytes = s->pinned_bytes;
    out->async        = s->io ? 1 : 0;
}

int eng_streamer_is_async(const EngStreamer *s) { return s && s->io ? 1 : 0; }

double eng_streamer_predicted_hit_rate(const EngStreamer *s)
{
    if (!s || s->nblocks <= 0) return 0.0;
    return (double)s->npin / (double)s->nblocks;
}

void eng_streamer_report(const EngStreamer *s, const char *label)
{
    if (!s) return;
    const uint64_t req = s->hits + s->misses;
    printf("%s%s%s streamer: %d pinned + %d ring slots, %s\n",
           label ? label : "", label ? " " : "", s->name,
           s->npin, s->nslot, s->io ? "overlapped reads" : "synchronous reads");
    printf("  requests %llu  hits %llu (%.1f%%, predicted %.1f%%)  misses %llu\n",
           (unsigned long long)req, (unsigned long long)s->hits,
           req ? 100.0 * (double)s->hits / (double)req : 0.0,
           100.0 * eng_streamer_predicted_hit_rate(s),
           (unsigned long long)s->misses);
    printf("  read %.2f GB in %.2f s", (double)s->bytes_read / 1e9, s->load_seconds);
    if (s->load_seconds > 0.0)
        printf(" (%.0f MB/s device rate)", (double)s->bytes_read / 1e6 / s->load_seconds);
    printf("\n");
}
