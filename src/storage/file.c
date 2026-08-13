/* SPDX-License-Identifier: Apache-2.0 */
/*
 * file.c - the pread-backed storage backend.
 *
 * The aligned-read logic here is lifted from k3_st.c:k3_st_read_aligned() rather than
 * re-derived, because its edge cases were found against a real 1.56 TB checkpoint and
 * are not obvious:
 *
 *   - the final aligned window of a file can extend PAST EOF, which returns a short
 *     read rather than an error. That is success as long as the payload itself was
 *     covered, and treating it as failure makes the last tensor of every file
 *     unreadable;
 *   - the return value is payload bytes, not window bytes, so a caller never has to
 *     know the read was widened;
 *   - when no direct descriptor exists the function still has to work, setting
 *     *payload_off = 0, or every call site needs its own fallback.
 *
 * WHY pread AND NOT mmap BY DEFAULT
 *   Pages read into a buffer the engine owns never become file-backed mappings counted
 *   against the process, so peak RSS tracks what is genuinely resident. K3 makes this
 *   argument about a 1.56 TB checkpoint (k3_st.h:22-25) where mmap would render the RSS
 *   figure meaningless; it holds for a 4.7 GB GGUF on an 8 GB machine too, which is the
 *   configuration this engine has to plan memory for. mmap is offered as a separate
 *   backend for cases where it genuinely wins, not as the default.
 */
#define _GNU_SOURCE               /* O_DIRECT */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "storage.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Darwin spells O_DIRECT as fcntl(F_NOCACHE) after the fact; keep the same shim shape
 * k3_portable_io.h uses so behaviour matches across the two readers. */
#if defined(__APPLE__)
#  ifndef O_DIRECT
#    define O_DIRECT 0
#  endif
static int eng_set_direct(int fd) { return fd < 0 ? -1 : fcntl(fd, F_NOCACHE, 1); }
#else
static int eng_set_direct(int fd) { (void)fd; return 0; }
#endif

typedef struct {
    int     fd;          /* buffered */
    int     dfd;         /* O_DIRECT, or -1 */
    int64_t size;
    char   *path;
} FileCtx;

static double now_s(void)
{
    struct timespec ts;
#if defined(CLOCK_MONOTONIC)
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
    return (double)time(NULL);
}

/* Loop: a short pread is normal, not an error. */
static int64_t pread_all(int fd, void *dst, int64_t nbytes, int64_t off)
{
    int64_t got = 0;
    while (got < nbytes) {
        const ssize_t r = pread(fd, (char *)dst + got, (size_t)(nbytes - got),
                                (off_t)(off + got));
        if (r <= 0) break;
        got += r;
    }
    return got;
}

static int64_t file_read(EngStorage *s, int64_t off, int64_t nbytes, void *dst)
{
    FileCtx *c = (FileCtx *)s->ctx;
    if (!c || off < 0 || nbytes < 0 || !dst) return 0;

    const double t0 = now_s();
    const int64_t got = pread_all(c->fd, dst, nbytes, off);
    s->read_seconds += now_s() - t0;
    s->reads++;
    s->bytes_read += (uint64_t)got;
    return got;
}

static int64_t file_read_aligned(EngStorage *s, int64_t off, int64_t nbytes,
                                 void *buf, int64_t bufcap, int64_t *payload_off)
{
    FileCtx *c = (FileCtx *)s->ctx;
    if (!c || off < 0 || nbytes < 0 || !buf) return 0;

    if (c->dfd < 0) {                       /* no direct path: buffered, unwidened */
        if (payload_off) *payload_off = 0;
        if (bufcap < nbytes) return 0;
        return file_read(s, off, nbytes, buf);
    }

    /* Widen outward to the enclosing aligned window. */
    const int64_t lo  = off & ~(int64_t)(ENG_IO_ALIGN - 1);
    const int64_t hi  = (off + nbytes + ENG_IO_ALIGN - 1) & ~(int64_t)(ENG_IO_ALIGN - 1);
    const int64_t len = hi - lo;
    const int64_t pad = off - lo;
    if (len > bufcap) return 0;
    if (payload_off) *payload_off = pad;

    const double t0 = now_s();
    /* Not pread_all: the last window of a file legitimately runs past EOF and returns
     * short. That is success provided the payload was covered, which the return
     * expression below is what checks. */
    int64_t got = 0;
    while (got < len) {
        const ssize_t r = pread(c->dfd, (char *)buf + got, (size_t)(len - got),
                                (off_t)(lo + got));
        if (r <= 0) break;
        got += r;
    }
    s->read_seconds += now_s() - t0;
    s->reads++;
    s->direct_reads++;
    s->bytes_read += (uint64_t)got;

    return got >= pad + nbytes ? nbytes : (got > pad ? got - pad : 0);
}

static int64_t file_size(EngStorage *s)
{
    FileCtx *c = (FileCtx *)s->ctx;
    return c ? c->size : 0;
}

static void file_close(EngStorage *s)
{
    if (!s) return;
    FileCtx *c = (FileCtx *)s->ctx;
    if (c) {
        if (c->fd  >= 0) close(c->fd);
        if (c->dfd >= 0) close(c->dfd);
        free(c->path);
        free(c);
    }
    free(s);
}

EngStorage *eng_storage_open_file(const char *path, int want_direct)
{
    if (!path) return NULL;

    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "storage: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "storage: cannot stat %s: %s\n", path, strerror(errno));
        close(fd);
        return NULL;
    }

    FileCtx *c = (FileCtx *)calloc(1, sizeof *c);
    EngStorage *s = (EngStorage *)calloc(1, sizeof *s);
    if (!c || !s) { free(c); free(s); close(fd); return NULL; }

    c->fd   = fd;
    c->dfd  = -1;
    c->size = (int64_t)st.st_size;
    c->path = strdup(path);

    /* A second descriptor on the same file. Optional by design: a filesystem that
     * refuses O_DIRECT (tmpfs, some network mounts, WSL's 9p bridge) must still work,
     * just without bypassing the page cache. */
    if (want_direct) {
        c->dfd = open(path, O_RDONLY | O_DIRECT);
        if (c->dfd >= 0) eng_set_direct(c->dfd);
    }

    s->kind         = "file";
    s->path         = c->path;
    s->read         = file_read;
    s->read_aligned = file_read_aligned;
    s->map          = NULL;          /* the mmap backend is a separate one */
    s->size         = file_size;
    s->close        = file_close;
    s->ctx          = c;
    return s;
}

int eng_storage_is_direct(const EngStorage *s)
{
    if (!s || !s->ctx) return 0;
    return ((const FileCtx *)s->ctx)->dfd >= 0;
}

void *eng_storage_alloc_aligned(int64_t nbytes, int64_t *out_cap)
{
    if (nbytes < 0) return NULL;
    /* Room for the widening in both directions, rounded to a whole number of pages. */
    int64_t cap = nbytes + 2 * ENG_IO_ALIGN;
    cap = (cap + ENG_IO_ALIGN - 1) & ~(int64_t)(ENG_IO_ALIGN - 1);

    void *p = NULL;
#if defined(_POSIX_VERSION)
    if (posix_memalign(&p, ENG_IO_ALIGN, (size_t)cap) != 0) p = NULL;
#else
    p = malloc((size_t)cap);
#endif
    if (p && out_cap) *out_cap = cap;
    return p;
}

void eng_storage_free_aligned(void *p) { free(p); }

void eng_storage_report(const EngStorage *s, const char *label)
{
    if (!s) return;
    const double mb = (double)s->bytes_read / 1e6;
    const double sec = s->read_seconds;
    printf("%s%s%s: %llu reads (%llu direct), %.1f MB, %.3f s",
           label ? label : "", label ? " " : "", s->kind,
           (unsigned long long)s->reads, (unsigned long long)s->direct_reads, mb, sec);
    if (sec > 0.0) printf(", %.0f MB/s", mb / sec);
    printf("\n");
}
