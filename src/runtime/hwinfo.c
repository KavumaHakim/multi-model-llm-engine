/* SPDX-License-Identifier: Apache-2.0 */
/* hwinfo.c - see hwinfo.h. */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include "hwinfo.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__linux__)
#  include <sys/statvfs.h>
#  include <sys/stat.h>
#  include <unistd.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <sys/statvfs.h>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#  include <cpuid.h>
#endif

/* --------------------------------------------------------------------- ISA -- */

static uint32_t detect_isa(void)
{
    uint32_t isa = 0;
#if defined(__x86_64__) || defined(__i386__)
    /* __builtin_cpu_supports issues the CPUID itself and knows the XGETBV dance an
     * AVX check needs (the OS must have enabled the register state, not merely the
     * silicon supporting it). Doing that by hand is how a detector reports AVX2 on a
     * kernel that will SIGILL on the first instruction. */
    __builtin_cpu_init();
    if (__builtin_cpu_supports("sse2"))     isa |= ENG_ISA_SSE2;
    if (__builtin_cpu_supports("avx"))      isa |= ENG_ISA_AVX;
    if (__builtin_cpu_supports("avx2"))     isa |= ENG_ISA_AVX2;
    if (__builtin_cpu_supports("fma"))      isa |= ENG_ISA_FMA;
    if (__builtin_cpu_supports("avx512f"))  isa |= ENG_ISA_AVX512F;
    if (__builtin_cpu_supports("avx512bw")) isa |= ENG_ISA_AVX512BW;
#elif defined(__aarch64__)
    isa |= ENG_ISA_NEON;      /* mandatory on aarch64 */
#endif
    return isa;
}

/* What this BINARY contains, which is a different question from what the CPU can run.
 * A kernel can be missing because the chip lacks the ISA or because the build omitted
 * it, and those have different fixes. */
static uint32_t built_isa(void)
{
    uint32_t isa = 0;
#if defined(__SSE2__)
    isa |= ENG_ISA_SSE2;
#endif
#if defined(__AVX__)
    isa |= ENG_ISA_AVX;
#endif
#if defined(__AVX2__)
    isa |= ENG_ISA_AVX2;
#endif
#if defined(__FMA__)
    isa |= ENG_ISA_FMA;
#endif
#if defined(__AVX512F__)
    isa |= ENG_ISA_AVX512F;
#endif
#if defined(__AVX512BW__)
    isa |= ENG_ISA_AVX512BW;
#endif
#if defined(__ARM_NEON)
    isa |= ENG_ISA_NEON;
#endif
    return isa;
}

static void cpu_brand(char *out, size_t cap)
{
    out[0] = '\0';
#if defined(__x86_64__) || defined(__i386__)
    unsigned int r[4];
    if (!__get_cpuid(0x80000000u, &r[0], &r[1], &r[2], &r[3]) || r[0] < 0x80000004u)
        return;
    char buf[49];
    for (int i = 0; i < 3; i++) {
        if (!__get_cpuid(0x80000002u + (unsigned)i, &r[0], &r[1], &r[2], &r[3])) return;
        memcpy(buf + i * 16, r, 16);
    }
    buf[48] = '\0';
    /* The brand string is space-padded on the left on many parts. */
    const char *p = buf;
    while (*p == ' ') p++;
    snprintf(out, cap, "%s", p);
#else
    (void)cap;
#endif
}

/* -------------------------------------------------------------------- cores -- */

#if defined(__linux__)
/* Physical cores are counted by unique (physical_id, core_id) pairs in /proc/cpuinfo.
 * sysconf(_SC_NPROCESSORS_ONLN) reports LOGICAL processors and would count each
 * hyperthread, which on the reference machine is 4 where only 2 can do independent
 * work -- exactly the confusion the thread sweep in docs/baseline-m0.md measured. */
static int count_physical_cores(void)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return 0;
    /* Bounded: 4096 distinct (package, core) pairs is far beyond anything this engine
     * targets, and an unbounded set here would be a parsing-driven allocation. */
    static int phys[4096], core[4096];
    int n = 0, cur_phys = 0, cur_core = -1;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        if (!strncmp(line, "physical id", 11)) {
            const char *c = strchr(line, ':');
            if (c) cur_phys = atoi(c + 1);
        } else if (!strncmp(line, "core id", 7)) {
            const char *c = strchr(line, ':');
            if (c) cur_core = atoi(c + 1);
        } else if (line[0] == '\n' && cur_core >= 0) {
            int seen = 0;
            for (int i = 0; i < n; i++)
                if (phys[i] == cur_phys && core[i] == cur_core) { seen = 1; break; }
            if (!seen && n < (int)(sizeof phys / sizeof *phys)) {
                phys[n] = cur_phys; core[n] = cur_core; n++;
            }
            cur_core = -1;
        }
    }
    fclose(f);
    return n;
}
#endif

/* ------------------------------------------------------------------- memory -- */

static void detect_memory(EngHwInfo *hw)
{
#if defined(__linux__)
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        long long kb = 0;
        if (sscanf(line, "MemTotal: %lld kB", &kb) == 1)
            hw->ram_total = (int64_t)kb * 1024;
        else if (sscanf(line, "MemAvailable: %lld kB", &kb) == 1)
            hw->ram_available = (int64_t)kb * 1024;
    }
    fclose(f);
#elif defined(__APPLE__)
    int64_t v = 0;
    size_t len = sizeof v;
    if (sysctlbyname("hw.memsize", &v, &len, NULL, 0) == 0) hw->ram_total = v;
    /* Darwin has no direct MemAvailable analogue. Free pages understate badly because
     * of the unified buffer cache, so report the total and let the planner apply its
     * reserve rather than inventing a number. */
    hw->ram_available = hw->ram_total;
#endif
}

/* ------------------------------------------------------------------ storage -- */

void eng_hwinfo_probe_path(EngHwInfo *hw, const char *path)
{
    if (!hw || !path) return;

#if defined(__linux__) || defined(__APPLE__)
    struct statvfs vfs;
    if (statvfs(path, &vfs) == 0)
        hw->storage_free = (int64_t)vfs.f_bavail * (int64_t)vfs.f_frsize;
#endif

#if defined(__linux__)
    /* Rotational or not decides whether random reads are affordable, which is the
     * difference between "stream freely" and "batch everything into sequential runs".
     * The device is found from the file's st_dev; if anything in the chain is missing
     * (a bind mount, an overlay, a 9p share under WSL) the answer stays UNKNOWN and the
     * planner is conservative rather than wrong. */
    struct stat st;
    if (stat(path, &st) != 0) return;
    const unsigned maj = (unsigned)((st.st_dev >> 8) & 0xfff);
    const unsigned min = (unsigned)(st.st_dev & 0xff);

    char p[256], buf[32];
    snprintf(p, sizeof p, "/sys/dev/block/%u:%u/queue/rotational", maj, min);
    FILE *f = fopen(p, "r");
    if (!f) {
        /* Partitions carry the flag on their parent. Try the whole-disk node. */
        snprintf(p, sizeof p, "/sys/dev/block/%u:0/queue/rotational", maj);
        f = fopen(p, "r");
    }
    if (f) {
        if (fgets(buf, sizeof buf, f))
            hw->storage = atoi(buf) ? ENG_STORE_ROTATIONAL : ENG_STORE_SSD;
        fclose(f);
    }
#endif
}

/* ------------------------------------------------------------------- detect -- */

void eng_hwinfo_detect(EngHwInfo *hw)
{
    if (!hw) return;
    memset(hw, 0, sizeof *hw);

    cpu_brand(hw->cpu_name, sizeof hw->cpu_name);
    if (!hw->cpu_name[0]) snprintf(hw->cpu_name, sizeof hw->cpu_name, "unknown CPU");

    hw->isa       = detect_isa();
    hw->isa_built = built_isa();

#if defined(__linux__)
    hw->cores_logical  = (int)sysconf(_SC_NPROCESSORS_ONLN);
    hw->cores_physical = count_physical_cores();
#elif defined(__APPLE__)
    int v = 0; size_t len = sizeof v;
    if (sysctlbyname("hw.logicalcpu", &v, &len, NULL, 0) == 0) hw->cores_logical = v;
    len = sizeof v;
    if (sysctlbyname("hw.physicalcpu", &v, &len, NULL, 0) == 0) hw->cores_physical = v;
#endif
    if (hw->cores_logical <= 0)  hw->cores_logical = 1;
    /* Fall back to logical rather than to zero: a planner dividing by physical cores
     * must never see 0, and assuming no SMT is the conservative error. */
    if (hw->cores_physical <= 0) hw->cores_physical = hw->cores_logical;

    detect_memory(hw);
}

/* --------------------------------------------------------------- reporting -- */

int eng_hwinfo_has(const EngHwInfo *hw, uint32_t bit)
{
    return hw && (hw->isa & bit) ? 1 : 0;
}

void eng_hwinfo_isa_string(uint32_t isa, char *buf, size_t cap)
{
    if (!buf || !cap) return;
    buf[0] = '\0';
    static const struct { uint32_t bit; const char *name; } T[] = {
        { ENG_ISA_SSE2, "sse2" }, { ENG_ISA_AVX, "avx" }, { ENG_ISA_AVX2, "avx2" },
        { ENG_ISA_FMA, "fma" }, { ENG_ISA_AVX512F, "avx512f" },
        { ENG_ISA_AVX512BW, "avx512bw" }, { ENG_ISA_NEON, "neon" }
    };
    size_t n = 0;
    for (size_t i = 0; i < sizeof T / sizeof *T && n + 1 < cap; i++) {
        if (!(isa & T[i].bit)) continue;
        const int w = snprintf(buf + n, cap - n, "%s%s", n ? " " : "", T[i].name);
        if (w < 0) break;
        n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "(none)");
}

void eng_hwinfo_report(const EngHwInfo *hw, const char *label)
{
    if (!hw) return;
    char have[96], built[96];
    eng_hwinfo_isa_string(hw->isa, have, sizeof have);
    eng_hwinfo_isa_string(hw->isa_built, built, sizeof built);

    printf("%s%shost\n", label ? label : "", label ? " " : "");
    printf("  cpu       : %s\n", hw->cpu_name);
    printf("  cores     : %d physical / %d logical\n",
           hw->cores_physical, hw->cores_logical);
    printf("  isa       : %s\n", have);
    /* Report the two separately: a kernel can be unavailable because the CPU lacks the
     * ISA or because the build omitted it, and the fixes differ. */
    if (hw->isa_built != hw->isa)
        printf("  built for : %s%s\n", built,
               (hw->isa & ~hw->isa_built) ? "  (binary does not use everything this CPU has)" : "");
    if (hw->ram_total)
        printf("  ram       : %.2f GB total, %.2f GB available\n",
               (double)hw->ram_total / 1e9, (double)hw->ram_available / 1e9);
    else
        printf("  ram       : unknown\n");
    if (hw->storage != ENG_STORE_UNKNOWN || hw->storage_free)
        printf("  storage   : %s, %.1f GB free\n",
               hw->storage == ENG_STORE_ROTATIONAL ? "rotational" :
               hw->storage == ENG_STORE_SSD ? "solid state" : "unknown class",
               (double)hw->storage_free / 1e9);
}
