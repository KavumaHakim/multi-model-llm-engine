/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_streaming_equiv.c - the same answer at every memory budget.
 *
 * WHY THIS EXISTS. The whole design rests on a property nothing was checking: that a
 * weight produces the same result whether it was resident, served from a cache slot, or
 * read from disk moments before use. Every other test in this suite fixes one
 * configuration and checks the arithmetic within it. This one varies the configuration
 * and checks that the arithmetic did not move.
 *
 * That is not a theoretical worry. The failure modes it covers have all occurred in this
 * codebase or its parent:
 *
 *   - a ring slot recycled while still in use, so a layer computes on the NEXT layer's
 *     bytes (upstream measured a prompt silently changing its output);
 *   - a payload pointer that ignores the alignment padding a direct read introduces,
 *     which shifts every weight in the layer by up to 4095 bytes;
 *   - a cache serving a hit for an object whose read failed;
 *   - a pinned prefix and a streamed tail taking different code paths to the same
 *     tensor.
 *
 * None of those change the SHAPE of the output. They change its value, plausibly, and
 * only a cross-budget comparison catches them.
 *
 * WHAT IS COMPARED. Logits, on RAW BITS. Not the greedy token -- an argmax is stable
 * under small perturbations, so comparing tokens would pass on a run whose distribution
 * had visibly moved. Bit equality is achievable here because the arithmetic genuinely
 * does not depend on residency: the same kernel sees the same bytes in the same order
 * regardless of how they arrived.
 *
 * THE BUDGETS are derived from the host and chosen to force structurally different
 * plans, not merely different numbers: one where most layers pin, one where fewer do,
 * and one at the floor. Measured on the development machine those come out at 13, 7 and
 * ZERO pinned layers -- the last streams every weight of the model on every token. If
 * those agree bit for bit, residency is not affecting the result.
 *
 * WHAT THIS DOES NOT REACH, stated because the header once claimed it did: the
 * single-ring-slot path, where the asynchronous reader is disabled because it would
 * otherwise read over the block being computed on. That configuration is not reachable
 * through the planner on a host like this one. The memory plan's safety reserve has a
 * 768 MB floor, so the smallest viable budget is around 1.1 GB, and by then the weight
 * budget still affords two ring slots. The two-slot rule is covered instead by
 * tests/unit/test_streamer.c, which constructs the one-slot case directly and asserts
 * the reader stays off.
 *
 * SKIPS, LOUDLY, without a model. This needs the real GGUF; GGUF_MODEL names it. A
 * silent skip would let the gate rot.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "planner.h"
#include "hwinfo.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-46s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

/* A short fixed prompt. Content does not matter; only that every arm sees the same. */
static const int TOKENS[] = { 9707, 11, 1879, 0 };
#define NTOK ((int)(sizeof TOKENS / sizeof TOKENS[0]))

typedef struct {
    int64_t budget;
    const char *label;
    float  *logits;
    int     n_vocab;
    int     ok;
    int     pinned, nslot, async;
} Arm;

/* Run the model once at one budget and capture the final logits. */
static int run_arm(const EngModelBackend *b, const char *path, Arm *a)
{
    EngHwInfo hw;
    eng_hwinfo_detect(&hw);
    eng_hwinfo_probe_path(&hw, path);

    EngModelFacts facts;
    if (b->inspect(path, &facts) != 0) return -1;

    EngPlanRequest req;
    eng_plan_request_init(&req);
    req.memory_budget = a->budget;
    req.context = 64;

    EngPlan plan;
    if (eng_plan(&plan, &hw, &facts, &req) != 0) {
        printf("        plan refused at %s: %s\n", a->label, plan.problem);
        return -1;
    }

    EngLoadReq lr;
    memset(&lr, 0, sizeof lr);
    lr.path = path;
    lr.plan = &plan;
    lr.verbose = 0;

    EngModel *m = b->load(&lr);
    if (!m) return -1;

    EngSeqState *st = b->state_create(m, 64);
    if (!st) { b->destroy(m); return -1; }

    int rc = 0;
    for (int t = 0; t < NTOK; t++) {
        /* Logits on the last position only, which is the path a real run takes. */
        const unsigned flags = (t == NTOK - 1) ? ENG_DEC_LOGITS : 0u;
        rc = b->decode(m, st, TOKENS[t], t, flags);
        if (rc != 0) break;
    }

    if (rc == 0) {
        int nv = 0;
        const float *lg = b->logits(m, &nv);
        if (lg && nv > 0) {
            a->logits = (float *)malloc((size_t)nv * sizeof *a->logits);
            memcpy(a->logits, lg, (size_t)nv * sizeof *a->logits);
            a->n_vocab = nv;
            a->ok = 1;
        } else {
            rc = -1;
        }
    }

    EngRunStats rs;
    if (b->stats) {
        memset(&rs, 0, sizeof rs);
        b->stats(m, &rs);
        /* Recorded so the report can show the arms really were configured differently.
         * Three runs that agree because they all chose the same plan prove nothing. */
        a->pinned = rs.pinned_blocks;
        a->nslot  = rs.ring_slots;
        a->async  = rs.async_reader;
    }

    b->state_destroy(st);
    b->destroy(m);
    return rc;
}

int main(int argc, char **argv)
{
    const char *path = getenv("GGUF_MODEL");
    if (argc > 1) path = argv[1];

    printf("streaming equivalence: the same logits at every memory budget\n\n");

    if (!path) {
        printf("  SKIP  set GGUF_MODEL (or pass a path) to run this gate\n");
        printf("        it needs the real model: the property under test is about how\n");
        printf("        weights ARRIVE, which a fixture cannot exercise\n");
        return 0;
    }

    int score = 0;
    const EngModelBackend *b = eng_model_probe(path, &score);
    if (!b) { printf("  no backend recognises %s\n", path); return 1; }

    EngModelCaps caps;
    b->caps(&caps);
    if (!(caps.flags & ENG_MCAP_EXECUTE)) {
        printf("  SKIP  %s cannot execute yet\n", b->name);
        return 0;
    }

    /* THE BUDGETS ARE DERIVED FROM THE HOST, not hardcoded.
     *
     * The first version of this test asked for 6 GB, 2400 MB and 1100 MB. On the
     * development machine -- 3.8 GB total, 3.37 GB available -- the 6 GB arm cannot be
     * satisfied at all: it would exhaust RAM and thrash a 1 GB swap, and an arm that
     * dies proves nothing about equivalence. Hardcoding budgets makes a test that
     * passes only on the machine it was written on, which is worse than useless in a
     * repository other people clone.
     *
     * So the top arm is whatever this host can actually afford, and the others are
     * spaced beneath it. What the test REQUIRES is not any particular number but that
     * the arms end up with different plans, which is asserted below -- if a host is too
     * small to produce a spread, that shows up as a failed assertion rather than a
     * false pass. */
    EngHwInfo hw;
    eng_hwinfo_detect(&hw);
    int64_t avail = hw.ram_available > 0 ? hw.ram_available : (2048LL << 20);

    int64_t hi = (int64_t)((double)avail * 0.75);
    if (hi > (6000LL << 20)) hi = 6000LL << 20;   /* no point beyond the model size */
    int64_t lo = 1100LL << 20;                    /* about the floor a 5 GB model allows */
    if (hi < lo + (256LL << 20)) {
        printf("  SKIP  this host has %.2f GB available; not enough spread between\n"
               "        budgets to produce structurally different plans\n",
               (double)avail / 1e9);
        return 0;
    }
    int64_t mid = lo + (int64_t)((double)(hi - lo) * 0.55);

    char lbl[3][64];
    snprintf(lbl[0], sizeof lbl[0], "%lldM (most pinned)", (long long)(hi  >> 20));
    snprintf(lbl[1], sizeof lbl[1], "%lldM (fewer pinned)", (long long)(mid >> 20));
    snprintf(lbl[2], sizeof lbl[2], "%lldM (floor)",        (long long)(lo  >> 20));

    Arm arms[] = {
        { hi,  lbl[0], NULL, 0, 0, 0, 0, 0 },
        { mid, lbl[1], NULL, 0, 0, 0, 0, 0 },
        { lo,  lbl[2], NULL, 0, 0, 0, 0, 0 }
    };
    const int narm = (int)(sizeof arms / sizeof arms[0]);

    printf("  host: %.2f GB available; budgets %lldM / %lldM / %lldM\n\n",
           (double)avail / 1e9,
           (long long)(hi >> 20), (long long)(mid >> 20), (long long)(lo >> 20));

    for (int i = 0; i < narm; i++) {
        printf("  running %s ...\n", arms[i].label);
        fflush(stdout);
        if (run_arm(b, path, &arms[i]) != 0)
            printf("        arm failed\n");
    }

    /* Every arm has to have run, or "they all agree" is vacuous. */
    int ran = 0;
    for (int i = 0; i < narm; i++) ran += arms[i].ok;
    char d[160];
    snprintf(d, sizeof d, "%d of %d arms produced logits", ran, narm);
    ok(ran == narm, "every budget ran", d);

    /* And they must have chosen DIFFERENT plans, or the comparison is trivial. */
    if (ran == narm) {
        snprintf(d, sizeof d, "pinned %d / %d / %d, ring %d / %d / %d",
                 arms[0].pinned, arms[1].pinned, arms[2].pinned,
                 arms[0].nslot,  arms[1].nslot,  arms[2].nslot);
        ok(arms[0].pinned != arms[2].pinned,
           "the budgets really produced different plans", d);
    }

    /* THE GATE. Raw bits, against arm 0. */
    for (int i = 1; i < narm && ran == narm; i++) {
        const int nv = arms[0].n_vocab < arms[i].n_vocab
                     ? arms[0].n_vocab : arms[i].n_vocab;
        int diff = 0;
        double worst = 0.0;
        int at = -1;
        for (int v = 0; v < nv; v++) {
            if (memcmp(&arms[0].logits[v], &arms[i].logits[v], sizeof(float)) != 0) {
                diff++;
                const double e = fabs((double)arms[0].logits[v] - (double)arms[i].logits[v]);
                if (e > worst) { worst = e; at = v; }
            }
        }
        snprintf(d, sizeof d, "%s vs %s: %d of %d logits differ",
                 arms[i].label, arms[0].label, diff, nv);
        ok(diff == 0, "logits are bit-identical across budgets", d);
        if (diff) {
            printf("        worst |delta| %.6g at vocab index %d\n", worst, at);
            printf("        a weight is arriving differently depending on residency\n");
        }
    }

    for (int i = 0; i < narm; i++) free(arms[i].logits);

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("STREAMING EQUIVALENCE FAILED\n"); return 1; }
    printf("STREAMING EQUIVALENCE PASSED\n");
    return 0;
}
