/* SPDX-License-Identifier: Apache-2.0 */
/* memory.c - see memory.h. */
#include "memory.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void eng_mem_human(int64_t b, char *buf, size_t cap)
{
    if (!buf || !cap) return;
    const double v = (double)b;
    if (b >= 1000LL * 1000 * 1000) snprintf(buf, cap, "%.2f GB", v / 1e9);
    else if (b >= 1000LL * 1000)   snprintf(buf, cap, "%.1f MB", v / 1e6);
    else if (b >= 1000)            snprintf(buf, cap, "%.1f KB", v / 1e3);
    else                           snprintf(buf, cap, "%lld B", (long long)b);
}

int64_t eng_mem_parse_size(const char *s, int *is_auto)
{
    if (is_auto) *is_auto = 0;
    if (!s || !*s) return -1;

    if (!strcasecmp(s, "auto")) {
        if (is_auto) *is_auto = 1;
        return 0;
    }

    char *end = NULL;
    const double v = strtod(s, &end);
    if (end == s || v < 0) return -1;

    while (*end && isspace((unsigned char)*end)) end++;

    int64_t mult = 1;
    if (*end) {
        switch (tolower((unsigned char)*end)) {
            case 'k': mult = 1024LL; break;
            case 'm': mult = 1024LL * 1024; break;
            case 'g': mult = 1024LL * 1024 * 1024; break;
            case 't': mult = 1024LL * 1024 * 1024 * 1024; break;
            case 'b': mult = 1; break;
            default:  return -1;      /* refuse rather than guess */
        }
        end++;
        /* Allow a trailing 'B' after a unit ("5GB"), and nothing else. */
        if (*end && tolower((unsigned char)*end) == 'b') end++;
        while (*end && isspace((unsigned char)*end)) end++;
        if (*end) return -1;
    }
    return (int64_t)(v * (double)mult);
}

int64_t eng_mem_auto_budget(int64_t ram_available, int64_t ram_total)
{
    /* Prefer MemAvailable: it is what the kernel believes it can hand out, whereas
     * total ignores everything else already running. Fall back to a fraction of total
     * when availability is unknown rather than assuming the machine is idle. */
    int64_t base = ram_available > 0 ? ram_available : (ram_total * 3) / 4;
    if (base <= 0) return 0;

    /* Leave the larger of 1 GB or 20% to the rest of the system. The engine is not the
     * only process, and MemAvailable is a measurement at one instant that another
     * process can invalidate a second later. */
    const int64_t keep_prop = base / 5;
    const int64_t keep = keep_prop > (1LL << 30) ? keep_prop : (1LL << 30);
    const int64_t budget = base - keep;
    return budget > 0 ? budget : 0;
}

int eng_mem_max_context(int64_t avail, int64_t kv_per_pos)
{
    if (kv_per_pos <= 0) return 0;
    if (avail <= 0) return 0;
    const int64_t n = avail / kv_per_pos;
    return n > (int64_t)1 << 30 ? (int)((int64_t)1 << 30) : (int)n;
}

int eng_mem_plan(EngMemPlan *plan, int64_t budget, const EngMemNeeds *needs,
                 int *context_out)
{
    if (!plan || !needs) return -1;
    memset(plan, 0, sizeof *plan);
    plan->budget = budget;

    char a[32], b[32];

    if (budget <= 0) {
        snprintf(plan->problem, sizeof plan->problem,
                 "no memory budget given (pass --memory, or --memory auto)");
        return -1;
    }

    /* See memory.h: floor covers fixed overheads, proportion covers estimate error. */
    const int64_t res_prop = budget * 12 / 100;
    plan->reserve = res_prop > (768LL << 20) ? res_prop : (768LL << 20);
    if (plan->reserve >= budget) {
        eng_mem_human(budget, a, sizeof a);
        eng_mem_human(plan->reserve, b, sizeof b);
        snprintf(plan->problem, sizeof plan->problem,
                 "budget %s is smaller than the safety reserve %s; "
                 "raise --memory above %s", a, b, b);
        return -1;
    }

    int64_t left = budget - plan->reserve;

    /* Non-negotiable, in order. A model that cannot hold its embeddings and LM head
     * cannot run at any budget, so say so precisely rather than reporting a small
     * weight budget and failing later. */
    plan->resident = needs->resident;
    if (plan->resident > left) {
        eng_mem_human(plan->resident, a, sizeof a);
        eng_mem_human(budget, b, sizeof b);
        snprintf(plan->problem, sizeof plan->problem,
                 "weights that cannot stream need %s but the budget is %s; "
                 "this model needs at least %.1f GB", a, b,
                 (double)(plan->resident + plan->reserve + needs->activations +
                          needs->scratch + needs->min_weights) / 1e9);
        return -1;
    }
    left -= plan->resident;

    plan->activations = needs->activations;
    plan->scratch     = needs->scratch;
    if (plan->activations + plan->scratch > left) {
        eng_mem_human(plan->activations + plan->scratch, a, sizeof a);
        eng_mem_human(left, b, sizeof b);
        snprintf(plan->problem, sizeof plan->problem,
                 "activations and scratch need %s but only %s remains after the "
                 "reserve and resident weights", a, b);
        return -1;
    }
    left -= plan->activations + plan->scratch;

    /* The KV cache is the one component the user trades directly against context, so it
     * is capped rather than allowed to consume the weight budget. Leave at least
     * min_weights for the weights; without that the model has nothing to compute with. */
    int ctx = needs->context;
    if (needs->kv_per_pos > 0 && ctx > 0) {
        const int64_t for_kv = left - needs->min_weights;
        const int64_t want   = (int64_t)ctx * needs->kv_per_pos;
        if (for_kv <= 0) {
            eng_mem_human(needs->min_weights, a, sizeof a);
            snprintf(plan->problem, sizeof plan->problem,
                     "nothing left for the KV cache: the weight budget needs at least %s", a);
            return -1;
        }
        if (want > for_kv) {
            /* Reduce the context rather than refuse the run: a shorter run is almost
             * always more useful than none, and the caller is told what it got. */
            const int fits = eng_mem_max_context(for_kv, needs->kv_per_pos);
            ctx = fits;
        }
        if (ctx <= 0) {
            eng_mem_human(needs->kv_per_pos, a, sizeof a);
            eng_mem_human(for_kv, b, sizeof b);
            snprintf(plan->problem, sizeof plan->problem,
                     "not one KV position fits: each costs %s and only %s is free", a, b);
            return -1;
        }
        plan->kv_cache = (int64_t)ctx * needs->kv_per_pos;
        left -= plan->kv_cache;
    }
    if (context_out) *context_out = ctx;

    plan->weights = left;
    if (plan->weights < needs->min_weights) {
        eng_mem_human(plan->weights, a, sizeof a);
        eng_mem_human(needs->min_weights, b, sizeof b);
        snprintf(plan->problem, sizeof plan->problem,
                 "only %s left for weights but at least %s is needed "
                 "(one layer must fit); raise --memory or lower --context", a, b);
        return -1;
    }

    plan->ok = 1;
    return 0;
}

void eng_mem_report(const EngMemPlan *p, const char *label)
{
    if (!p) return;
    char t[32];

    eng_mem_human(p->budget, t, sizeof t);
    printf("%s%smemory plan: %s budget\n", label ? label : "", label ? " " : "", t);

    if (!p->ok) {
        printf("  NOT VIABLE: %s\n", p->problem);
        return;
    }

    struct { const char *name; int64_t v; } rows[] = {
        { "reserve (OS + headroom)", p->reserve },
        { "resident weights",        p->resident },
        { "activations",             p->activations },
        { "scratch",                 p->scratch },
        { "KV cache",                p->kv_cache },
        { "weights (stream + cache)",p->weights }
    };
    int64_t sum = 0;
    for (size_t i = 0; i < sizeof rows / sizeof *rows; i++) {
        if (!rows[i].v) continue;
        eng_mem_human(rows[i].v, t, sizeof t);
        printf("  %-26s %10s  %5.1f%%\n", rows[i].name, t,
               p->budget ? 100.0 * (double)rows[i].v / (double)p->budget : 0.0);
        sum += rows[i].v;
    }
    eng_mem_human(sum, t, sizeof t);
    printf("  %-26s %10s\n", "total planned", t);
}
