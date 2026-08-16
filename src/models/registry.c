/* SPDX-License-Identifier: Apache-2.0 */
/* registry.c - backend registration and probing. See model.h. */
#include "model.h"

#include <stdio.h>
#include <string.h>

#define ENG_MAX_BACKENDS 16

/* Built-ins, defined in their own directories. Adding an architecture touches this
 * list and nothing else in the runtime -- that property is the whole point, and
 * tests/unit/test_model.c asserts it by registering a synthetic backend that shares no
 * code with either real one. */
extern const EngModelBackend eng_backend_kimi_k3;
extern const EngModelBackend eng_backend_qwen3;

static const EngModelBackend *REG[ENG_MAX_BACKENDS];
static int reg_n = 0;
static int reg_ready = 0;

void eng_model_register_builtins(void)
{
    if (reg_ready) return;
    reg_ready = 1;          /* set FIRST: register() calls back into find() */
    eng_model_register(&eng_backend_kimi_k3);
    eng_model_register(&eng_backend_qwen3);
}

int eng_model_register(const EngModelBackend *b)
{
    if (!b || !b->name) return -1;

    /* probe, inspect and caps are what make a backend usable at all. A backend missing
     * one of them would register successfully and fail somewhere later, which is worse
     * than refusing here. execute-path entry points are NOT required: a backend may be
     * registered while its forward pass is still being migrated, and declares that
     * through ENG_MCAP_EXECUTE. */
    if (!b->probe || !b->inspect || !b->caps) {
        fprintf(stderr, "models: backend '%s' is missing probe, inspect or caps\n",
                b->name);
        return -1;
    }

    /* A backend claiming ENG_MCAP_EXECUTE must actually be able to. Checking the claim
     * against the vtable here turns a null-pointer call during generation into a
     * startup error. */
    EngModelCaps c;
    memset(&c, 0, sizeof c);
    b->caps(&c);
    if ((c.flags & ENG_MCAP_EXECUTE) &&
        (!b->load || !b->destroy || !b->decode || !b->logits ||
         !b->state_create || !b->state_destroy || !b->state_bytes)) {
        fprintf(stderr, "models: backend '%s' claims ENG_MCAP_EXECUTE but does not "
                        "implement the full run path\n", b->name);
        return -1;
    }

    eng_model_register_builtins();
    if (reg_n >= ENG_MAX_BACKENDS) {
        fprintf(stderr, "models: too many backends (max %d)\n", ENG_MAX_BACKENDS);
        return -1;
    }
    for (int i = 0; i < reg_n; i++)
        if (!strcmp(REG[i]->name, b->name)) {
            fprintf(stderr, "models: '%s' is already registered\n", b->name);
            return -1;              /* refuse to shadow silently */
        }

    REG[reg_n++] = b;
    return 0;
}

const EngModelBackend *eng_model_find(const char *name)
{
    if (!name) return NULL;
    eng_model_register_builtins();
    for (int i = 0; i < reg_n; i++)
        if (!strcmp(REG[i]->name, name)) return REG[i];
    return NULL;
}

const EngModelBackend *eng_model_probe(const char *path, int *score_out)
{
    if (score_out) *score_out = 0;
    if (!path) return NULL;
    eng_model_register_builtins();

    const EngModelBackend *best = NULL;
    int best_score = 0;

    /* Highest score wins, and ties keep the FIRST registered rather than the last, so
     * the result does not depend on registration order changing under someone. A tie is
     * still worth reporting: two backends claiming one container equally is a design
     * problem, not something to resolve silently. */
    int ties = 0;
    for (int i = 0; i < reg_n; i++) {
        const int s = REG[i]->probe(path);
        if (s <= 0) continue;
        if (s > best_score) { best = REG[i]; best_score = s; ties = 0; }
        else if (s == best_score) ties++;
    }
    if (ties)
        fprintf(stderr, "models: %d backends claim %s with equal confidence %d; "
                        "using '%s'\n", ties + 1, path, best_score, best->name);

    if (score_out) *score_out = best_score;
    return best;
}

int eng_model_count(void)
{
    eng_model_register_builtins();
    return reg_n;
}

const EngModelBackend *eng_model_at(int i)
{
    eng_model_register_builtins();
    return (i >= 0 && i < reg_n) ? REG[i] : NULL;
}

void eng_model_caps_string(const EngModelCaps *c, char *buf, size_t cap)
{
    if (!buf || !cap) return;
    buf[0] = '\0';
    if (!c) return;

    static const struct { uint32_t bit; const char *name; } T[] = {
        { ENG_MCAP_EXECUTE,     "execute"     },
        { ENG_MCAP_MOE,         "moe"         },
        { ENG_MCAP_POSITIONAL,  "positional"  },
        { ENG_MCAP_RECURRENT,   "recurrent"   },
        { ENG_MCAP_STREAMABLE,  "streamable"  },
        { ENG_MCAP_INCREMENTAL, "incremental" }
    };
    size_t n = 0;
    for (size_t i = 0; i < sizeof T / sizeof *T && n + 1 < cap; i++) {
        if (!(c->flags & T[i].bit)) continue;
        const int w = snprintf(buf + n, cap - n, "%s%s", n ? " " : "", T[i].name);
        if (w < 0) break;
        n += (size_t)w;
    }
    if (n + 1 < cap) {
        const int w = snprintf(buf + n, cap - n, "%s%s", n ? " " : "",
                               c->num_policy == ENG_NUM_EXACT ? "exact" : "fast");
        if (w > 0) n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "(none)");
}
