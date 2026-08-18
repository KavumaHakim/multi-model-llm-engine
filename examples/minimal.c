/* SPDX-License-Identifier: Apache-2.0 */
/*
 * minimal.c - embedding libengine, the way another project would.
 *
 * Built by `make example` with ONE include directory and the archive:
 *
 *     cc -Iinclude examples/minimal.c -o example bin/libengine.a -lm -fopenmp -pthread
 *
 * It knows nothing about src/. That is the point: `make test` builds this, so if an
 * internal header leaked into the public surface or an object is missing from the
 * archive, it fails here rather than in someone else's project.
 *
 * With no arguments it exercises everything that needs no model file. Given a path it
 * inspects and plans -- still without reading a weight -- and generates only if asked,
 * because on a streaming configuration that costs real time.
 *
 *     ./bin/example_minimal
 *     ./bin/example_minimal model.gguf
 *     ./bin/example_minimal model.gguf "What is the capital of France?"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engine.h"

int main(int argc, char **argv)
{
    printf("libengine %s\n\n", ENGINE_VERSION_STRING);

    /* ---- what this machine is ---- */
    EngHwInfo hw;
    eng_hwinfo_detect(&hw);
    eng_hwinfo_report(&hw, NULL);
    eng_kernel_report("\n");

    printf("\nbackends:");
    for (int i = 0; i < eng_model_count(); i++)
        printf(" %s", eng_model_at(i)->name);
    printf("\nquant formats:");
    for (int i = 0; i < eng_quant_count(); i++)
        printf(" %s", eng_quant_at(i)->name);
    printf("\n");

    if (argc < 2) {
        printf("\npass a model path to inspect one.\n");
        return 0;
    }
    const char *path = argv[1];

    /* ---- which backend claims it ---- */
    int score = 0;
    const EngModelBackend *b = eng_model_probe(path, &score);
    if (!b) {
        fprintf(stderr, "no backend recognises %s\n", path);
        return 1;
    }
    printf("\nbackend: %s (confidence %d)\n", b->name, score);

    /* ---- what it costs, without loading anything ---- */
    EngModelFacts facts;
    if (b->inspect(path, &facts) != 0) return 1;
    char h[32];
    eng_mem_human(facts.total_weight_bytes, h, sizeof h);
    printf("  %s: %d layers, %s of weights, %.3f bytes/weight, context %d\n",
           facts.arch, facts.n_layers, h, facts.bytes_per_weight, facts.context_max);

    /* ---- the plan this host would choose ---- */
    EngPlanRequest req;
    eng_plan_request_init(&req);
    req.context = 256;

    EngPlan plan;
    if (eng_plan(&plan, &hw, &facts, &req) != 0) {
        fprintf(stderr, "cannot plan: %s\n", plan.problem);
        return 1;
    }
    eng_plan_report(&plan, &facts, "\n");

    if (argc < 3) {
        printf("\npass a prompt as a second argument to generate.\n");
        return 0;
    }

    /* ---- load, and generate ---- */
    EngModelCaps caps;
    b->caps(&caps);
    if (!(caps.flags & ENG_MCAP_EXECUTE)) {
        fprintf(stderr, "%s cannot execute yet\n", b->name);
        return 1;
    }

    EngLoadReq lr;
    memset(&lr, 0, sizeof lr);
    lr.path = path;
    lr.plan = &plan;

    EngModel *m = b->load(&lr);
    if (!m) return 1;

    /* The tokenizer and the chat template come from the same container. */
    Gguf g;
    if (gguf_open(&g, path) != 0) { b->destroy(m); return 1; }
    EngBpe *tok = eng_bpe_from_gguf(&g);
    EngChat chat;
    eng_chat_detect(&chat, &g, tok);
    printf("\nchat template: %s\n", eng_chat_family_name(chat.family));

    int ids[512];
    int n = -1;
    if (chat.family != ENG_CHAT_NONE)
        n = eng_chat_build(&chat, tok, NULL, argv[2], 0, ids, 512);
    if (n < 0)   /* no template, or it did not fit: fall back to a continuation */
        n = eng_bpe_encode(tok, argv[2], (int)strlen(argv[2]), ids, 512, 0);
    if (n <= 0) { fprintf(stderr, "tokenizer produced nothing\n"); return 1; }

    EngSeqState *st = b->state_create(m, plan.context);
    printf("\n%s", argv[2]);
    fflush(stdout);

    /* Prompt: no logits except on the last position -- see ENG_DEC_LOGITS. */
    for (int i = 0; i < n; i++) {
        const unsigned f = (i == n - 1) ? ENG_DEC_LOGITS : 0u;
        if (b->decode(m, st, ids[i], i, f) != 0) return 1;
    }

    for (int step = 0; step < 24; step++) {
        int nv = 0;
        const float *lg = b->logits(m, &nv);
        int best = 0;
        for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v;
        if (best == eng_bpe_eos(tok)) break;

        /* Control tokens are structure, not text: printing them would leak the turn
         * markers into the answer. */
        if (!eng_bpe_is_special(tok, best)) {
            char piece[256];
            const int len = eng_bpe_decode_one(tok, best, piece, sizeof piece);
            if (len > 0) { fwrite(piece, 1, (size_t)len, stdout); fflush(stdout); }
        }
        if (b->decode(m, st, best, n + step, ENG_DEC_LOGITS) != 0) break;
    }
    printf("\n");

    b->state_destroy(st);
    b->destroy(m);
    eng_bpe_free(tok);
    gguf_close(&g);
    return 0;
}
