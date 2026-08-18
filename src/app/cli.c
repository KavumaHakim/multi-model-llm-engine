/* SPDX-License-Identifier: Apache-2.0 */
/*
 * cli.c - the engine's command line.
 *
 *   engine inspect MODEL              what it is, and what it would need to run
 *   engine run MODEL [options]        generate
 *   engine benchmark MODEL            where the time goes
 *
 * WHY inspect EXISTS AS A SEPARATE COMMAND. It answers "will this run here, and if not
 * why not" WITHOUT loading a weight, so it works on a machine that could not host the
 * model at all. Everything it prints comes from the container's metadata and the host's
 * own properties, and the plan it shows is the same one `run` would use -- not a
 * description of it. A user who has to start a run to find out it will not fit has
 * been given a worse tool than necessary.
 *
 * THE PLANNER EXPLAINS ITSELF. Every automatic choice carries its reason, because a
 * tool that silently picks four threads and a 3 GB budget is one the user has to
 * reverse-engineer before they can trust it.
 */
#define _POSIX_C_SOURCE 200809L

#include "bpe.h"
#include "chat.h"
#include "gguf.h"
#include "hwinfo.h"
#include "kernel.h"
#include "memory.h"
#include "model.h"
#include "planner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static void usage(void)
{
    printf(
"engine - a CPU inference runtime\n"
"\n"
"  engine inspect MODEL\n"
"        Report the architecture, what it would cost to run, and the plan this\n"
"        host would choose. Loads no weights.\n"
"\n"
"  engine run MODEL [options]\n"
"        -p, --prompt TEXT       prompt (default: a short greeting)\n"
"        -n, --tokens N          tokens to generate (default 16)\n"
"        -s, --system TEXT       system message; implies --chat\n"
"        --chat / --raw          apply the model's chat template, or do not. Default\n"
"                                is to apply it when the model has one. WITHOUT it an\n"
"                                instruct model CONTINUES your prompt rather than\n"
"                                answering it.\n"
"        --think                 let the model reason first, where its template\n"
"                                supports it. Off by default: reasoning costs the same\n"
"                                per token as the answer.\n"
"        --memory SIZE           budget, e.g. 6G, 512M, or auto (default auto)\n"
"        --threads N             worker threads (default: chosen from the model)\n"
"        --context N             context length (default: the model's, capped to fit)\n"
"        --auto                  choose everything; the default when nothing is given\n"
"        --resident              refuse to run unless the weights fit in RAM\n"
"        --no-stream             synonym for --resident\n"
"        -v, --verbose           show the plan and per-token timing\n"
"\n"
"  engine benchmark MODEL [--tokens N]\n"
"        Decode N tokens and report where the time went.\n"
"\n"
"  engine list\n"
"        Registered model backends.\n");
}

/* ------------------------------------------------------------------ inspect -- */

static int cmd_inspect(const char *path)
{
    EngHwInfo hw;
    eng_hwinfo_detect(&hw);
    eng_hwinfo_probe_path(&hw, path);
    eng_hwinfo_report(&hw, NULL);
    printf("\n");

    int score = 0;
    const EngModelBackend *b = eng_model_probe(path, &score);
    if (!b) {
        fprintf(stderr, "no registered backend recognises %s\n", path);
        fprintf(stderr, "registered: ");
        for (int i = 0; i < eng_model_count(); i++)
            fprintf(stderr, "%s%s", i ? ", " : "", eng_model_at(i)->name);
        fprintf(stderr, "\n");
        return 1;
    }

    EngModelCaps caps;
    char cbuf[128];
    b->caps(&caps);
    eng_model_caps_string(&caps, cbuf, sizeof cbuf);

    printf("model\n");
    printf("  backend   : %s (confidence %d)\n", b->name, score);
    printf("  about     : %s\n", b->description);
    printf("  caps      : %s\n", cbuf);
    if (caps.notes) printf("  notes     : %s\n", caps.notes);

    EngModelFacts f;
    if (b->inspect(path, &f) != 0) {
        fprintf(stderr, "\n%s could not describe this container\n", b->name);
        return 1;
    }

    char t[32];
    printf("  arch      : %s, %d layers\n", f.arch ? f.arch : "?", f.n_layers);
    eng_mem_human(f.total_weight_bytes, t, sizeof t);
    printf("  weights   : %s", f.total_weight_bytes ? t : "unknown");
    if (f.bytes_per_weight > 0.0) printf("  (%.3f bytes/weight)", f.bytes_per_weight);
    printf("\n");
    eng_mem_human(f.avg_layer_bytes, t, sizeof t);
    printf("  per layer : %s", t);
    eng_mem_human(f.max_layer_bytes, t, sizeof t);
    printf(" average, %s largest\n", t);
    if (f.context_max) printf("  context   : up to %d\n", f.context_max);
    if (f.n_experts)
        printf("  experts   : %d, top-%d\n", f.n_experts, f.topk);
    eng_mem_human(f.kv_bytes_per_pos, t, sizeof t);
    printf("  kv cache  : %s per position", t);
    if (f.kv_bytes_per_pos && f.context_max) {
        eng_mem_human(f.kv_bytes_per_pos * f.context_max, t, sizeof t);
        printf(" (%s at full context)", t);
    }
    printf("\n\n");

    if (!(caps.flags & ENG_MCAP_EXECUTE))
        printf("NOTE: this backend can describe the model but not yet execute it.\n"
               "      %s\n\n", caps.notes ? caps.notes : "");

    EngPlan plan;
    EngPlanRequest req;
    eng_plan_request_init(&req);
    if (eng_plan(&plan, &hw, &f, &req) != 0) {
        printf("plan\n  NOT VIABLE: %s\n", plan.problem);
        return 1;
    }
    eng_plan_report(&plan, &f, NULL);
    return 0;
}

/* ---------------------------------------------------------------------- run -- */

typedef struct {
    const char *prompt;
    const char *system;
    int chat;
    int think;
    int   ntok;
    int64_t memory;
    int   threads, context, verbose, resident, bench;
} Opts;

static int parse_opts(int argc, char **argv, int start, Opts *o)
{
    memset(o, 0, sizeof *o);
    o->ntok = 16;
    o->prompt = "Hello, world";
    o->system = NULL;
    /* -1 = auto: use the template when the model has one. A model tuned for
     * chat produces base-model continuations without it, which looks like a
     * quality problem rather than a formatting one. */
    o->chat = -1;
    o->think = 0;

    for (int i = start; i < argc; i++) {
        const char *a = argv[i];
        const char *next = (i + 1 < argc) ? argv[i + 1] : NULL;

        if ((!strcmp(a, "-p") || !strcmp(a, "--prompt")) && next)        { o->prompt = next; i++; }
        else if ((!strcmp(a, "-s") || !strcmp(a, "--system")) && next)   { o->system = next; i++; }
        else if (!strcmp(a, "--chat"))                                   { o->chat = 1; }
        else if (!strcmp(a, "--raw") || !strcmp(a, "--no-chat"))         { o->chat = 0; }
        else if (!strcmp(a, "--think"))                                  { o->think = 1; }
        else if ((!strcmp(a, "-n") || !strcmp(a, "--tokens")) && next)   { o->ntok = atoi(next); i++; }
        else if (!strcmp(a, "--threads") && next)                        { o->threads = atoi(next); i++; }
        else if (!strcmp(a, "--context") && next)                        { o->context = atoi(next); i++; }
        else if (!strcmp(a, "--memory") && next) {
            int is_auto = 0;
            const int64_t v = eng_mem_parse_size(next, &is_auto);
            if (v < 0) {
                /* Refuse rather than guess: a mistyped budget that silently becomes a
                 * different number is how a machine gets OOM-killed. */
                fprintf(stderr, "cannot read --memory '%s'. "
                                "Use a size like 6G, 512M, or 'auto'.\n", next);
                return -1;
            }
            o->memory = is_auto ? 0 : v;
            i++;
        }
        else if (!strcmp(a, "--auto"))      { o->memory = 0; o->threads = 0; }
        else if (!strcmp(a, "--resident") || !strcmp(a, "--no-stream")) { o->resident = 1; }
        else if (!strcmp(a, "-v") || !strcmp(a, "--verbose")) { o->verbose = 1; }
        else {
            fprintf(stderr, "unknown option: %s\n", a);
            return -1;
        }
    }
    if (o->ntok < 1) o->ntok = 1;
    return 0;
}

static int cmd_run(const char *path, const Opts *o)
{
    EngHwInfo hw;
    eng_hwinfo_detect(&hw);
    eng_hwinfo_probe_path(&hw, path);

    int score = 0;
    const EngModelBackend *b = eng_model_probe(path, &score);
    if (!b) { fprintf(stderr, "no backend recognises %s\n", path); return 1; }

    EngModelCaps caps;
    b->caps(&caps);
    if (!(caps.flags & ENG_MCAP_EXECUTE)) {
        fprintf(stderr, "%s can describe this model but not execute it.\n", b->name);
        if (caps.notes) fprintf(stderr, "  %s\n", caps.notes);
        return 1;
    }

    EngModelFacts f;
    if (b->inspect(path, &f) != 0) return 1;

    EngPlanRequest req;
    eng_plan_request_init(&req);
    req.memory_budget = o->memory;
    req.threads = o->threads;
    req.context = o->context;
    req.force_stream = o->resident ? 0 : -1;

    EngPlan plan;
    if (eng_plan(&plan, &hw, &f, &req) != 0) {
        fprintf(stderr, "cannot plan a run: %s\n", plan.problem);
        return 1;
    }
    if (o->verbose) { eng_plan_report(&plan, &f, NULL); printf("\n"); }

#ifdef _OPENMP
    omp_set_num_threads(plan.threads);
#endif
    eng_kernel_select(ENG_IMPL_AUTO);
    if (o->verbose) eng_kernel_report(NULL);

    /* The tokenizer comes from the same container as the weights, so there is no way
     * for them to disagree about the vocabulary. */
    Gguf g;
    EngBpe *tok = NULL;
    EngChat chat;
    memset(&chat, 0, sizeof chat);
    chat.family = ENG_CHAT_NONE;
    if (gguf_open(&g, path) == 0) {
        tok = eng_bpe_from_gguf(&g);
        /* Detected while the container is still OPEN: the template is a pointer into
         * its metadata, so doing this after the close would read freed memory. */
        if (tok) eng_chat_detect(&chat, &g, tok);
        gguf_close(&g);
    }
    if (!tok) {
        fprintf(stderr, "no usable tokenizer in %s\n", path);
        return 1;
    }

    int ids[4096];
    /* CHAT FORMATTING. Built as token IDS rather than a formatted string: the turn
     * markers are single control tokens, and passing them through BPE would merge them
     * into ordinary pieces so the model never sees the boundary it was tuned on. See
     * src/tokenizer/chat.h. */
    const int want_chat = (o->chat >= 0)     ? o->chat
                        : (o->system != NULL) ? 1
                        : (chat.family != ENG_CHAT_NONE);
    if (o->system && o->chat == 0)
        fprintf(stderr, "chat: --system is ignored with --raw\n");

    int n = -1;
    if (want_chat && chat.family != ENG_CHAT_NONE) {
        n = eng_chat_build(&chat, tok, o->system, o->prompt, o->think,
                           ids, (int)(sizeof ids / sizeof *ids));
        if (n < 0)
            fprintf(stderr, "chat: the formatted prompt does not fit; "
                            "falling back to a plain continuation\n");
        else if (o->verbose)
            printf("chat template: %s%s\n", eng_chat_family_name(chat.family),
                   (!o->think && chat.supports_think) ? " (reasoning suppressed)" : "");
    } else if (want_chat) {
        fprintf(stderr, "chat: this model has no template the engine implements; "
                        "treating the prompt as a continuation\n");
    }

    if (n < 0)
        n = eng_bpe_encode(tok, o->prompt, -1, ids, (int)(sizeof ids / sizeof *ids), 1);
    if (n < 0) { fprintf(stderr, "cannot encode the prompt\n"); eng_bpe_free(tok); return 1; }
    if (n == 0) { fprintf(stderr, "the prompt encoded to nothing\n"); eng_bpe_free(tok); return 1; }

    EngLoadReq lr;
    memset(&lr, 0, sizeof lr);
    lr.path = path;
    lr.plan = &plan;
    lr.verbose = o->verbose;

    const double t_load = now_s();
    EngModel *m = b->load(&lr);
    if (!m) { eng_bpe_free(tok); return 1; }
    const double load_s = now_s() - t_load;

    int ctx = plan.context;
    if (ctx > n + o->ntok) ctx = n + o->ntok;
    if (ctx < n + 1) {
        fprintf(stderr, "the context (%d) cannot hold the prompt (%d tokens)\n", ctx, n);
        b->destroy(m);
        eng_bpe_free(tok);
        return 1;
    }
    EngSeqState *st = b->state_create(m, ctx);
    if (!st) { b->destroy(m); eng_bpe_free(tok); return 1; }

    if (o->verbose)
        printf("loaded in %.1f s; prompt is %d token%s\n\n", load_s, n, n == 1 ? "" : "s");

    /* Echo the prompt, then stream the continuation as it is produced. */
    fputs(o->prompt, stdout);
    fflush(stdout);

    double prefill_s = 0.0, decode_s = 0.0;
    int pos = 0, produced = 0;
    int tok_id = ids[0];

    for (; pos < ctx - 1; pos++) {
        /* LOGITS ONLY WHERE THEY ARE READ. While consuming the prompt the next token is
         * already known, so the distribution at that position is computed and discarded.
         * On this model that is a 151,936-row projection and, with a streamed LM head,
         * a 510 MB read -- per prompt token. The sequence state is updated either way,
         * so the KV cache and every later position are unaffected. */
        const int last_prompt = (pos + 1 >= n);
        const uint32_t fl = last_prompt ? ENG_DEC_LOGITS : 0u;

        const double t0 = now_s();
        if (b->decode(m, st, tok_id, pos, fl) != 0) {
            fprintf(stderr, "\ndecode failed\n");
            break;
        }
        const double dt = now_s() - t0;

        if (!last_prompt) {
            /* Still consuming the prompt: the next input is known. */
            prefill_s += dt;
            tok_id = ids[pos + 1];
            continue;
        }
        decode_s += dt;

        int nv = 0;
        const float *lg = b->logits(m, &nv);
        int best = 0;
        for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;

        if (best == eng_bpe_eos(tok)) break;

        /* Control tokens are STRUCTURE, not text. A chat-formatted run can emit one --
         * <|im_start|> opening a turn the model invented -- and printing it would leak
         * the turn markers into the answer. EOS already broke above; this covers the
         * rest. The token is still fed back, so the model's own context is unchanged. */
        if (!eng_bpe_is_special(tok, best)) {
            char piece[512];
            const int pn = eng_bpe_decode_one(tok, best, piece, sizeof piece - 1);
            if (pn > 0) { piece[pn] = '\0'; fputs(piece, stdout); fflush(stdout); }
        }
        if (o->verbose) fprintf(stderr, " [%.1fs]", dt);

        tok_id = best;
        if (++produced >= o->ntok) break;
    }
    printf("\n\n");

    const int prefilled = n > 1 ? n - 1 : 0;
    printf("%d prompt token%s in %.1f s", prefilled, prefilled == 1 ? "" : "s", prefill_s);
    if (prefill_s > 0.0 && prefilled) printf(" (%.2f tok/s)", prefilled / prefill_s);
    printf("\n%d generated in %.1f s", produced, decode_s);
    if (decode_s > 0.0 && produced) printf(" (%.2f tok/s)", produced / decode_s);
    printf("\n");

    /* WHERE THE TIME WENT, in wall seconds throughout, and through the interface rather
     * than by asking a named backend -- see EngRunStats. `stalled` is what the decode
     * waited for; the device time is what the transfer cost. The gap between them is
     * what prefetching hid. */
    if ((o->verbose || o->bench) && b->stats) {
        EngRunStats rs;
        b->stats(m, &rs);

        printf("\nwhere the time went (wall)\n");
        printf("  total       : %7.1f s over %d step%s, %d with logits\n",
               rs.total_s, rs.steps, rs.steps == 1 ? "" : "s", rs.logit_steps);
        printf("  stalled     : %7.1f s waiting for weights\n", rs.io_stall_s);
        printf("  compute     : %7.1f s\n", rs.compute_s);
        printf("  weights     : %.2f GB read in %.1f s of device time (%.0f MB/s)\n",
               (double)rs.bytes_read / 1e9, rs.device_s,
               rs.device_s > 0 ? (double)rs.bytes_read / 1e6 / rs.device_s : 0.0);
        if (rs.notes[0]) printf("  detail      : %s\n", rs.notes);
        if (rs.total_s > 0.0)
            printf("  verdict     : %.0f%% waiting on storage, %.0f%% computing\n",
                   100.0 * rs.io_stall_s / rs.total_s,
                   100.0 * rs.compute_s / rs.total_s);
    }

    b->state_destroy(st);
    b->destroy(m);
    eng_bpe_free(tok);
    return 0;
}

/* --------------------------------------------------------------------- main -- */

int main(int argc, char **argv)
{
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1];

    if (!strcmp(cmd, "-h") || !strcmp(cmd, "--help") || !strcmp(cmd, "help")) {
        usage();
        return 0;
    }

    if (!strcmp(cmd, "list")) {
        printf("registered backends\n");
        for (int i = 0; i < eng_model_count(); i++) {
            const EngModelBackend *b = eng_model_at(i);
            EngModelCaps c;
            char cb[128];
            b->caps(&c);
            eng_model_caps_string(&c, cb, sizeof cb);
            printf("  %-10s %s\n", b->name, b->description);
            printf("  %-10s   caps: %s\n", "", cb);
        }
        return 0;
    }

    if (argc < 3) { usage(); return 1; }
    const char *path = argv[2];

    if (!strcmp(cmd, "inspect")) return cmd_inspect(path);

    if (!strcmp(cmd, "run") || !strcmp(cmd, "benchmark")) {
        Opts o;
        if (parse_opts(argc, argv, 3, &o) != 0) return 1;
        if (!strcmp(cmd, "benchmark")) { o.verbose = 1; o.bench = 1; }
        return cmd_run(path, &o);
    }

    fprintf(stderr, "unknown command: %s\n\n", cmd);
    usage();
    return 1;
}
