/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kimi.c - Kimi K3 behind the model backend interface.
 *
 * WHAT THIS MILESTONE DOES AND DOES NOT DO, stated plainly because the difference
 * matters to anyone reading `caps`:
 *
 *   DOES   probe, inspect and capabilities. That is what `engine inspect` needs, what
 *          sizes the memory plan, and what proves the registry is architecture-neutral.
 *          All of it is answerable from config.json without touching a weight.
 *
 *   DOES NOT  execute. K3's forward pass currently lives inside src/cli/k3_run.c,
 *          tangled with argument parsing, the memory ladder, the draft model and the
 *          gate harness. Extracting it is a migration of its own and is deliberately
 *          not attempted here alongside a new interface -- doing both at once would
 *          leave no way to tell which change broke the determinism hashes.
 *
 *          So ENG_MCAP_EXECUTE is NOT declared, the registry checks that claim against
 *          the vtable, and `run` refuses with a message instead of calling through a
 *          null pointer. The existing `bin/k3` CLI keeps working exactly as before and
 *          remains the way to run K3 today; nothing about it was touched.
 *
 * WHY THE BYTE FIGURES ARE READ, NOT DERIVED
 *   The tempting shortcut is to compute total weight bytes from the architecture:
 *   experts times matrices times width times bits. It is wrong often enough to matter.
 *   K3's routed expert is documented at 33,030,144 parameters, which does not match
 *   3 x latent x hidden for the configured latent of 3584 -- the expert's inner width
 *   is not the field an outside reader would assume. Deriving it would produce a
 *   confident, wrong memory plan.
 *
 *   So sizes come from the safetensors index when the checkpoint is present, and are
 *   reported as UNKNOWN (zero) when it is not. The planner already treats zero as
 *   unknown rather than as none. An honest gap beats a plausible fabrication.
 */
#define _POSIX_C_SOURCE 200809L

#include "model.h"

#include "k3.h"
#include "k3_cfg.h"
#include "k3_st.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------- helpers -- */

static int is_dir(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

/* A config is a small JSON file. Anything larger is not one, and reading it would be a
 * probe with a cost proportional to the thing it is declining to claim -- pointing this
 * backend at a 5 GB GGUF used to read the whole container into memory before deciding
 * it was not K3. A probe must be cheap enough to run over every registered backend. */
#define K3_MAX_CONFIG_BYTES (16 << 20)

static char *slurp(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0 || n > K3_MAX_CONFIG_BYTES) { fclose(f); return NULL; }
    char *b = (char *)malloc((size_t)n + 1);
    if (!b) { fclose(f); return NULL; }
    const size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    b[got] = '\0';
    return b;
}

/* Resolve `path` to the JSON that holds the config: either the file itself, or
 * config.json inside a directory. Caller frees. */
static char *config_path(const char *path, char *buf, size_t cap)
{
    if (is_dir(path)) {
        snprintf(buf, cap, "%s/config.json", path);
        return buf;
    }
    snprintf(buf, cap, "%s", path);
    return buf;
}

/* ---------------------------------------------------------------------- probe -- */

/* K3's signature is the COMBINATION of a delta-attention head count and an MLA latent
 * rank. Either alone is not enough: other linear-attention models have the first and
 * DeepSeek-family models have the second. Requiring both is what stops this backend
 * claiming a container it cannot run. */
static int kimi_probe(const char *path)
{
    if (!path) return 0;
    char pbuf[1024];
    char *cp = config_path(path, pbuf, sizeof pbuf);

    /* Cheap rejection before any allocation: a config starts with '{' after optional
     * whitespace. This is what stops a container in another format being read in full
     * just to be declined. */
    {
        FILE *probe = fopen(cp, "rb");
        if (!probe) return 0;
        int c;
        do { c = fgetc(probe); } while (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        fclose(probe);
        if (c != '{') return 0;
    }

    char *txt = slurp(cp);
    if (!txt) return 0;

    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    int score = 0;

    if (root) {
        /* The fixture nests the config under "config"; a real checkpoint is flat.
         * k3_cfg_load handles both, and so must the probe. */
        jval *c = json_get(root, "config");
        jval *r = c ? c : root;

        const int has_kda = json_get(r, "kda_num_heads") || json_get(r, "linear_num_heads");
        const int has_mla = json_get(r, "kv_lora_rank") != NULL;
        const int has_moe = json_get(r, "num_experts") != NULL;

        if (has_kda && has_mla) score = has_moe ? 95 : 85;

        /* An explicit model_type is worth more than inference from shape. */
        jval *mt = json_get(r, "model_type");
        if (mt && mt->t == J_STR && strstr(mt->str, "kimi")) score = 100;
    }

    /* The whole tree, not just the arena pointer: nodes and strings are individually
     * allocated. probe runs once per registered backend per path, so leaking a tree
     * here would accumulate for every model a long-lived process opens. Safe because
     * nothing above is retained past this point. */
    json_free(root);
    free(arena);
    free(txt);
    return score;
}

/* -------------------------------------------------------------------- inspect -- */

/* Sum the tensor bytes a shard set actually contains, and separate the ones that cannot
 * stream. Returns 0 when no shards are present, which is not an error: inspect must work
 * on a config alone. */
static void measure_shards(const char *dir, int64_t *total, int64_t *resident)
{
    *total = *resident = 0;
    if (!dir || !is_dir(dir)) return;

    K3St st;
    if (k3_st_open(&st, dir) != 0) return;

    for (int i = 0; i < st.nt; i++) {
        const K3Tensor *t = &st.t[i];
        *total += t->nbytes;
        /* Embeddings and the LM head are read for every token at unpredictable indices,
         * so streaming them buys nothing and costs a seek per token. They are the
         * resident floor the planner must reserve before anything else. */
        if (strstr(t->name, "embed_tokens") || strstr(t->name, "lm_head"))
            *resident += t->nbytes;
    }
    k3_st_close(&st);
}

static int kimi_inspect(const char *path, EngModelFacts *out)
{
    if (!path || !out) return -1;
    memset(out, 0, sizeof *out);

    char pbuf[1024];
    char *cp = config_path(path, pbuf, sizeof pbuf);

    /* Parse here rather than calling k3_cfg_load_file, for two reasons that both
     * produced bugs when this was written the short way:
     *
     *   1. The reference fixture nests its config under "config", while a released
     *      checkpoint is flat and k3_cfg_load itself only unwraps "text_config". So the
     *      right node has to be chosen before loading, exactly as probe does.
     *   2. k3_cfg_load RETURNS 1 ON SUCCESS and 0 on failure -- the opposite of the
     *      0-is-success convention every other entry point in this engine uses. Reading
     *      it as 0-is-success made inspect proceed with a zeroed config on failure, and
     *      k3_layer_scratch then divided by a zero head count. */
    char *txt = slurp(cp);
    if (!txt) {
        fprintf(stderr, "kimi-k3: cannot read %s\n", cp);
        return -1;
    }
    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    if (!root) {
        fprintf(stderr, "kimi-k3: %s is not valid JSON\n", cp);
        free(arena); free(txt);
        return -1;
    }
    jval *nested = json_get(root, "config");
    jval *cfg_node = nested ? nested : root;

    K3Cfg c;
    int fa[256];
    const int loaded = k3_cfg_load(&c, fa, (int)(sizeof fa / sizeof *fa), cfg_node, cp);

    /* Safe to release: K3Cfg is scalars only and fa is a plain int array, so nothing
     * below points into the tree. A backend whose config carried strings would have to
     * copy them before this line. */
    json_free(root);
    free(arena);
    free(txt);
    if (loaded != 1) {
        fprintf(stderr, "kimi-k3: cannot read a usable config from %s\n", cp);
        return -1;
    }

    out->arch     = "kimi-k3";
    out->n_layers = c.n_layers;
    out->n_experts = c.n_experts;
    out->topk      = c.topk;

    /* K3 has no rotary embedding and its published context is not carried in the config
     * the engine parses, so this is left at zero rather than invented. The planner reads
     * zero as "the caller must say", which is the truthful state. */
    out->context_max = 0;

    /* Activations and scratch ARE derivable, because they are functions of the config
     * that the engine's own helpers already compute -- the same numbers the run path
     * allocates, not an estimate of them. T=1 is the incremental decode case. */
    {
        const size_t layer_scratch = k3_layer_scratch(&c, 1);
        const size_t moe_scratch   = k3_moe_scratch(&c);
        out->scratch_bytes = (int64_t)(layer_scratch > moe_scratch ? layer_scratch
                                                                   : moe_scratch)
                           * (int64_t)sizeof(float);
        /* Hidden state, the block residual K3 carries for attention residuals, and the
         * logits row. */
        out->activation_bytes = (int64_t)(c.hidden * 3 + c.vocab) * (int64_t)sizeof(float);
    }

    /* Per-position sequence state. MLA caches the EXPANDED keys plus values, which is
     * what k3_mla_cached stores; KDA's recurrent state is per-layer and NOT per
     * position, so it does not belong in a per-position figure and is added to
     * activations by the state allocator instead. */
    out->kv_bytes_per_pos = (int64_t)c.n_heads * (c.qk_nope + c.v_head)
                          * (int64_t)sizeof(float);

    /* Sizes from the checkpoint when it is there, zero when it is not. See the header:
     * deriving these from the architecture produces confident wrong numbers. */
    if (is_dir(path)) {
        measure_shards(path, &out->total_weight_bytes, &out->resident_bytes);
        if (out->total_weight_bytes > 0 && c.n_layers > 0) {
            const int64_t streamable = out->total_weight_bytes - out->resident_bytes;
            out->avg_layer_bytes = streamable / c.n_layers;
            /* Layer 0 is the one dense layer and carries a 33792-wide MLP, so it is the
             * largest by a wide margin. Without per-layer accounting the average is the
             * honest answer for both fields; the streamer measures the real maximum from
             * trunk.json when one exists. */
            out->max_layer_bytes = out->avg_layer_bytes;
        }
    }

    /* MXFP4 experts dominate the parameter count: 4 bits plus one E8M0 byte per 32
     * elements is 0.531 bytes per weight. This is what tells the planner the matmuls are
     * compute-bound enough for SMT to help -- see planner.h. */
    out->bytes_per_weight = c.n_experts > 0 ? 0.531 : 2.0;

    return 0;
}

/* ----------------------------------------------------------------------- caps -- */

static void kimi_caps(EngModelCaps *out)
{
    if (!out) return;
    memset(out, 0, sizeof *out);

    /* ENG_MCAP_EXECUTE is deliberately absent: see the header comment. The registry
     * verifies this claim against the vtable, so declaring it here without implementing
     * decode would be a startup error rather than a crash mid-generation. */
    out->flags = ENG_MCAP_MOE | ENG_MCAP_RECURRENT | ENG_MCAP_STREAMABLE
               | ENG_MCAP_INCREMENTAL;

    /* K3's claim is that output does not depend on how much memory it was given, which
     * requires bit-identical arithmetic across implementations. See kernels/kernel.h. */
    out->num_policy = ENG_NUM_EXACT;

    out->notes = "KDA + MLA + latent MoE; no positional encoding; "
                 "execution still served by bin/k3 pending migration";
}

const EngModelBackend eng_backend_kimi_k3 = {
    .name        = "kimi-k3",
    .description = "Kimi K3: Kimi Delta Attention, gated MLA, Stable LatentMoE",
    .probe       = kimi_probe,
    .inspect     = kimi_inspect,
    .caps        = kimi_caps,
    /* Execution entry points arrive with the forward-pass migration. Left NULL rather
     * than stubbed: a stub that returns an error looks like a supported path that
     * failed, whereas NULL plus the missing capability flag is unambiguous. */
    .load          = NULL,
    .destroy       = NULL,
    .state_create  = NULL,
    .state_destroy = NULL,
    .state_bytes   = NULL,
    .decode        = NULL,
    .logits        = NULL
};
