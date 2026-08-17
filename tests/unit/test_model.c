/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_model.c - the model backend interface and its registry.
 *
 * THE POINT OF THIS FILE IS THE SYNTHETIC BACKEND.
 *
 * An interface with one implementation is not an interface, it is that implementation
 * with extra indirection. Until Qwen3 exists there is nothing to prove the registry and
 * the vtable are actually architecture-neutral rather than K3-shaped -- and by the time
 * Qwen3 exists, discovering the interface is wrong means reworking both.
 *
 * So `toy` below is a complete backend that shares NO code with K3 and disagrees with it
 * on every axis the interface abstracts: dense instead of MoE, positional instead of
 * recurrent, FAST numerics instead of EXACT, per-position state instead of carried
 * state, and it implements the full execution path K3 does not yet have. Driving it
 * through load -> state_create -> decode -> logits -> destroy exercises every entry
 * point against something the interface was not designed around.
 *
 * If the interface has a K3 assumption baked into it, `toy` is where it shows up.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "model.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-50s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

static void eqi(long long got, long long want, const char *what)
{
    char d[96];
    snprintf(d, sizeof d, "got %lld want %lld", got, want);
    ok(got == want, what, d);
}

/* ===================================================================== toy ===== */
/*
 * A dense, positional, fp32 architecture. Deliberately unlike K3 in every respect the
 * interface is supposed to abstract.
 */
#define TOY_VOCAB   16
#define TOY_LAYERS   4
#define TOY_HIDDEN   8

struct EngModel {           /* the interface says this is backend-defined; here it is */
    int   loaded;
    int   n_vocab;
    float logits[TOY_VOCAB];
};

struct EngSeqState {
    int    context;
    int    n_seen;
    float *k;               /* per-POSITION state: [context][hidden] */
};

static int toy_probe(const char *path)
{
    /* Claims only a path ending in ".toy", so it can never contend with a real
     * container and the probe ordering stays meaningful. */
    if (!path) return 0;
    const size_t n = strlen(path);
    return (n > 4 && !strcmp(path + n - 4, ".toy")) ? 50 : 0;
}

static int toy_inspect(const char *path, EngModelFacts *out)
{
    if (!path || !out) return -1;
    memset(out, 0, sizeof *out);
    out->arch               = "toy";
    out->n_layers           = TOY_LAYERS;
    out->total_weight_bytes = 1024 * 1024;
    out->resident_bytes     = 64 * 1024;
    out->avg_layer_bytes    = (1024 * 1024 - 64 * 1024) / TOY_LAYERS;
    out->max_layer_bytes    = out->avg_layer_bytes;
    out->activation_bytes   = TOY_HIDDEN * 4;
    out->scratch_bytes      = 256;
    out->kv_bytes_per_pos   = TOY_HIDDEN * (int)sizeof(float);
    out->context_max        = 64;
    out->bytes_per_weight   = 4.0;      /* dense fp32: bandwidth-bound */
    out->n_experts          = 0;        /* NOT MoE */
    return 0;
}

static EngModel *toy_load(const EngLoadReq *req)
{
    if (!req) return NULL;
    EngModel *m = (EngModel *)calloc(1, sizeof *m);
    if (!m) return NULL;
    m->loaded = 1;
    m->n_vocab = TOY_VOCAB;
    return m;
}

static void toy_destroy(EngModel *m) { free(m); }

static EngSeqState *toy_state_create(EngModel *m, int context)
{
    if (!m || context <= 0) return NULL;
    EngSeqState *s = (EngSeqState *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->context = context;
    s->k = (float *)calloc((size_t)context * TOY_HIDDEN, sizeof *s->k);
    if (!s->k) { free(s); return NULL; }
    return s;
}

static void toy_state_destroy(EngSeqState *s)
{
    if (!s) return;
    free(s->k);
    free(s);
}

static int64_t toy_state_bytes(const EngModel *m, int context)
{
    (void)m;
    return (int64_t)context * TOY_HIDDEN * (int64_t)sizeof(float);
}

static int toy_decode(EngModel *m, EngSeqState *s, int token, int pos, uint32_t flags)
{
    if (!m || !s || pos < 0 || pos >= s->context) return -1;
    if (token < 0 || token >= TOY_VOCAB) return -1;
    /* The state update happens either way; only the logits are optional. */
    const int want_logits = (flags & ENG_DEC_LOGITS) != 0;

    /* Write per-position state, then produce logits that depend on both the token and
     * the position, so a test can tell a real decode from a stub. */
    for (int i = 0; i < TOY_HIDDEN; i++)
        s->k[(size_t)pos * TOY_HIDDEN + i] = (float)(token + i);
    if (pos + 1 > s->n_seen) s->n_seen = pos + 1;

    if (want_logits) {
        for (int v = 0; v < TOY_VOCAB; v++) m->logits[v] = 0.0f;
        m->logits[(token + 1) % TOY_VOCAB] = 1.0f + (float)pos;
    }
    return 0;
}

static const float *toy_logits(const EngModel *m, int *n_vocab)
{
    if (!m) return NULL;
    if (n_vocab) *n_vocab = m->n_vocab;
    return m->logits;
}

static void toy_caps(EngModelCaps *out)
{
    if (!out) return;
    memset(out, 0, sizeof *out);
    out->flags = ENG_MCAP_EXECUTE | ENG_MCAP_POSITIONAL | ENG_MCAP_STREAMABLE
               | ENG_MCAP_INCREMENTAL;
    out->num_policy = ENG_NUM_FAST;     /* the opposite of K3's requirement */
    out->notes = "synthetic dense backend for interface tests";
}

static const EngModelBackend toy_backend = {
    .name          = "toy",
    .description   = "synthetic dense positional model",
    .probe         = toy_probe,
    .inspect       = toy_inspect,
    .load          = toy_load,
    .destroy       = toy_destroy,
    .state_create  = toy_state_create,
    .state_destroy = toy_state_destroy,
    .state_bytes   = toy_state_bytes,
    .decode        = toy_decode,
    .logits        = toy_logits,
    .caps          = toy_caps
};

/* ================================================================= the tests ===== */

/* ------------------------------------------------------- checkpoint fixture -- */
/*
 * Neither fixture directory looks like a real checkpoint: tests/fixtures holds the
 * reference config with no weights, tests/fixtures/st holds two tiny shards with no
 * config. A directory containing BOTH is what kimi_inspect's measuring path expects, so
 * the test builds one under build/ rather than reshaping the fixtures other tests rely
 * on.
 */
#define CKPT_DIR "build/k3_ckpt_fixture"

static int copy_file(const char *from, const char *to)
{
    FILE *in = fopen(from, "rb");
    if (!in) return -1;
    FILE *out = fopen(to, "wb");
    if (!out) { fclose(in); return -1; }
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, in)) > 0)
        if (fwrite(buf, 1, n, out) != n) { fclose(in); fclose(out); return -1; }
    fclose(in);
    fclose(out);
    return 0;
}

static int build_checkpoint_fixture(void)
{
    if (mkdir(CKPT_DIR, 0755) != 0 && errno != EEXIST) return -1;
    if (copy_file("tests/fixtures/ref_k3.json", CKPT_DIR "/config.json") != 0) return -1;
    if (copy_file("tests/fixtures/st/model-00001-of-00002.safetensors",
                  CKPT_DIR "/model-00001-of-00002.safetensors") != 0) return -1;
    if (copy_file("tests/fixtures/st/model-00002-of-00002.safetensors",
                  CKPT_DIR "/model-00002-of-00002.safetensors") != 0) return -1;
    return 0;
}

static void test_registry(void)
{
    printf("== registry ==\n");

    ok(eng_model_count() >= 1, "built-in backends self-register", NULL);

    const EngModelBackend *k = eng_model_find("kimi-k3");
    ok(k != NULL, "kimi-k3 is registered", NULL);
    ok(eng_model_find("nope") == NULL, "unknown name returns NULL", NULL);
    ok(eng_model_find(NULL) == NULL, "NULL name returns NULL", NULL);

    ok(eng_model_register(&toy_backend) == 0, "a new backend registers", "toy");
    ok(eng_model_find("toy") != NULL, "and is then findable", NULL);
    ok(eng_model_register(&toy_backend) == -1, "re-registering a name is refused",
       "shadowing must not be silent");

    /* A backend missing probe/inspect/caps would register and fail later. */
    {
        EngModelBackend bad;
        memset(&bad, 0, sizeof bad);
        bad.name = "incomplete";
        ok(eng_model_register(&bad) == -1, "backend without probe/inspect/caps refused",
           NULL);
    }

    /* Claiming ENG_MCAP_EXECUTE without implementing the run path must fail at STARTUP,
     * not with a null-pointer call during generation. */
    {
        static EngModelBackend liar;
        memset(&liar, 0, sizeof liar);
        liar.name    = "liar";
        liar.probe   = toy_probe;
        liar.inspect = toy_inspect;
        liar.caps    = toy_caps;     /* declares EXECUTE */
        ok(eng_model_register(&liar) == -1,
           "EXECUTE claimed without a decode path is refused",
           "checked against the vtable at registration");
    }
}

static void test_probe(void)
{
    printf("\n== probe ==\n");

    int score = 0;
    const EngModelBackend *b = eng_model_probe("something.toy", &score);
    char d[96];
    snprintf(d, sizeof d, "score %d", score);
    ok(b && !strcmp(b->name, "toy"), "toy claims its own extension", d);

    ok(eng_model_probe("/definitely/not/a/model", &score) == NULL,
       "no backend claims an unrelated path", NULL);
    ok(eng_model_probe(NULL, NULL) == NULL, "NULL path is safe", NULL);

    /* K3 must claim the reference fixture, which is a config in the nested shape. */
    b = eng_model_probe("tests/fixtures/ref_k3.json", &score);
    snprintf(d, sizeof d, "score %d", score);
    ok(b && !strcmp(b->name, "kimi-k3"), "kimi-k3 claims the reference config", d);

    /* And must NOT claim a config that lacks its signature. Both a delta-attention head
     * count and an MLA latent rank are required: other linear-attention models have the
     * first, DeepSeek-family models have the second. */
    ok(eng_model_find("kimi-k3")->probe("tests/fixtures/mxfp4.json") == 0,
       "kimi-k3 refuses an unrelated json", "needs BOTH kda and mla markers");
}

static void test_kimi_inspect(void)
{
    printf("\n== kimi-k3 inspect ==\n");

    const EngModelBackend *k = eng_model_find("kimi-k3");
    if (!k) { ok(0, "kimi-k3 present", NULL); return; }

    EngModelFacts f;
    ok(k->inspect("tests/fixtures/ref_k3.json", &f) == 0,
       "inspect reads the reference config", NULL);

    /* Against the fixture's own values, so a config-parsing regression is caught here
     * rather than as a wrong memory plan much later. */
    eqi(f.n_layers,  13, "n_layers from the config");
    eqi(f.n_experts,  8, "n_experts from the config");
    eqi(f.topk,       2, "topk from the config");
    ok(f.arch && !strcmp(f.arch, "kimi-k3"), "arch is reported", f.arch);

    /* n_heads 4, qk_nope 24, v_head 16 -> 4 * (24+16) * 4 bytes. Hand-computed so a
     * change in the MLA cache layout has to be deliberate. */
    eqi(f.kv_bytes_per_pos, 4 * (24 + 16) * 4, "kv bytes per position");

    ok(f.scratch_bytes > 0, "scratch derived from the config", NULL);
    ok(f.activation_bytes > 0, "activations derived from the config", NULL);

    /* MXFP4 experts: 4 bits plus one E8M0 byte per 32 elements. This is what tells the
     * planner to use logical rather than physical cores. */
    char d[64];
    snprintf(d, sizeof d, "%.3f", f.bytes_per_weight);
    ok(f.bytes_per_weight > 0.0 && f.bytes_per_weight < 1.0,
       "bytes/weight marks it compute-bound", d);

    /* Sizes are UNKNOWN without a checkpoint, and must read as zero rather than as a
     * derived guess. The planner treats zero as unknown. */
    snprintf(d, sizeof d, "total=%lld", (long long)f.total_weight_bytes);
    ok(f.total_weight_bytes == 0, "byte totals are zero without shards", d);

    /* With shards present they are MEASURED rather than derived.
     *
     * Neither fixture directory is shaped like a checkpoint on its own -- tests/fixtures
     * has the config but no shards, tests/fixtures/st has shards but no config -- so the
     * test assembles one. Without this the shard-measuring path in kimi_inspect has no
     * coverage at all, which is exactly where a wrong byte total would come from. */
    if (build_checkpoint_fixture() == 0) {
        EngModelFacts g;
        if (k->inspect(CKPT_DIR, &g) == 0) {
            snprintf(d, sizeof d, "total=%lld", (long long)g.total_weight_bytes);
            ok(g.total_weight_bytes > 0, "byte totals measured from a shard set", d);

            /* The two shards are 2906 + 973 bytes of file, of which the tensor payload
             * is whatever sits past each header. So the total must be positive and
             * strictly under the files' combined size -- a figure that would be wrong
             * in both directions if headers were counted or a shard were missed. */
            ok(g.total_weight_bytes < 2906 + 973,
               "measured payload excludes the shard headers", d);

            snprintf(d, sizeof d, "avg layer %lld over %d layers",
                     (long long)g.avg_layer_bytes, g.n_layers);
            ok(g.avg_layer_bytes > 0, "per-layer size derived from the measurement", d);
            ok(g.max_layer_bytes >= g.avg_layer_bytes,
               "max layer is not below the average", NULL);
        } else {
            ok(0, "inspect reads an assembled checkpoint directory", CKPT_DIR);
        }
    } else {
        ok(0, "could not assemble a checkpoint fixture", CKPT_DIR);
    }
}

static void test_capabilities(void)
{
    printf("\n== capabilities ==\n");

    char buf[128];
    const EngModelBackend *k = eng_model_find("kimi-k3");
    EngModelCaps kc;
    k->caps(&kc);
    eng_model_caps_string(&kc, buf, sizeof buf);

    ok(kc.num_policy == ENG_NUM_EXACT, "kimi-k3 requires exact arithmetic",
       "its output must not depend on the memory budget");
    ok((kc.flags & ENG_MCAP_MOE) != 0, "kimi-k3 declares MoE", buf);
    ok((kc.flags & ENG_MCAP_RECURRENT) != 0, "kimi-k3 declares recurrent state", NULL);
    ok((kc.flags & ENG_MCAP_POSITIONAL) == 0,
       "kimi-k3 declares NO positional encoding", "MLA is NoPE, KDA recurs");
    ok((kc.flags & ENG_MCAP_EXECUTE) == 0,
       "kimi-k3 does not yet claim execution",
       "bin/k3 still serves the forward pass");

    EngModelCaps tc;
    toy_caps(&tc);
    ok(tc.num_policy == ENG_NUM_FAST, "toy requires only fast arithmetic",
       "the opposite of K3, through one interface");
    ok((tc.flags & ENG_MCAP_POSITIONAL) != 0, "toy declares positional encoding", NULL);
    ok((tc.flags & ENG_MCAP_MOE) == 0, "toy is dense", NULL);
}

/* THE ONE THAT MATTERS: a full run through the interface on a backend the interface was
 * not designed around. */
static void test_toy_execution(void)
{
    printf("\n== full execution path (synthetic backend) ==\n");

    const EngModelBackend *b = eng_model_find("toy");
    if (!b) { ok(0, "toy registered", NULL); return; }

    EngModelCaps c;
    b->caps(&c);
    ok((c.flags & ENG_MCAP_EXECUTE) != 0, "toy claims execution", NULL);

    EngLoadReq req;
    memset(&req, 0, sizeof req);
    req.path = "unit.toy";

    EngModel *m = b->load(&req);
    ok(m != NULL, "load", NULL);
    if (!m) return;

    /* state_bytes must be answerable WITHOUT allocating: the planner needs it before it
     * can choose a context length. */
    const int64_t want = b->state_bytes(m, 32);
    char d[96];
    snprintf(d, sizeof d, "%lld bytes for 32 positions", (long long)want);
    ok(want > 0, "state_bytes answers without allocating", d);

    EngSeqState *s = b->state_create(m, 32);
    ok(s != NULL, "state_create", NULL);

    /* Decode a short sequence and check the logits actually depend on what was fed in.
     * A stub returning a constant would pass a weaker test. */
    int argmax_ok = 1;
    for (int pos = 0; pos < 5; pos++) {
        const int tok = pos * 3 + 1;
        if (b->decode(m, s, tok, pos, ENG_DEC_LOGITS) != 0) { argmax_ok = 0; break; }
        int nv = 0;
        const float *lg = b->logits(m, &nv);
        if (!lg || nv != TOY_VOCAB) { argmax_ok = 0; break; }
        int best = 0;
        for (int v = 1; v < nv; v++) if (lg[v] > lg[best]) best = v;
        if (best != (tok + 1) % TOY_VOCAB) { argmax_ok = 0; break; }
    }
    ok(argmax_ok, "decode then logits, 5 incremental steps",
       "logits track the fed token, not a constant");

    /* pos is an argument rather than hidden state, so a rewind is expressible. That is
     * what speculative decoding and beam search need. */
    ok(b->decode(m, s, 4, 2, ENG_DEC_LOGITS) == 0, "decode can rewind to an earlier position",
       "pos is explicit, not an internal counter");

    ok(b->decode(m, s, 4, 9999, ENG_DEC_LOGITS) != 0, "decode refuses a position past the context", NULL);
    ok(b->decode(m, s, -1, 0, ENG_DEC_LOGITS) != 0, "decode refuses an invalid token", NULL);

    b->state_destroy(s);
    b->destroy(m);
    ok(1, "state_destroy and destroy", NULL);
}

int main(void)
{
    /* Fixture paths are relative to the repo root, which is where make runs tests. */
    printf("model backend interface\n\n");
    test_registry();
    test_probe();
    test_kimi_inspect();
    test_capabilities();
    test_toy_execution();

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("MODEL TESTS FAILED\n"); return 1; }
    printf("MODEL TESTS PASSED\n");
    return 0;
}
