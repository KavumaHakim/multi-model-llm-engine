/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_qwen3.c - the Qwen3 forward pass, tensor by tensor, against numpy.
 *
 * WHY PER-TENSOR AND NOT JUST THE TOKEN. The brief is explicit that generating text is
 * not evidence of correctness, and it is right: a wrong RoPE base, the adjacent instead
 * of halved pairing, a missing QK-norm, or GQA head mapping done with % instead of /
 * all produce a model that runs and emits fluent output. Comparing only the final token
 * would pass or fail with no indication of where.
 *
 * So tools/qwen3_ref.py dumps the hidden state after EVERY layer plus seven named
 * intermediates inside layer 0, and this walks them in order. The first tensor that
 * diverges names the bug:
 *
 *   l0.attn_norm  wrong      -> RMSNorm or eps
 *   l0.q_proj     wrong      -> projection orientation or the quant kernel
 *   l0.q_normed   wrong      -> QK-norm missing or applied over the wrong axis
 *   l0.q_roped    wrong      -> RoPE base or pairing convention
 *   l0.attn_heads wrong      -> GQA mapping, causal mask, or the score scale
 *   l0.mlp_out    wrong      -> SwiGLU halves swapped
 *   layer N       wrong      -> a residual, or drift
 *
 * THE REFERENCE SHARES NO DEQUANTISATION PATH WITH THIS ENGINE at the architecture
 * level: it uses tools/gguf_ref.py, which M8 validated bit-exactly against the C
 * kernels. So a disagreement here is an ARCHITECTURE difference, not a decode one --
 * which is what makes a failure locatable to one layer of the stack.
 *
 * TOLERANCES. The C accumulates in fp32 (Qwen3 declares ENG_NUM_FAST); numpy widens to
 * f64 for reductions. So exact agreement is not expected and would in fact be
 * suspicious. What IS expected: layer-0 intermediates agree to ~1e-3 relative, because
 * each is only a few operations deep, and the drift grows slowly and smoothly through
 * the stack. A structural error is not subtle at these tolerances -- it shows up as
 * order-1 relative error at the first affected tensor.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "model.h"
#include "qwen3.h"
#include "streamer.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-30s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

/* ------------------------------------------------------------------- golden -- */

typedef struct {
    char   name[48];
    int    rows, cols;
    float *v;
} Case;

static Case  *g_case = NULL;
static int    g_n = 0;
static int    g_greedy = -1;
static int   *g_tok = NULL;
static int    g_ntok = 0;

static int load_golden(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8];
    uint32_t n = 0, greedy = 0, nt = 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "Q3REF1", 6)) { fclose(f); return -1; }
    if (fread(&n, 4, 1, f) != 1 || fread(&greedy, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&nt, 4, 1, f) != 1 || nt > 1024) { fclose(f); return -1; }

    g_greedy = (int)greedy;
    g_ntok = (int)nt;
    g_tok = (int *)malloc((size_t)nt * sizeof *g_tok);
    if (fread(g_tok, sizeof *g_tok, nt, f) != nt) { fclose(f); return -1; }

    g_case = (Case *)calloc(n, sizeof *g_case);
    for (uint32_t i = 0; i < n; i++) {
        Case *c = &g_case[i];
        uint32_t nl = 0, rows = 0, cols = 0;
        if (fread(&nl, 4, 1, f) != 1 || nl >= sizeof c->name) { fclose(f); return -1; }
        if (fread(c->name, 1, nl, f) != nl) { fclose(f); return -1; }
        c->name[nl] = '\0';
        if (fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1) { fclose(f); return -1; }
        c->rows = (int)rows;
        c->cols = (int)cols;
        c->v = (float *)malloc((size_t)rows * cols * sizeof *c->v);
        if (fread(c->v, sizeof *c->v, (size_t)rows * cols, f) != (size_t)rows * cols) {
            fclose(f);
            return -1;
        }
        g_n++;
    }
    fclose(f);
    return 0;
}

static const Case *find_case(const char *name)
{
    for (int i = 0; i < g_n; i++) if (!strcmp(g_case[i].name, name)) return &g_case[i];
    return NULL;
}

/* --------------------------------------------------------------- comparison -- */

typedef struct {
    int    pos;
    double worst_rel;
    char   worst_name[48];
    int    n_compared, n_missing, n_bad;
    double layer_rel[64];       /* per-layer drift, for the progression printout */
    int    n_layers_seen;
    /* Every tensor's result, kept by name so the layer-0 walk can be printed in
     * pipeline order rather than in whatever order the capture happened to fire. */
    char   seen_name[80][48];
    double seen_rel[80];
    int    n_seen;
} Cmp;

static double rel_for(const Cmp *c, const char *name)
{
    for (int i = 0; i < c->n_seen; i++)
        if (!strcmp(c->seen_name[i], name)) return c->seen_rel[i];
    return -1.0;
}

/* Relative RMS difference: ||a - b|| / ||b||.
 *
 * Chosen over a per-element relative error because these tensors contain values near
 * zero by the thousand, and relative error at such an element is unbounded no matter
 * how correct the arithmetic -- the same trap that made the bf16 kernel check fail on
 * a correct kernel at M5. Normalising by the tensor's own magnitude measures what is
 * actually being asked. */
static double rel_rms(const float *a, const float *b, int n)
{
    double sd = 0.0, sb = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double)a[i] - (double)b[i];
        sd += d * d;
        sb += (double)b[i] * (double)b[i];
    }
    if (sb <= 0.0) return sd > 0.0 ? 1.0 : 0.0;
    return sqrt(sd / sb);
}

static void on_capture(void *ctx, const char *name, const float *v, int n)
{
    Cmp *c = (Cmp *)ctx;
    const Case *g = find_case(name);
    if (!g) { c->n_missing++; return; }

    /* Most goldens hold one row per position; the logits row is the last position only,
     * because that is the only one a decode consumes. */
    int row = c->pos;
    if (g->rows == 1) {
        if (c->pos != g_ntok - 1) return;      /* not this step's business */
        row = 0;
    }
    if (row >= g->rows || n != g->cols) { c->n_missing++; return; }

    const double r = rel_rms(v, g->v + (size_t)row * g->cols, n);
    c->n_compared++;

    if (c->n_seen < (int)(sizeof c->seen_rel / sizeof c->seen_rel[0])) {
        int slot = -1;
        for (int i = 0; i < c->n_seen; i++)
            if (!strcmp(c->seen_name[i], name)) { slot = i; break; }
        if (slot < 0) { slot = c->n_seen++; snprintf(c->seen_name[slot], 48, "%s", name); }
        c->seen_rel[slot] = r;
    }
    if (r > c->worst_rel) {
        c->worst_rel = r;
        snprintf(c->worst_name, sizeof c->worst_name, "%s", name);
    }
    /* A structural error is order-1 here; drift is orders of magnitude below. */
    if (r > 5e-2) c->n_bad++;

    int L = -1;
    if (sscanf(name, "layer%d", &L) == 1 && L >= 0 && L < 64) {
        c->layer_rel[L] = r;
        if (L + 1 > c->n_layers_seen) c->n_layers_seen = L + 1;
    }
}

/* ---------------------------------------------------------------------- main -- */

int main(int argc, char **argv)
{
    const char *golden = argc > 1 ? argv[1] : "tests/fixtures/gguf/qwen3_golden.bin";
    const char *model  = getenv("GGUF_MODEL");
    if (!model) model = "/mnt/c/Users/SHAMI/HAKIM/AI/Qwen3-8B-Q4_K_M.gguf";

    printf("Qwen3 forward pass vs an independent numpy reference\n\n");

    if (load_golden(golden) != 0) {
        printf("  no golden at %s\n", golden);
        printf("  regenerate with: python3 tools/qwen3_ref.py MODEL.gguf tests/fixtures/gguf\n");
        return 1;
    }
    printf("  golden: %d tensors, %d tokens, greedy %d\n", g_n, g_ntok, g_greedy);

    FILE *probe = fopen(model, "rb");
    if (!probe) {
        printf("\n  SKIP  %s is not present.\n", model);
        printf("        This test needs the 5 GB container; the k-quant kernels it\n"
               "        depends on are covered without it by test_gguf.\n");
        return 0;
    }
    fclose(probe);

    /* ---- registry ---- */
    const EngModelBackend *b = eng_model_find("qwen3");
    ok(b != NULL, "qwen3 backend registered", NULL);
    if (!b) return 1;

    int score = 0;
    const EngModelBackend *pb = eng_model_probe(model, &score);
    char d[192];
    snprintf(d, sizeof d, "%s, score %d", pb ? pb->name : "(none)", score);
    ok(pb == b, "probe claims the container", d);

    EngModelCaps caps;
    b->caps(&caps);
    ok((caps.flags & ENG_MCAP_EXECUTE) != 0, "declares execution", NULL);
    ok(caps.num_policy == ENG_NUM_FAST, "declares fast numerics",
       "no bit-identity claim, unlike K3");

    /* ---- inspect ---- */
    EngModelFacts f;
    ok(b->inspect(model, &f) == 0, "inspect", NULL);
    snprintf(d, sizeof d, "%d layers, ctx %d, %.3f bytes/weight",
             f.n_layers, f.context_max, f.bytes_per_weight);
    ok(f.n_layers == 36 && f.context_max == 40960, "architecture facts", d);
    snprintf(d, sizeof d, "avg %.1f MB, max %.1f MB",
             (double)f.avg_layer_bytes / 1e6, (double)f.max_layer_bytes / 1e6);
    ok(f.avg_layer_bytes > 0, "per-layer size measured", d);
    /* K and V, 8 KV heads of 128, fp32, times all 36 layers. Every layer attends
     * independently and keeps its own -- a per-layer figure here would understate the
     * KV cost 36-fold and let the planner accept a context that cannot run. */
    snprintf(d, sizeof d, "%lld bytes/pos = 2*36*8*128*4", (long long)f.kv_bytes_per_pos);
    ok(f.kv_bytes_per_pos == 2LL * 36 * 8 * 128 * 4, "kv per position, all layers", d);

    /* ---- load ---- */
    EngPlan plan;
    memset(&plan, 0, sizeof plan);
    /* Deliberately smaller than the model: this run must stream, because streaming is
     * the case the whole engine exists for and a resident run would not exercise it. */
    plan.stream_budget = 700LL << 20;
    plan.context = g_ntok + 4;

    EngLoadReq req;
    memset(&req, 0, sizeof req);
    req.path = model;
    req.plan = &plan;
    req.verbose = 1;

    printf("\n");
    EngModel *m = b->load(&req);
    ok(m != NULL, "load", "streaming under a 700 MB weight budget");
    if (!m) return 1;

    EngSeqState *st = b->state_create(m, plan.context);
    ok(st != NULL, "state_create", NULL);

    /* ---- decode, comparing every captured tensor ---- */
    printf("\n== forward pass ==\n");
    Cmp cmp;
    memset(&cmp, 0, sizeof cmp);
    qwen3_set_capture(m, on_capture, &cmp);

    /* A decode currently re-reads every layer plus the LM head, so a full run is
     * minutes per token on this host. Capping the count keeps iteration affordable;
     * two tokens is the minimum that exercises the KV cache, since position 1 attends
     * to position 0. */
    int ntok = g_ntok;
    if (argc > 2) {
        const int lim = atoi(argv[2]);
        if (lim > 0 && lim < ntok) ntok = lim;
    }
    if (ntok < g_ntok)
        printf("  (%d of %d tokens; the greedy check needs all %d)\n",
               ntok, g_ntok, g_ntok);

    int rc = 0;
    for (int t = 0; t < ntok; t++) {
        cmp.pos = t;
        const clock_t t0 = clock();
        rc = b->decode(m, st, g_tok[t], t);
        const double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        if (rc != 0) break;
        printf("  token %d (id %6d): %d tensors, worst rel %.3e (%s)  [%.1f s cpu]\n",
               t, g_tok[t], cmp.n_compared, cmp.worst_rel, cmp.worst_name, secs);
        fflush(stdout);
    }
    ok(rc == 0, "decode every token", NULL);

    snprintf(d, sizeof d, "%d compared, %d unmatched", cmp.n_compared, cmp.n_missing);
    ok(cmp.n_compared > 0, "tensors were actually compared", d);

    /* Layer-0 intermediates are only a few operations deep, so they are the tightest
     * and most diagnostic check in the file. */
    /* IN PIPELINE ORDER, because the first one to diverge names the bug. Each line's
     * comment is what a failure there would mean. */
    printf("\n== layer 0, step by step ==\n");
    static const struct { const char *name; const char *means; } L0[] = {
        { "embedding",     "token lookup / embedding dequant" },
        { "l0.attn_norm",  "RMSNorm or eps" },
        { "l0.q_proj",     "projection orientation or the quant kernel" },
        { "l0.k_proj",     "K projection (GQA width)" },
        { "l0.v_proj",     "V projection (q6_k here)" },
        { "l0.q_normed",   "QK-norm on Q: missing, or over the wrong axis" },
        { "l0.k_normed",   "QK-norm on K" },
        { "l0.q_roped",    "RoPE base or pairing convention" },
        { "l0.k_roped",    "RoPE on K" },
        { "l0.attn_heads", "GQA mapping, causal mask, or the score scale" },
        { "l0.attn_out",   "output projection" },
        { "l0.mlp_out",    "SwiGLU: halves swapped, or ffn_down" },
        { "layer0",        "a residual" }
    };
    for (size_t i = 0; i < sizeof L0 / sizeof *L0; i++) {
        const double r = rel_for(&cmp, L0[i].name);
        if (r < 0.0) { printf("  %-16s   (not captured)\n", L0[i].name); continue; }
        printf("  %-16s %9.3e   %s\n", L0[i].name, r,
               r > 5e-2 ? L0[i].means : "");
    }

    printf("\n== drift through the stack ==\n");
    for (int L = 0; L < cmp.n_layers_seen; L += 5)
        printf("  layer %2d: rel %.3e\n", L, cmp.layer_rel[L]);
    if (cmp.n_layers_seen)
        printf("  layer %2d: rel %.3e  (last)\n",
               cmp.n_layers_seen - 1, cmp.layer_rel[cmp.n_layers_seen - 1]);

    snprintf(d, sizeof d, "worst rel %.3e on %s", cmp.worst_rel, cmp.worst_name);
    ok(cmp.n_bad == 0, "every tensor matches the reference", d);

    /* ---- the end-to-end gate ----
     * Only meaningful after the whole sequence: the greedy token is a property of the
     * final position, and stopping early would compare against a different prefix. */
    if (ntok == g_ntok) {
        int nv = 0;
        const float *lg = b->logits(m, &nv);
        int arg = 0;
        for (int i = 1; i < nv; i++) if (lg[i] > lg[arg]) arg = i;
        snprintf(d, sizeof d, "got %d want %d (logit %.4f)",
                 arg, g_greedy, (double)lg[arg]);
        ok(arg == g_greedy, "greedy token matches the reference", d);
    } else {
        printf("  ....  greedy token not checked (partial run)\n");
    }

    /* Where the time went. The streamer times only its read loop, so this separates
     * device time from everything else and says which one to attack. */
    {
        EngStreamerStats ss;
        eng_streamer_stats(qwen3_streamer(m), &ss);
        printf("\n== where the time went ==\n");
        printf("  layer reads : %.2f GB in %.1f s (%.0f MB/s)\n",
               (double)ss.bytes_read / 1e9, ss.load_seconds,
               ss.load_seconds > 0 ? (double)ss.bytes_read / 1e6 / ss.load_seconds : 0.0);
        printf("  hits/misses : %llu / %llu  (%d pinned, %d ring slots, %s)\n",
               (unsigned long long)ss.hits, (unsigned long long)ss.misses,
               ss.npin, ss.nslot, ss.async ? "overlapped" : "synchronous");
    }

    b->state_destroy(st);
    b->destroy(m);

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("QWEN3 TESTS FAILED\n"); return 1; }
    printf("QWEN3 TESTS PASSED\n");
    return 0;
}
