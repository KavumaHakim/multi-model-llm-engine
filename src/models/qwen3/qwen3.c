/* SPDX-License-Identifier: Apache-2.0 */
/* qwen3.c - see qwen3.h. */
#define _POSIX_C_SOURCE 200809L

#include "qwen3.h"

#include "gguf.h"
#include "kernel.h"
#include "quant.h"
#include "streamer.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ layout -- */

/* Where one weight sits inside its layer's streamed run. */
typedef struct {
    int64_t  rel;        /* byte offset within the layer block */
    int64_t  nbytes;
    EngDType dtype;
    int64_t  in, out;    /* shape[0] is the input width, shape[1] the row count */
} Slot;

typedef struct {
    Slot attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_out;
    Slot ffn_norm, ffn_gate, ffn_up, ffn_down;
} LayerSlots;

struct EngModel {
    Gguf gguf;

    int   n_layers, d_model, n_heads, n_kv, head_dim, d_ff, vocab, ctx_max;
    float eps, rope_theta;

    EngStreamer *stream;
    LayerSlots  *slot;

    /* Resident: only the final norm, which is 16 KB. */
    float *out_norm;

    /* Read on demand. See qwen3.h for why neither is held resident. */
    const GgufTensor *embd;
    const GgufTensor *lm_head;
    unsigned char    *rowbuf;      /* one embedding row, quantised */
    unsigned char    *headbuf;     /* a chunk of LM head rows      */
    int               head_chunk;  /* rows per chunk               */

    /* Scratch, all sized at load. */
    float *x, *xn, *q, *k, *v, *att, *gate, *up, *ffn, *scores;
    float *logits;

    Qwen3Capture cap;
    void        *cap_ctx;
    int          verbose;
};

struct EngSeqState {
    int    context, n_seen;
    float *kc;      /* [context][n_kv * head_dim] */
    float *vc;
};

static void emit(EngModel *m, const char *name, const float *v, int n)
{
    if (m->cap) m->cap(m->cap_ctx, name, v, n);
}

void qwen3_set_capture(EngModel *m, Qwen3Capture cb, void *ctx)
{
    if (!m) return;
    m->cap = cb;
    m->cap_ctx = ctx;
}

struct EngStreamer *qwen3_streamer(EngModel *m) { return m ? m->stream : NULL; }

/* --------------------------------------------------------------------- probe -- */

static int qwen3_probe(const char *path)
{
    if (!path) return 0;
    /* Open only far enough to read general.architecture. gguf_open reads the header and
     * nothing else, so this costs a few MB and no weights. */
    Gguf g;
    if (gguf_open(&g, path) != 0) return 0;
    const char *arch = NULL;
    int64_t n = 0;
    int score = 0;
    if (gguf_str(&g, "general.architecture", &arch, &n) == 0 &&
        n == 5 && !memcmp(arch, "qwen3", 5))
        score = 100;
    gguf_close(&g);
    return score;
}

/* ------------------------------------------------------------------- inspect -- */

/* Sum the byte span of one layer's tensors, and find where the run starts. Relies on
 * the layer's tensors being adjacent, which gguf_layout_is_sequential verifies. */
static int layer_span(const Gguf *g, int L, int64_t *off, int64_t *len)
{
    char pre[32];
    const int n = snprintf(pre, sizeof pre, "blk.%d.", L);
    int64_t lo = -1, hi = -1;
    for (int64_t i = 0; i < g->n_tensors; i++) {
        if (strncmp(g->t[i].name, pre, (size_t)n)) continue;
        const int64_t a = g->t[i].file_off, b = a + g->t[i].nbytes;
        if (lo < 0 || a < lo) lo = a;
        if (hi < 0 || b > hi) hi = b;
    }
    if (lo < 0) return -1;
    *off = lo;
    *len = hi - lo;
    return 0;
}

static int qwen3_inspect(const char *path, EngModelFacts *out)
{
    if (!path || !out) return -1;
    memset(out, 0, sizeof *out);

    Gguf g;
    if (gguf_open(&g, path) != 0) return -1;

    int64_t v = 0;
    int ok = 1;
    ok &= gguf_i64(&g, "qwen3.block_count", &v) == 0;      const int nl = (int)v;
    ok &= gguf_i64(&g, "qwen3.embedding_length", &v) == 0; const int dm = (int)v;
    ok &= gguf_i64(&g, "qwen3.attention.head_count", &v) == 0;    const int nh = (int)v;
    ok &= gguf_i64(&g, "qwen3.attention.head_count_kv", &v) == 0; const int nkv = (int)v;
    ok &= gguf_i64(&g, "qwen3.attention.key_length", &v) == 0;    const int hd = (int)v;
    ok &= gguf_i64(&g, "qwen3.feed_forward_length", &v) == 0;     const int ff = (int)v;
    ok &= gguf_i64(&g, "qwen3.context_length", &v) == 0;          const int cl = (int)v;
    if (!ok) {
        fprintf(stderr, "qwen3: the container is missing a required architecture key\n");
        gguf_close(&g);
        return -1;
    }

    out->arch        = "qwen3";
    out->n_layers    = nl;
    out->context_max = cl;
    out->n_experts   = 0;                  /* dense */

    const GgufTensor *emb = gguf_tensor(&g, "token_embd.weight");
    const GgufTensor *lmh = gguf_tensor(&g, "output.weight");
    const GgufTensor *onm = gguf_tensor(&g, "output_norm.weight");
    if (!emb || !lmh || !onm) {
        fprintf(stderr, "qwen3: token_embd, output or output_norm is missing\n");
        gguf_close(&g);
        return -1;
    }

    out->total_weight_bytes = g.data_bytes;
    /* Only output_norm is genuinely resident. The embedding and LM head are read on
     * demand: holding 860 MB to serve a few kilobytes per step is a poor trade at the
     * budgets this engine targets. Counting them as resident would make the planner
     * refuse configurations that in fact run. */
    out->resident_bytes = onm->nbytes;

    int64_t off = 0, len = 0, big = 0, sum = 0;
    for (int L = 0; L < nl; L++) {
        if (layer_span(&g, L, &off, &len) != 0) continue;
        if (len > big) big = len;
        sum += len;
    }
    out->max_layer_bytes = big;
    out->avg_layer_bytes = nl ? sum / nl : 0;

    /* One position of K and V, fp32, ACROSS EVERY LAYER. Each layer attends
     * independently and keeps its own K and V, so a position costs 36 times one
     * layer's worth. Reporting a single layer here would let the planner accept a
     * context 36x larger than the memory it actually needs -- the plan would look
     * comfortable and the run would be killed.
     *
     * GQA is still what makes this affordable: 8 KV heads rather than 32 is a 4x
     * reduction against a model of the same hidden width. */
    out->kv_bytes_per_pos = (int64_t)2 * nl * nkv * hd * (int64_t)sizeof(float);

    /* Hidden states and the per-step intermediates, plus the logits row. */
    out->activation_bytes = (int64_t)(dm * 6 + nh * hd + 2 * nkv * hd + 2 * ff)
                          * (int64_t)sizeof(float)
                          + (int64_t)emb->shape[1] * (int64_t)sizeof(float);
    out->scratch_bytes = (int64_t)cl * (int64_t)sizeof(float)   /* attention scores */
                       + 16 * 1024 * 1024;                      /* LM head chunk    */

    /* Q4_K is 4.5 bits and Q6_K is 6.5625; the mixture across this file works out just
     * over half a byte per weight, which tells the planner the matmuls are quantised
     * enough for SMT to help. Computed rather than assumed: total bytes over total
     * elements. */
    int64_t elems = 0;
    for (int64_t i = 0; i < g.n_tensors; i++) {
        int64_t e = 1;
        for (int d = 0; d < g.t[i].n_dims; d++) e *= g.t[i].shape[d];
        elems += e;
    }
    out->bytes_per_weight = elems ? (double)g.data_bytes / (double)elems : 0.0;

    gguf_close(&g);
    return 0;
}

/* ---------------------------------------------------------------------- load -- */

static int bind_slot(const Gguf *g, Slot *s, const char *fmt, int L, int64_t base)
{
    char nm[64];
    snprintf(nm, sizeof nm, fmt, L);
    const GgufTensor *t = gguf_tensor(g, nm);
    if (!t) {
        fprintf(stderr, "qwen3: missing tensor %s\n", nm);
        return -1;
    }
    s->rel    = t->file_off - base;
    s->nbytes = t->nbytes;
    s->dtype  = t->dtype;
    s->in     = t->shape[0];
    s->out    = t->n_dims > 1 ? t->shape[1] : 1;
    return 0;
}

static EngModel *qwen3_load(const EngLoadReq *req)
{
    if (!req || !req->path) return NULL;

    EngModel *m = (EngModel *)calloc(1, sizeof *m);
    if (!m) return NULL;
    m->verbose = req->verbose;

    if (gguf_open(&m->gguf, req->path) != 0) { free(m); return NULL; }
    Gguf *g = &m->gguf;

    if (!gguf_layout_is_sequential(g)) {
        /* Not fatal, but the streaming plan assumes a layer is one forward run. Say so
         * rather than stream pathologically without explanation. */
        fprintf(stderr, "qwen3: WARNING: tensor offsets are not monotonic; "
                        "streaming will seek within layers\n");
    }

    int64_t v = 0;
    gguf_i64(g, "qwen3.block_count", &v);           m->n_layers = (int)v;
    gguf_i64(g, "qwen3.embedding_length", &v);      m->d_model  = (int)v;
    gguf_i64(g, "qwen3.attention.head_count", &v);  m->n_heads  = (int)v;
    gguf_i64(g, "qwen3.attention.head_count_kv", &v); m->n_kv   = (int)v;
    gguf_i64(g, "qwen3.attention.key_length", &v);  m->head_dim = (int)v;
    gguf_i64(g, "qwen3.feed_forward_length", &v);   m->d_ff     = (int)v;
    gguf_i64(g, "qwen3.context_length", &v);        m->ctx_max  = (int)v;
    gguf_f32(g, "qwen3.attention.layer_norm_rms_epsilon", &m->eps);
    gguf_f32(g, "qwen3.rope.freq_base", &m->rope_theta);

    m->embd    = gguf_tensor(g, "token_embd.weight");
    m->lm_head = gguf_tensor(g, "output.weight");
    const GgufTensor *onm = gguf_tensor(g, "output_norm.weight");
    if (!m->embd || !m->lm_head || !onm) {
        fprintf(stderr, "qwen3: token_embd, output or output_norm is missing\n");
        goto fail;
    }
    m->vocab = (int)m->embd->shape[1];

    if (m->n_heads % m->n_kv) {
        fprintf(stderr, "qwen3: %d query heads do not divide into %d KV heads\n",
                m->n_heads, m->n_kv);
        goto fail;
    }
    if (m->n_heads * m->head_dim != m->d_model) {
        /* Not universal across architectures, but true here, and a mismatch means the
         * head reshape below is wrong. */
        fprintf(stderr, "qwen3: %d heads x %d dim != %d hidden\n",
                m->n_heads, m->head_dim, m->d_model);
        goto fail;
    }

    /* ---- per-layer blocks for the streamer ---- */
    EngBlock *blocks = (EngBlock *)calloc((size_t)m->n_layers, sizeof *blocks);
    m->slot = (LayerSlots *)calloc((size_t)m->n_layers, sizeof *m->slot);
    if (!blocks || !m->slot) { free(blocks); goto fail; }

    for (int L = 0; L < m->n_layers; L++) {
        int64_t off = 0, len = 0;
        if (layer_span(g, L, &off, &len) != 0) {
            fprintf(stderr, "qwen3: layer %d has no tensors\n", L);
            free(blocks);
            goto fail;
        }
        blocks[L].off = off;
        blocks[L].nbytes = len;

        LayerSlots *s = &m->slot[L];
        int rc = 0;
        rc |= bind_slot(g, &s->attn_norm,   "blk.%d.attn_norm.weight",   L, off);
        rc |= bind_slot(g, &s->attn_q,      "blk.%d.attn_q.weight",      L, off);
        rc |= bind_slot(g, &s->attn_k,      "blk.%d.attn_k.weight",      L, off);
        rc |= bind_slot(g, &s->attn_v,      "blk.%d.attn_v.weight",      L, off);
        rc |= bind_slot(g, &s->attn_q_norm, "blk.%d.attn_q_norm.weight", L, off);
        rc |= bind_slot(g, &s->attn_k_norm, "blk.%d.attn_k_norm.weight", L, off);
        rc |= bind_slot(g, &s->attn_out,    "blk.%d.attn_output.weight", L, off);
        rc |= bind_slot(g, &s->ffn_norm,    "blk.%d.ffn_norm.weight",    L, off);
        rc |= bind_slot(g, &s->ffn_gate,    "blk.%d.ffn_gate.weight",    L, off);
        rc |= bind_slot(g, &s->ffn_up,      "blk.%d.ffn_up.weight",      L, off);
        rc |= bind_slot(g, &s->ffn_down,    "blk.%d.ffn_down.weight",    L, off);
        if (rc) { free(blocks); goto fail; }
    }

    {
        EngStreamerCfg cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.name  = "qwen3-layer";
        cfg.store = g->store;
        cfg.blocks = blocks;
        cfg.nblocks = m->n_layers;
        cfg.budget_bytes = req->plan ? req->plan->stream_budget : (512LL << 20);
        cfg.ring_want = 2;
        cfg.async = 1;
        cfg.hugepages = -1;
        cfg.quiet = !req->verbose;
        m->stream = eng_streamer_create(&cfg);
        free(blocks);
        if (!m->stream) goto fail;
    }

    /* ---- resident final norm ---- */
    m->out_norm = (float *)malloc((size_t)onm->nbytes);
    if (!m->out_norm ||
        g->store->read(g->store, onm->file_off, onm->nbytes, m->out_norm) != onm->nbytes) {
        fprintf(stderr, "qwen3: cannot read output_norm\n");
        goto fail;
    }

    /* ---- on-demand buffers ---- */
    {
        const EngQuantOps *eq = eng_quant_ops(m->embd->dtype);
        const int64_t erow = eq ? eq->row_bytes(m->d_model)
                                : eng_dtype_row_bytes(m->embd->dtype, m->d_model);
        m->rowbuf = (unsigned char *)malloc((size_t)erow);

        const EngQuantOps *hq = eng_quant_ops(m->lm_head->dtype);
        const int64_t hrow = hq ? hq->row_bytes(m->d_model)
                                : eng_dtype_row_bytes(m->lm_head->dtype, m->d_model);
        /* ~16 MB of LM head per read: deep enough that the device is not idling
         * between chunks, small enough to sit inside any sane scratch budget. */
        m->head_chunk = (int)((16 << 20) / (hrow > 0 ? hrow : 1));
        if (m->head_chunk < 1) m->head_chunk = 1;
        if (m->head_chunk > m->vocab) m->head_chunk = m->vocab;
        m->headbuf = (unsigned char *)malloc((size_t)m->head_chunk * (size_t)hrow);
        if (!m->rowbuf || !m->headbuf) goto fail;
    }

    /* ---- scratch ---- */
    m->x      = (float *)calloc((size_t)m->d_model, sizeof(float));
    m->xn     = (float *)calloc((size_t)m->d_model, sizeof(float));
    m->q      = (float *)calloc((size_t)m->n_heads * m->head_dim, sizeof(float));
    m->k      = (float *)calloc((size_t)m->n_kv * m->head_dim, sizeof(float));
    m->v      = (float *)calloc((size_t)m->n_kv * m->head_dim, sizeof(float));
    m->att    = (float *)calloc((size_t)m->n_heads * m->head_dim, sizeof(float));
    m->gate   = (float *)calloc((size_t)m->d_ff, sizeof(float));
    m->up     = (float *)calloc((size_t)m->d_ff, sizeof(float));
    m->ffn    = (float *)calloc((size_t)m->d_model, sizeof(float));
    m->scores = (float *)calloc((size_t)m->ctx_max + 1, sizeof(float));
    m->logits = (float *)calloc((size_t)m->vocab, sizeof(float));
    if (!m->x || !m->xn || !m->q || !m->k || !m->v || !m->att ||
        !m->gate || !m->up || !m->ffn || !m->scores || !m->logits) goto fail;

    if (req->verbose)
        printf("qwen3: %d layers, d_model %d, heads %d/%d, head_dim %d, ffn %d, "
               "vocab %d, ctx %d, eps %.1e, rope %.0f\n",
               m->n_layers, m->d_model, m->n_heads, m->n_kv, m->head_dim,
               m->d_ff, m->vocab, m->ctx_max, (double)m->eps, (double)m->rope_theta);
    return m;

fail:
    if (m) {
        eng_streamer_destroy(m->stream);
        gguf_close(&m->gguf);
        free(m->slot); free(m->out_norm); free(m->rowbuf); free(m->headbuf);
        free(m->x); free(m->xn); free(m->q); free(m->k); free(m->v); free(m->att);
        free(m->gate); free(m->up); free(m->ffn); free(m->scores); free(m->logits);
        free(m);
    }
    return NULL;
}

static void qwen3_destroy(EngModel *m)
{
    if (!m) return;
    eng_streamer_destroy(m->stream);
    gguf_close(&m->gguf);
    free(m->slot); free(m->out_norm); free(m->rowbuf); free(m->headbuf);
    free(m->x); free(m->xn); free(m->q); free(m->k); free(m->v); free(m->att);
    free(m->gate); free(m->up); free(m->ffn); free(m->scores); free(m->logits);
    free(m);
}

/* --------------------------------------------------------------------- state -- */

static EngSeqState *qwen3_state_create(EngModel *m, int context)
{
    if (!m || context <= 0) return NULL;
    EngSeqState *s = (EngSeqState *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->context = context;
    /* PER LAYER, per position. Attention is computed independently in each of the 36
     * layers, so each keeps its own K and V for every position -- the cache is indexed
     * [layer][position][kv_head * head_dim]. Sizing this for one layer and indexing it
     * for 36 overruns the allocation on the very first layer past 0. */
    const size_t n = (size_t)m->n_layers * (size_t)context
                   * (size_t)(m->n_kv * m->head_dim);
    s->kc = (float *)calloc(n, sizeof(float));
    s->vc = (float *)calloc(n, sizeof(float));
    if (!s->kc || !s->vc) { free(s->kc); free(s->vc); free(s); return NULL; }
    return s;
}

static void qwen3_state_destroy(EngSeqState *s)
{
    if (!s) return;
    free(s->kc);
    free(s->vc);
    free(s);
}

static int64_t qwen3_state_bytes(const EngModel *m, int context)
{
    if (!m) return 0;
    /* K and V, for every layer and every position. */
    return (int64_t)2 * m->n_layers * context * m->n_kv * m->head_dim
         * (int64_t)sizeof(float);
}

/* ------------------------------------------------------------------- forward -- */

/* One projection: y = W . x, dispatching on how W is stored. */
static int project(float *y, const float *x, const unsigned char *base,
                   const Slot *s)
{
    if (s->dtype == ENG_DT_F32) {
        eng_matmul_f32(y, x, (const float *)(base + s->rel),
                       (int)s->in, (int)s->out, ENG_NUM_FAST);
        return 0;
    }
    return eng_matmul_quant(y, x, base + s->rel, NULL,
                            (int)s->in, (int)s->out, s->dtype);
}

/* RMSNorm applied per head over head_dim. Qwen3's QK-norm: after the projection and
 * before RoPE. */
static void head_norm(float *v, int nheads, int hd, const float *w, float eps)
{
    for (int h = 0; h < nheads; h++)
        eng_rmsnorm(v + (size_t)h * hd, v + (size_t)h * hd, w, hd, eps);
}

static int qwen3_decode(EngModel *m, EngSeqState *s, int token, int pos)
{
    if (!m || !s) return -1;
    if (token < 0 || token >= m->vocab) {
        fprintf(stderr, "qwen3: token %d out of range (vocab %d)\n", token, m->vocab);
        return -1;
    }
    if (pos < 0 || pos >= s->context) {
        fprintf(stderr, "qwen3: position %d outside the %d-position state\n",
                pos, s->context);
        return -1;
    }

    Gguf *g = &m->gguf;
    const int hd = m->head_dim, nh = m->n_heads, nkv = m->n_kv;
    const int rep = nh / nkv;                 /* GQA: query heads per KV head */
    const int kvw = nkv * hd;

    /* ---- embedding: one row, read on demand ---- */
    {
        const EngQuantOps *eq = eng_quant_ops(m->embd->dtype);
        if (eq) {
            const int64_t rb = eq->row_bytes(m->d_model);
            if (g->store->read(g->store, m->embd->file_off + (int64_t)token * rb,
                               rb, m->rowbuf) != rb) return -1;
            eq->dequant_row(m->x, m->rowbuf, NULL, m->d_model);
        } else {
            const int64_t rb = (int64_t)m->d_model * (int64_t)sizeof(float);
            if (g->store->read(g->store, m->embd->file_off + (int64_t)token * rb,
                               rb, m->x) != rb) return -1;
        }
    }
    emit(m, "embedding", m->x, m->d_model);

    /* ---- layers ---- */
    for (int L = 0; L < m->n_layers; L++) {
        unsigned char *base = eng_streamer_get(m->stream, L, NULL);
        if (!base) return -1;
        /* The walk order is fixed, so the next read can start now and overlap this
         * layer's compute. Never wrong, because there is nothing to predict. */
        eng_streamer_prefetch(m->stream, L + 1 < m->n_layers ? L + 1 : 0);

        const LayerSlots *sl = &m->slot[L];
        char nm[32];

        /* --- attention --- */
        eng_rmsnorm(m->xn, m->x, (const float *)(base + sl->attn_norm.rel),
                    m->d_model, m->eps);
        if (L == 0) emit(m, "l0.attn_norm", m->xn, m->d_model);

        if (project(m->q, m->xn, base, &sl->attn_q) != 0) return -1;
        if (project(m->k, m->xn, base, &sl->attn_k) != 0) return -1;
        if (project(m->v, m->xn, base, &sl->attn_v) != 0) return -1;
        if (L == 0) {
            emit(m, "l0.q_proj", m->q, nh * hd);
            emit(m, "l0.k_proj", m->k, kvw);
            emit(m, "l0.v_proj", m->v, kvw);
        }

        /* QK-norm, per head, before RoPE. */
        head_norm(m->q, nh,  hd, (const float *)(base + sl->attn_q_norm.rel), m->eps);
        head_norm(m->k, nkv, hd, (const float *)(base + sl->attn_k_norm.rel), m->eps);
        if (L == 0) {
            emit(m, "l0.q_normed", m->q, nh * hd);
            emit(m, "l0.k_normed", m->k, kvw);
        }

        for (int h = 0; h < nh;  h++)
            eng_rope(m->q + (size_t)h * hd, hd, pos, m->rope_theta, ENG_ROPE_HALVED);
        for (int h = 0; h < nkv; h++)
            eng_rope(m->k + (size_t)h * hd, hd, pos, m->rope_theta, ENG_ROPE_HALVED);
        if (L == 0) {
            emit(m, "l0.q_roped", m->q, nh * hd);
            emit(m, "l0.k_roped", m->k, kvw);
        }

        /* Append this position's K and V, then attend over 0..pos. The cache is
         * per-layer, so it is indexed by (layer, position). */
        {
            float *kdst = s->kc + ((size_t)L * s->context + pos) * kvw;
            float *vdst = s->vc + ((size_t)L * s->context + pos) * kvw;
            memcpy(kdst, m->k, (size_t)kvw * sizeof(float));
            memcpy(vdst, m->v, (size_t)kvw * sizeof(float));
        }

        const float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < nh; h++) {
            const int kvh = h / rep;          /* NOT h % nkv: 4 query heads share one */
            const float *qh = m->q + (size_t)h * hd;
            for (int t = 0; t <= pos; t++) {
                const float *kt = s->kc + ((size_t)L * s->context + t) * kvw
                                + (size_t)kvh * hd;
                m->scores[t] = eng_dot_f32(qh, kt, hd, ENG_NUM_FAST) * scale;
            }
            eng_softmax(m->scores, pos + 1);

            float *oh = m->att + (size_t)h * hd;
            memset(oh, 0, (size_t)hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float w = m->scores[t];
                const float *vt = s->vc + ((size_t)L * s->context + t) * kvw
                                + (size_t)kvh * hd;
                for (int d = 0; d < hd; d++) oh[d] += w * vt[d];
            }
        }
        if (L == 0) emit(m, "l0.attn_heads", m->att, nh * hd);

        if (project(m->ffn, m->att, base, &sl->attn_out) != 0) return -1;
        if (L == 0) emit(m, "l0.attn_out", m->ffn, m->d_model);
        for (int i = 0; i < m->d_model; i++) m->x[i] += m->ffn[i];

        /* --- MLP --- */
        eng_rmsnorm(m->xn, m->x, (const float *)(base + sl->ffn_norm.rel),
                    m->d_model, m->eps);
        if (project(m->gate, m->xn, base, &sl->ffn_gate) != 0) return -1;
        if (project(m->up,   m->xn, base, &sl->ffn_up)   != 0) return -1;
        for (int i = 0; i < m->d_ff; i++)
            m->gate[i] = (m->gate[i] / (1.0f + expf(-m->gate[i]))) * m->up[i];
        if (project(m->ffn, m->gate, base, &sl->ffn_down) != 0) return -1;
        if (L == 0) emit(m, "l0.mlp_out", m->ffn, m->d_model);
        for (int i = 0; i < m->d_model; i++) m->x[i] += m->ffn[i];

        snprintf(nm, sizeof nm, "layer%d", L);
        emit(m, nm, m->x, m->d_model);
    }

    /* ---- final norm and the LM head ---- */
    eng_rmsnorm(m->xn, m->x, m->out_norm, m->d_model, m->eps);
    emit(m, "final_norm", m->xn, m->d_model);

    {
        const EngQuantOps *hq = eng_quant_ops(m->lm_head->dtype);
        const int64_t hrow = hq ? hq->row_bytes(m->d_model)
                                : (int64_t)m->d_model * (int64_t)sizeof(float);
        for (int r0 = 0; r0 < m->vocab; r0 += m->head_chunk) {
            int n = m->head_chunk;
            if (r0 + n > m->vocab) n = m->vocab - r0;
            const int64_t want = (int64_t)n * hrow;
            if (g->store->read(g->store, m->lm_head->file_off + (int64_t)r0 * hrow,
                               want, m->headbuf) != want) return -1;
            if (hq) {
                if (eng_matmul_quant(m->logits + r0, m->xn, m->headbuf, NULL,
                                     m->d_model, n, m->lm_head->dtype) != 0) return -1;
            } else {
                eng_matmul_f32(m->logits + r0, m->xn, (const float *)m->headbuf,
                               m->d_model, n, ENG_NUM_FAST);
            }
        }
    }
    emit(m, "logits", m->logits, m->vocab);

    if (pos + 1 > s->n_seen) s->n_seen = pos + 1;
    return 0;
}

static const float *qwen3_logits(const EngModel *m, int *n_vocab)
{
    if (!m) return NULL;
    if (n_vocab) *n_vocab = m->vocab;
    return m->logits;
}

static void qwen3_caps(EngModelCaps *out)
{
    if (!out) return;
    memset(out, 0, sizeof *out);
    out->flags = ENG_MCAP_EXECUTE | ENG_MCAP_POSITIONAL | ENG_MCAP_STREAMABLE
               | ENG_MCAP_INCREMENTAL;
    /* Qwen3 makes no claim that its output is independent of the memory configuration,
     * so fp32 accumulation is appropriate. K3 declares EXACT for the opposite reason. */
    out->num_policy = ENG_NUM_FAST;
    out->notes = "GQA with QK-norm and RoPE(1e6), dense SwiGLU, untied embeddings";
}

const EngModelBackend eng_backend_qwen3 = {
    .name          = "qwen3",
    .description   = "Qwen3: grouped-query attention with QK-norm, RoPE, dense SwiGLU",
    .probe         = qwen3_probe,
    .inspect       = qwen3_inspect,
    .load          = qwen3_load,
    .destroy       = qwen3_destroy,
    .state_create  = qwen3_state_create,
    .state_destroy = qwen3_state_destroy,
    .state_bytes   = qwen3_state_bytes,
    .decode        = qwen3_decode,
    .logits        = qwen3_logits,
    .caps          = qwen3_caps
};
