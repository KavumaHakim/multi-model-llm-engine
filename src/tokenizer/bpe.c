/* SPDX-License-Identifier: Apache-2.0 */
/* bpe.c - see bpe.h. */
#define _POSIX_C_SOURCE 200809L

#include "bpe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ hashing -- */

typedef struct {
    const char *s;
    int         len;
    int         val;
} Ent;

typedef struct {
    Ent  *e;
    int   cap;      /* power of two */
    int   n;
} Map;

static uint64_t hash_bytes(const char *s, int n)
{
    uint64_t h = 1469598103934665603ULL;      /* FNV-1a */
    for (int i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int map_init(Map *m, int want)
{
    int cap = 16;
    while (cap < want * 2) cap <<= 1;
    m->e = (Ent *)calloc((size_t)cap, sizeof *m->e);
    m->cap = cap;
    m->n = 0;
    return m->e ? 0 : -1;
}

static void map_put(Map *m, const char *s, int len, int val)
{
    uint64_t h = hash_bytes(s, len) & (uint64_t)(m->cap - 1);
    while (m->e[h].s) {
        if (m->e[h].len == len && !memcmp(m->e[h].s, s, (size_t)len)) return; /* first wins */
        h = (h + 1) & (uint64_t)(m->cap - 1);
    }
    m->e[h].s = s;
    m->e[h].len = len;
    m->e[h].val = val;
    m->n++;
}

static int map_get(const Map *m, const char *s, int len)
{
    if (!m->e) return -1;
    uint64_t h = hash_bytes(s, len) & (uint64_t)(m->cap - 1);
    while (m->e[h].s) {
        if (m->e[h].len == len && !memcmp(m->e[h].s, s, (size_t)len)) return m->e[h].val;
        h = (h + 1) & (uint64_t)(m->cap - 1);
    }
    return -1;
}

/* -------------------------------------------------------------------- state -- */

struct EngBpe {
    int    n_vocab;
    char **tok;         /* stored text, NUL-terminated, in the byte-mapped alphabet */
    int   *tok_len;
    int   *tok_type;
    char  *pool;        /* one allocation backing every token and merge string */

    Map    vocab;       /* stored text -> id   */
    Map    merge;       /* "a b" joined -> rank */

    int    bos, eos, add_bos;

    /* Special tokens, gathered once. `spec_first` is a bitmap of their leading bytes,
     * so scanning input for a possible special is one array lookup per byte instead of
     * a walk over 151,936 entries. */
    int   *spec;
    int    n_spec;
    unsigned char spec_first[256];

    /* byte value -> its mapped codepoint's UTF-8 bytes, and the inverse */
    char   b2u[256][4];
    int    b2u_len[256];
    int    u2b[0x200];  /* codepoint -> byte, for the codepoints the map uses */
};

/* The GPT-2 byte<->unicode alphabet. Printable ASCII and Latin-1 map to themselves; the
 * other 68 byte values are assigned U+0100 upward in order. Built rather than tabulated
 * so the construction is visible and checkable. */
static void build_byte_map(EngBpe *b)
{
    int used[256];
    memset(used, 0, sizeof used);
    int cps[256];

    for (int c = '!'; c <= '~'; c++)      { cps[c] = c; used[c] = 1; }
    for (int c = 0xA1; c <= 0xAC; c++)    { cps[c] = c; used[c] = 1; }
    for (int c = 0xAE; c <= 0xFF; c++)    { cps[c] = c; used[c] = 1; }

    int n = 0;
    for (int c = 0; c < 256; c++)
        if (!used[c]) cps[c] = 0x100 + n++;

    memset(b->u2b, -1, sizeof b->u2b);
    for (int c = 0; c < 256; c++) {
        const int cp = cps[c];
        /* Encode the codepoint as UTF-8. Everything here is below U+0800, so one or two
         * bytes. */
        if (cp < 0x80) {
            b->b2u[c][0] = (char)cp;
            b->b2u_len[c] = 1;
        } else {
            b->b2u[c][0] = (char)(0xC0 | (cp >> 6));
            b->b2u[c][1] = (char)(0x80 | (cp & 0x3F));
            b->b2u_len[c] = 2;
        }
        if (cp < 0x200) b->u2b[cp] = c;
    }
}

/* ------------------------------------------------------------------- loading -- */

EngBpe *eng_bpe_from_gguf(const Gguf *g)
{
    if (!g) return NULL;

    const GgufKV *ktok = gguf_find(g, "tokenizer.ggml.tokens");
    const GgufKV *kmrg = gguf_find(g, "tokenizer.ggml.merges");
    const GgufKV *ktyp = gguf_find(g, "tokenizer.ggml.token_type");
    if (!ktok || ktok->type != GGUF_T_ARRAY || ktok->arr_type != GGUF_T_STRING) {
        fprintf(stderr, "bpe: no tokenizer.ggml.tokens in this container\n");
        return NULL;
    }

    EngBpe *b = (EngBpe *)calloc(1, sizeof *b);
    if (!b) return NULL;
    build_byte_map(b);

    b->n_vocab = (int)ktok->arr_len;
    b->tok     = (char **)calloc((size_t)b->n_vocab, sizeof *b->tok);
    b->tok_len = (int *)calloc((size_t)b->n_vocab, sizeof *b->tok_len);
    b->tok_type= (int *)calloc((size_t)b->n_vocab, sizeof *b->tok_type);
    if (!b->tok || !b->tok_len || !b->tok_type) { eng_bpe_free(b); return NULL; }

    /* One pool for every token and merge string. Sized from the arrays' own extent in
     * the header, which is an upper bound, and never reallocated: pointers into it are
     * handed out as they are made. */
    const int64_t pool_cap = g->blob_size + (int64_t)b->n_vocab + 8;
    b->pool = (char *)malloc((size_t)pool_cap);
    if (!b->pool) { eng_bpe_free(b); return NULL; }
    int64_t used = 0;

    if (map_init(&b->vocab, b->n_vocab) != 0) { eng_bpe_free(b); return NULL; }

    for (int i = 0; i < b->n_vocab; i++) {
        int64_t len = 0;
        const char *s = gguf_arr_str(g, ktok, i, &len);
        if (!s) { fprintf(stderr, "bpe: token %d is unreadable\n", i); eng_bpe_free(b); return NULL; }
        if (used + len + 1 > pool_cap) { fprintf(stderr, "bpe: string pool exhausted\n");
                                         eng_bpe_free(b); return NULL; }
        char *d = b->pool + used;
        memcpy(d, s, (size_t)len);
        d[len] = '\0';
        used += len + 1;
        b->tok[i] = d;
        b->tok_len[i] = (int)len;
        map_put(&b->vocab, d, (int)len, i);
    }

    if (ktyp && ktyp->type == GGUF_T_ARRAY && ktyp->arr_len == ktok->arr_len) {
        /* token_type is an int32 array, stored contiguously. 1 = normal, 3 = control. */
        const unsigned char *p = ktyp->arr_data;
        for (int i = 0; i < b->n_vocab; i++) {
            uint32_t v = 0;
            for (int k = 0; k < 4; k++) v |= (uint32_t)p[i * 4 + k] << (8 * k);
            b->tok_type[i] = (int)v;
        }
    }

    /* Merges: "a b", ranked by position. The rank IS the priority, so the table maps the
     * concatenation "ab" to its rank and the merge loop takes the lowest. */
    if (kmrg && kmrg->type == GGUF_T_ARRAY && kmrg->arr_type == GGUF_T_STRING) {
        if (map_init(&b->merge, (int)kmrg->arr_len) != 0) { eng_bpe_free(b); return NULL; }
        for (int64_t i = 0; i < kmrg->arr_len; i++) {
            int64_t len = 0;
            const char *s = gguf_arr_str(g, kmrg, i, &len);
            if (!s) continue;
            const char *sp = (const char *)memchr(s, ' ', (size_t)len);
            if (!sp) continue;
            const int64_t la = sp - s, lb = len - la - 1;
            if (used + la + lb + 1 > pool_cap) break;
            char *d = b->pool + used;
            memcpy(d, s, (size_t)la);
            memcpy(d + la, sp + 1, (size_t)lb);
            d[la + lb] = '\0';
            used += la + lb + 1;
            map_put(&b->merge, d, (int)(la + lb), (int)i);
        }
    } else {
        fprintf(stderr, "bpe: no merges; encoding will fall back to single bytes\n");
    }

    /* Index the special tokens. */
    b->spec = (int *)malloc((size_t)b->n_vocab * sizeof *b->spec);
    if (!b->spec) { eng_bpe_free(b); return NULL; }
    for (int i = 0; i < b->n_vocab; i++) {
        if (!eng_bpe_is_special(b, i) || b->tok_len[i] == 0) continue;
        b->spec[b->n_spec++] = i;
        b->spec_first[(unsigned char)b->tok[i][0]] = 1;
    }

    int64_t v = 0;
    b->bos = gguf_i64(g, "tokenizer.ggml.bos_token_id", &v) == 0 ? (int)v : -1;
    b->eos = gguf_i64(g, "tokenizer.ggml.eos_token_id", &v) == 0 ? (int)v : -1;
    b->add_bos = gguf_i64(g, "tokenizer.ggml.add_bos_token", &v) == 0 ? (int)v : 0;
    return b;
}

void eng_bpe_free(EngBpe *b)
{
    if (!b) return;
    free(b->tok); free(b->tok_len); free(b->tok_type); free(b->pool);
    free(b->vocab.e); free(b->merge.e); free(b->spec);
    free(b);
}

int eng_bpe_vocab_size(const EngBpe *b) { return b ? b->n_vocab : 0; }
int eng_bpe_bos(const EngBpe *b)        { return b ? b->bos : -1; }
int eng_bpe_eos(const EngBpe *b)        { return b ? b->eos : -1; }
int eng_bpe_add_bos(const EngBpe *b)    { return b ? b->add_bos : 0; }

const char *eng_bpe_token_text(const EngBpe *b, int id, int *len)
{
    if (!b || id < 0 || id >= b->n_vocab) return NULL;
    if (len) *len = b->tok_len[id];
    return b->tok[id];
}

int eng_bpe_is_special(const EngBpe *b, int id)
{
    if (!b || id < 0 || id >= b->n_vocab) return 0;
    return b->tok_type[id] == 3 || b->tok_type[id] == 4;   /* CONTROL / USER_DEFINED */
}

int eng_bpe_find(const EngBpe *b, const char *text, int len)
{
    return b ? map_get(&b->vocab, text, len) : -1;
}

/* ------------------------------------------------------------------ decoding -- */

int eng_bpe_decode_one(const EngBpe *b, int id, char *out, int cap)
{
    if (!b || id < 0 || id >= b->n_vocab) return -1;
    const char *s = b->tok[id];
    const int n = b->tok_len[id];
    int w = 0;

    /* Walk the stored text as UTF-8 and invert the byte map codepoint by codepoint. */
    for (int i = 0; i < n; ) {
        const unsigned char c = (unsigned char)s[i];
        int cp, adv;
        if (c < 0x80)              { cp = c;                                   adv = 1; }
        else if ((c & 0xE0) == 0xC0 && i + 1 < n)
                                   { cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F); adv = 2; }
        else if ((c & 0xF0) == 0xE0 && i + 2 < n)
                                   { cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6)
                                        | (s[i+2] & 0x3F);                     adv = 3; }
        else                       { cp = c;                                   adv = 1; }
        i += adv;

        const int byte = (cp >= 0 && cp < 0x200) ? b->u2b[cp] : -1;
        if (byte >= 0) {
            if (w + 1 > cap) return -1;
            out[w++] = (char)byte;
        } else {
            /* Not part of the byte alphabet: a special token like <|im_start|>, whose
             * text is literal UTF-8. Copy it through unchanged. */
            if (w + adv > cap) return -1;
            memcpy(out + w, s + i - adv, (size_t)adv);
            w += adv;
        }
    }
    return w;
}

int eng_bpe_decode(const EngBpe *b, const int *ids, int n, char *out, int cap)
{
    if (!b || !ids || !out) return -1;
    int w = 0;
    for (int i = 0; i < n; i++) {
        const int k = eng_bpe_decode_one(b, ids[i], out + w, cap - w);
        if (k < 0) return -1;
        w += k;
    }
    if (w < cap) out[w] = '\0';
    return w;
}

/* ------------------------------------------------------- pre-tokenisation -- */

/* ASCII classes. See bpe.h for the documented treatment of bytes >= 0x80. */
static int is_letter(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}
static int is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int is_space(unsigned char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

int eng_bpe_ascii_exact(const char *text, int len)
{
    /* The only inexactness is non-ASCII DIGITS, which this classifies as letters. There
     * is no cheap way to detect them without the Unicode tables, so this reports
     * conservatively: pure ASCII is certainly exact, anything else may not be. */
    for (int i = 0; i < len; i++)
        if ((unsigned char)text[i] >= 0x80) return 0;
    return 1;
}

/* One fragment of the qwen2 split, as [start, end). The pattern, in order:
 *     contractions | ?letters | digits | ?punctuation | newlines | trailing space | space
 * Returns the end offset of the fragment beginning at `i`. */
static int next_fragment(const char *s, int n, int i)
{
    const unsigned char c = (unsigned char)s[i];

    /* Contractions: 's 't 're 've 'm 'll 'd, case-insensitive. Kept whole because the
     * vocabulary has single tokens for them. */
    if (c == '\'' && i + 1 < n) {
        static const char *C[] = { "s", "t", "m", "d", "re", "ve", "ll" };
        for (size_t k = 0; k < sizeof C / sizeof *C; k++) {
            const int l = (int)strlen(C[k]);
            if (i + 1 + l > n) continue;
            int ok = 1;
            for (int j = 0; j < l; j++) {
                const char a = s[i + 1 + j] | 0x20;
                if (a != C[k][j]) { ok = 0; break; }
            }
            if (ok) return i + 1 + l;
        }
    }

    /* An optional single leading non-letter/non-digit, then a run of letters. This is
     * what attaches the leading space to a word: " the" is one fragment, and the
     * vocabulary stores it as "Ġthe". */
    {
        int j = i;
        if (!is_letter(c) && !is_digit(c) && c != '\r' && c != '\n') j++;
        if (j < n && is_letter((unsigned char)s[j])) {
            while (j < n && is_letter((unsigned char)s[j])) j++;
            return j;
        }
    }

    /* A single digit. The qwen2 pattern matches \p{N} one at a time, so "2024" becomes
     * four fragments rather than one -- and merges then rebuild whatever the vocabulary
     * actually contains. Grouping them here would produce different ids. */
    if (is_digit(c)) return i + 1;

    /* An optional leading space, then a run of punctuation, then any trailing newlines. */
    if (!is_space(c) || (c == ' ' && i + 1 < n && !is_space((unsigned char)s[i + 1]))) {
        int j = i;
        if (c == ' ') j++;
        if (j < n && !is_space((unsigned char)s[j]) &&
            !is_letter((unsigned char)s[j]) && !is_digit((unsigned char)s[j])) {
            while (j < n && !is_space((unsigned char)s[j]) &&
                   !is_letter((unsigned char)s[j]) && !is_digit((unsigned char)s[j])) j++;
            while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
            return j;
        }
    }

    /* Whitespace runs. A run ending in a newline is kept together; a run followed by a
     * non-space keeps its last space for the following word, which is why the letter
     * rule above takes an optional leading non-letter. */
    if (is_space(c)) {
        int j = i;
        while (j < n && is_space((unsigned char)s[j]) && s[j] != '\r' && s[j] != '\n') j++;
        if (j < n && (s[j] == '\r' || s[j] == '\n')) {
            while (j < n && (s[j] == '\r' || s[j] == '\n')) j++;
            return j;
        }
        /* \s+(?!\S): keep all but the last space, so the last one joins the next word. */
        if (j < n && j > i + 1) return j - 1;
        return j > i ? j : i + 1;
    }

    return i + 1;
}

/* --------------------------------------------------------------- encoding -- */

/* Merge one fragment, already byte-mapped into `sym` as `nsym` pieces. Repeatedly joins
 * the adjacent pair with the LOWEST rank, which is what makes BPE deterministic. */
typedef struct { int off, len; } Sym;

static int merge_fragment(const EngBpe *b, char *buf, Sym *sym, int nsym,
                          int *out, int cap, int nout)
{
    for (;;) {
        int best = -1, best_rank = 0;
        for (int i = 0; i + 1 < nsym; i++) {
            const int la = sym[i].len, lb = sym[i + 1].len;
            /* The pair's text is contiguous in buf, because merging only ever joins
             * neighbours and never moves bytes. */
            const int r = map_get(&b->merge, buf + sym[i].off, la + lb);
            if (r < 0) continue;
            if (best < 0 || r < best_rank) { best = i; best_rank = r; }
        }
        if (best < 0) break;
        sym[best].len += sym[best + 1].len;
        memmove(sym + best + 1, sym + best + 2, (size_t)(nsym - best - 2) * sizeof *sym);
        nsym--;
    }

    for (int i = 0; i < nsym; i++) {
        const int id = map_get(&b->vocab, buf + sym[i].off, sym[i].len);
        if (nout >= cap) return -1;
        if (id >= 0) {
            out[nout++] = id;
        } else {
            /* Unreachable with a well-formed vocabulary, which contains every single
             * mapped byte. Emitting nothing would silently drop input, so say so. */
            fprintf(stderr, "bpe: no token for a %d-byte piece; input dropped\n",
                    sym[i].len);
            return -1;
        }
    }
    return nout;
}

int eng_bpe_encode(const EngBpe *b, const char *text, int len,
                   int *out, int cap, int allow_special)
{
    if (!b || !text || !out) return -1;
    if (len < 0) len = (int)strlen(text);

    int nout = 0;
    int i = 0;
    while (i < len) {
        /* SPLIT ON SPECIAL TOKENS FIRST, then pre-tokenize the text between them.
         *
         * Checking for a special only at the current position is not enough, and the
         * failure is silent: in "helpful.<|im_end|>" the pre-tokenizer groups the '.'
         * and the '<' into one punctuation run, swallowing the special token, which
         * then encodes as six ordinary tokens. The model sees no end-of-turn marker and
         * the ids diverge from every other implementation's from that point on.
         *
         * So find where the next special STARTS, encode everything before it as
         * ordinary text, then emit it. Longest match wins, since some vocabularies
         * contain a token that is a prefix of another. */
        int sp_at = -1, sp_id = -1, sp_len = 0;
        if (allow_special && b->n_spec > 0) {
            for (int j = i; j < len && sp_at < 0; j++) {
                if (!b->spec_first[(unsigned char)text[j]]) continue;
                for (int t = 0; t < b->n_spec; t++) {
                    const int id = b->spec[t];
                    const int tl = b->tok_len[id];
                    if (tl <= sp_len || j + tl > len) continue;
                    if (!memcmp(text + j, b->tok[id], (size_t)tl)) {
                        sp_at = j; sp_id = id; sp_len = tl;
                    }
                }
            }
        }

        /* The chunk of ordinary text before the next special (or the rest of input). */
        const int chunk_end = sp_at >= 0 ? sp_at : len;
        if (i >= chunk_end) {
            if (nout >= cap) return -1;
            out[nout++] = sp_id;
            i = sp_at + sp_len;
            continue;
        }

        const int end = next_fragment(text, chunk_end, i);
        const int flen = end - i;

        /* Byte-map the fragment: each input byte becomes 1-2 bytes of mapped text. */
        char  fbuf[1024 * 4];
        Sym   sym[1024];
        if (flen > (int)(sizeof sym / sizeof *sym)) {
            fprintf(stderr, "bpe: fragment of %d bytes exceeds the limit\n", flen);
            return -1;
        }
        int w = 0, ns = 0;
        for (int k = 0; k < flen; k++) {
            const unsigned char c = (unsigned char)text[i + k];
            const int bl = b->b2u_len[c];
            if (w + bl > (int)sizeof fbuf) return -1;
            memcpy(fbuf + w, b->b2u[c], (size_t)bl);
            sym[ns].off = w;
            sym[ns].len = bl;
            ns++;
            w += bl;
        }

        nout = merge_fragment(b, fbuf, sym, ns, out, cap, nout);
        if (nout < 0) return -1;
        i = end;
    }
    return nout;
}
