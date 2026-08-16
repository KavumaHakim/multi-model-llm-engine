/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_tokenizer.c - byte-level BPE against llama.cpp's tokenizer.
 *
 * THE REFERENCE IS THE REFERENCE IMPLEMENTATION. tests/fixtures/gguf/tokenizer_golden.txt
 * holds token ids produced by llama.cpp's llama-tokenize on the same container. That is
 * a genuinely independent authority -- a different codebase, by different people,
 * against the same file -- rather than this engine checking itself.
 *
 * That distinction matters more for a tokenizer than almost anywhere else. A tokenizer
 * can be SELF-CONSISTENT and still wrong: if encode and decode share a misunderstanding
 * of the byte alphabet or the split pattern, every round-trip passes while every id
 * differs from what the model was trained on. The model then receives inputs it has
 * never seen and produces confident nonsense, with nothing in the pipeline complaining.
 *
 * The cases were chosen to hit each branch of the qwen2 pre-tokenizer and the places a
 * naive implementation diverges silently:
 *
 *   " hello" vs "hello"     the leading space belongs to the WORD, not to itself
 *   "  hello"               a run of spaces keeps all but the last
 *   "2024"                  digits split ONE AT A TIME, not as a run
 *   "don't"                 contractions are their own fragment
 *   "<|im_start|>"          special tokens match literally, never by merging
 *   CJK and accented Latin  multi-byte input through the byte alphabet
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bpe.h"
#include "gguf.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-44s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

/* Undo the fixture's escaping. Returns the decoded length. */
static int unescape(const char *s, int n, char *out, int cap)
{
    int w = 0;
    for (int i = 0; i < n && w < cap; i++) {
        if (s[i] != '\\' || i + 1 >= n) { out[w++] = s[i]; continue; }
        switch (s[++i]) {
            case 'n':  out[w++] = '\n'; break;
            case 'r':  out[w++] = '\r'; break;
            case 't':  out[w++] = '\t'; break;
            case '\\': out[w++] = '\\'; break;
            default:   out[w++] = '\\'; out[w++] = s[i]; break;
        }
    }
    return w;
}

static void show(const char *s, int n, char *buf, int cap)
{
    int w = 0;
    for (int i = 0; i < n && w < cap - 5; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\n')      { buf[w++] = '\\'; buf[w++] = 'n'; }
        else if (c == '\t') { buf[w++] = '\\'; buf[w++] = 't'; }
        else if (c < 0x20)  { w += snprintf(buf + w, (size_t)(cap - w), "\\x%02x", c); }
        else                { buf[w++] = (char)c; }
    }
    buf[w] = '\0';
}

int main(int argc, char **argv)
{
    const char *fixture = argc > 1 ? argv[1]
                                   : "tests/fixtures/gguf/tokenizer_golden.txt";
    const char *model = getenv("GGUF_MODEL");
    if (!model) model = "/mnt/c/Users/SHAMI/HAKIM/AI/Qwen3-8B-Q4_K_M.gguf";

    printf("byte-level BPE vs llama.cpp's tokenizer\n\n");

    FILE *probe = fopen(model, "rb");
    if (!probe) {
        printf("  SKIP  %s is not present; the tokenizer needs its vocabulary.\n", model);
        return 0;
    }
    fclose(probe);

    Gguf g;
    if (gguf_open(&g, model) != 0) { printf("  cannot open the model\n"); return 1; }

    EngBpe *b = eng_bpe_from_gguf(&g);
    ok(b != NULL, "tokenizer loads from the container", NULL);
    if (!b) { gguf_close(&g); return 1; }

    char d[256];
    snprintf(d, sizeof d, "%d tokens", eng_bpe_vocab_size(b));
    ok(eng_bpe_vocab_size(b) == 151936, "vocabulary size", d);
    snprintf(d, sizeof d, "bos %d eos %d add_bos %d",
             eng_bpe_bos(b), eng_bpe_eos(b), eng_bpe_add_bos(b));
    ok(eng_bpe_eos(b) == 151645 && eng_bpe_bos(b) == 151643, "special ids", d);
    ok(eng_bpe_add_bos(b) == 0, "add_bos is off for this model",
       "prepending BOS would shift every position");

    ok(eng_bpe_is_special(b, 151644), "im_start is a control token", NULL);
    ok(!eng_bpe_is_special(b, 9707),  "an ordinary token is not", NULL);

    /* ---- against the reference ---- */
    printf("\n== encode, against llama.cpp ==\n");
    FILE *f = fopen(fixture, "rb");
    if (!f) {
        ok(0, "fixture present", fixture);
        eng_bpe_free(b);
        gguf_close(&g);
        return 1;
    }

    char line[4096];
    int ncase = 0, nbad = 0, nrt = 0, nrt_bad = 0;
    while (fgets(line, sizeof line, f)) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';

        char text[2048];
        const int tn = unescape(line, (int)strlen(line), text, sizeof text);

        int want[512], nwant = 0;
        for (char *p = tab + 1; *p && nwant < 512; ) {
            while (*p == ' ' || *p == ',') p++;
            if (!*p) break;
            want[nwant++] = (int)strtol(p, &p, 10);
        }

        int got[512];
        const int ngot = eng_bpe_encode(b, text, tn, got, 512, 1);

        int same = (ngot == nwant);
        for (int i = 0; same && i < ngot; i++) same = (got[i] == want[i]);

        ncase++;
        if (!same) {
            nbad++;
            char pretty[128];
            show(text, tn, pretty, sizeof pretty);
            printf("  FAIL  %-30s\n", pretty);
            printf("        want:");
            for (int i = 0; i < nwant; i++) printf(" %d", want[i]);
            printf("\n        got :");
            for (int i = 0; i < ngot; i++) printf(" %d", got[i]);
            printf("\n");
        }

        /* Decode what the REFERENCE said, and check it reproduces the input. This
         * checks decode independently of encode: if both shared a byte-alphabet
         * mistake, encode-then-decode would still round-trip. */
        if (nwant > 0) {
            char back[2048];
            const int bn = eng_bpe_decode(b, want, nwant, back, sizeof back);
            nrt++;
            if (bn != tn || memcmp(back, text, (size_t)tn)) {
                nrt_bad++;
                char p1[128], p2[128];
                show(text, tn, p1, sizeof p1);
                show(back, bn > 0 ? bn : 0, p2, sizeof p2);
                printf("  FAIL  decode: %s -> %s\n", p1, p2);
            }
        }
    }
    fclose(f);

    snprintf(d, sizeof d, "%d of %d cases differ", nbad, ncase);
    ok(ncase > 0 && nbad == 0, "encode matches llama.cpp exactly", d);
    snprintf(d, sizeof d, "%d of %d cases differ", nrt_bad, nrt);
    ok(nrt > 0 && nrt_bad == 0, "decode of reference ids reproduces the input", d);

    /* ---- idempotence ---- */
    printf("\n== idempotence ==\n");
    {
        /* Encoding text that came out of a decode must be stable. This is weaker than
         * matching the reference, but it covers inputs the fixture does not. */
        static const char *T[] = {
            "The engine streams weights from disk.",
            "a b c d e f g",
            "((nested) [brackets] {here})",
            "1234567890",
            "MiXeD cAsE wOrDs"
        };
        int bad = 0;
        for (size_t i = 0; i < sizeof T / sizeof *T; i++) {
            int a[256], c2[256];
            const int na = eng_bpe_encode(b, T[i], -1, a, 256, 0);
            char back[1024];
            const int bn = eng_bpe_decode(b, a, na, back, sizeof back);
            const int nc = eng_bpe_encode(b, back, bn, c2, 256, 0);
            if (nc != na || memcmp(a, c2, (size_t)na * sizeof(int))) bad++;
        }
        snprintf(d, sizeof d, "%d of %zu unstable", bad, sizeof T / sizeof *T);
        ok(bad == 0, "encode(decode(encode(x))) == encode(x)", d);
    }

    /* ---- special-token handling is a policy, not a guess ---- */
    printf("\n== special tokens ==\n");
    {
        const char *s = "<|im_start|>";
        int on[16], off[16];
        const int non  = eng_bpe_encode(b, s, -1, on,  16, 1);
        const int noff = eng_bpe_encode(b, s, -1, off, 16, 0);
        snprintf(d, sizeof d, "allowed -> %d token(s), literal -> %d", non, noff);
        ok(non == 1 && on[0] == 151644, "matched whole when allowed", d);
        ok(noff > 1, "encoded as plain text when not allowed",
           "so untrusted input cannot inject control tokens");
    }

    eng_bpe_free(b);
    gguf_close(&g);

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("TOKENIZER TESTS FAILED\n"); return 1; }
    printf("TOKENIZER TESTS PASSED\n");
    return 0;
}
