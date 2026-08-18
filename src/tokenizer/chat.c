/* SPDX-License-Identifier: Apache-2.0 */
/* chat.c - see chat.h. */
#include "chat.h"

#include <stdio.h>
#include <string.h>

/* Substring search over a buffer that is NOT NUL-terminated: gguf_str hands back a
 * pointer into the mapped metadata and a length, so strstr would run off the end. */
static int contains(const char *hay, size_t hn, const char *needle)
{
    const size_t nn = strlen(needle);
    if (nn == 0 || hn < nn) return 0;
    for (size_t i = 0; i + nn <= hn; i++)
        if (!memcmp(hay + i, needle, nn)) return 1;
    return 0;
}

const char *eng_chat_family_name(EngChatFamily f)
{
    switch (f) {
        case ENG_CHAT_CHATML: return "chatml";
        case ENG_CHAT_NONE:   return "none";
        default:              return "?";
    }
}

int eng_chat_detect(EngChat *c, const Gguf *g, const EngBpe *b)
{
    if (!c || !g || !b) return -1;
    memset(c, 0, sizeof *c);
    c->family = ENG_CHAT_NONE;
    c->im_start = c->im_end = -1;
    c->eos = eng_bpe_eos(b);

    const char *tpl = NULL;
    int64_t n = 0;
    if (gguf_str(g, "tokenizer.chat_template", &tpl, &n) != 0 || !tpl || n <= 0) {
        /* A base model legitimately has none. Not an error. */
        return 0;
    }

    /* Family by marker. Cheap, and the alternative -- interpreting Jinja -- is a
     * different project. */
    const int chatml = contains(tpl, (size_t)n, "<|im_start|>");

    if (chatml) {
        /* The tokens must actually exist. A template naming markers the vocabulary does
         * not contain would produce a prompt full of unmatched literal text. */
        const int s = eng_bpe_find(b, "<|im_start|>", 12);
        const int e = eng_bpe_find(b, "<|im_end|>", 10);
        if (s >= 0 && e >= 0) {
            c->family = ENG_CHAT_CHATML;
            c->im_start = s;
            c->im_end = e;
            c->supports_think = contains(tpl, (size_t)n, "enable_thinking");
            return 0;
        }
        fprintf(stderr,
                "chat: the template is ChatML but the vocabulary has no %s token; "
                "falling back to a plain continuation\n", s < 0 ? "<|im_start|>" : "<|im_end|>");
        return 0;
    }

    /* Say so rather than quietly producing base-model continuations: "the answers look
     * odd" is a slow way to find out the prompt was never formatted. */
    fprintf(stderr,
            "chat: this model carries a chat template this engine does not implement "
            "(%lld chars, no ChatML markers). Prompts will be treated as plain "
            "continuations.\n", (long long)n);
    return 0;
}

/* Append helpers. Each returns 0 on success, -1 when the buffer is full -- and the
 * caller aborts on the first failure rather than emitting a truncated turn structure. */
static int put_id(int *out, int cap, int *n, int id)
{
    if (*n >= cap) return -1;
    out[(*n)++] = id;
    return 0;
}

/* Encode text as ORDINARY tokens: allow_special is 0, so a user typing "<|im_end|>"
 * gets the literal characters rather than the control token, and cannot forge a turn
 * boundary. See chat.h. */
static int put_text(const EngBpe *b, const char *s, int *out, int cap, int *n)
{
    if (!s || !*s) return 0;
    const int got = eng_bpe_encode(b, s, (int)strlen(s), out + *n, cap - *n, 0);
    if (got < 0) return -1;
    *n += got;
    return 0;
}

int eng_chat_build(const EngChat *c, const EngBpe *b,
                   const char *system, const char *user,
                   int think, int *out, int cap)
{
    if (!c || !b || !out || cap <= 0) return -1;
    if (c->family != ENG_CHAT_CHATML) return -1;

    int n = 0;
    /* <|im_start|>system\n{system}<|im_end|>\n */
    if (system && *system) {
        if (put_id(out, cap, &n, c->im_start)) return -1;
        if (put_text(b, "system\n", out, cap, &n)) return -1;
        if (put_text(b, system, out, cap, &n)) return -1;
        if (put_id(out, cap, &n, c->im_end)) return -1;
        if (put_text(b, "\n", out, cap, &n)) return -1;
    }

    /* <|im_start|>user\n{user}<|im_end|>\n */
    if (put_id(out, cap, &n, c->im_start)) return -1;
    if (put_text(b, "user\n", out, cap, &n)) return -1;
    if (put_text(b, user ? user : "", out, cap, &n)) return -1;
    if (put_id(out, cap, &n, c->im_end)) return -1;
    if (put_text(b, "\n", out, cap, &n)) return -1;

    /* <|im_start|>assistant\n , then optionally the empty think block that suppresses
     * reasoning. Both come straight from the template's add_generation_prompt branch. */
    if (put_id(out, cap, &n, c->im_start)) return -1;
    if (put_text(b, "assistant\n", out, cap, &n)) return -1;
    if (!think && c->supports_think) {
        if (put_text(b, "<think>\n\n</think>\n\n", out, cap, &n)) return -1;
    }

    return n;
}
