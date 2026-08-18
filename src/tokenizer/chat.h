/* SPDX-License-Identifier: Apache-2.0 */
/*
 * chat.h - turning messages into the token sequence an instruct model expects.
 *
 * WHY THIS IS NEEDED AT ALL. Without it the engine produces base-model continuations:
 * given "The capital of France is" it writes "Paris. The capital of the", and given a
 * question it continues the question rather than answering. That is not a bug in the
 * forward pass -- it is what a decoder does when the prompt is not wrapped in the turn
 * structure the model was tuned on.
 *
 * BUILT AS TOKEN IDS, NOT AS A STRING, and this is the part that matters.
 *
 *   The obvious implementation formats "<|im_start|>user\n...<|im_end|>\n" into a buffer
 *   and tokenises it. That is wrong in two ways:
 *
 *   1. Those markers are SINGLE CONTROL TOKENS in the vocabulary, not text. Passed
 *      through byte-level BPE they merge into ordinary pieces, and the model never sees
 *      the boundary it was trained on. Output looks plausible and the turn structure is
 *      simply absent.
 *
 *   2. It lets a user inject turns. Someone typing "<|im_end|>" into their message would
 *      close the turn and open a new one of their choosing. Here the markers are added as
 *      ids by this module, and message CONTENT is encoded with special-token matching
 *      OFF, so the same text tokenises to ordinary pieces and cannot forge a boundary.
 *
 * THE TEMPLATE IS READ FROM THE CONTAINER, not assumed. Qwen3-8B-Q4_K_M carries a 4,100
 * character Jinja template under tokenizer.chat_template. Interpreting Jinja is far out
 * of scope, so this recognises the FAMILY the template belongs to and implements that
 * family directly. An unrecognised template is reported rather than silently falling
 * back to a base continuation, because "the answer looks odd" is a very slow way to
 * discover the prompt was never formatted.
 *
 * THINKING IS OFF BY DEFAULT, and that is a decision this file is the right place to
 * explain. Qwen3's template ends:
 *
 *     {%- if add_generation_prompt %}
 *         {{- '<|im_start|>assistant\n' }}
 *         {%- if enable_thinking is defined and enable_thinking is false %}
 *             {{- '<think>\n\n</think>\n\n' }}
 *         {%- endif %}
 *     {%- endif %}
 *
 * So an empty think block SUPPRESSES reasoning; omitting it invites the model to reason
 * first. On a machine where this engine produces a token every 30 seconds, a few hundred
 * tokens of reasoning before the answer starts is not a trade worth making silently.
 * eng_chat_build takes it as a flag so the caller can choose, and the CLI exposes
 * --think.
 */
#ifndef ENG_CHAT_H
#define ENG_CHAT_H

#include "bpe.h"
#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ENG_CHAT_NONE = 0,   /* no template, or one this engine does not implement */
    ENG_CHAT_CHATML      /* <|im_start|>role\ncontent<|im_end|>\n */
} EngChatFamily;

typedef struct {
    EngChatFamily family;
    int im_start;        /* token id, -1 when absent */
    int im_end;
    int eos;
    int supports_think;  /* the template has an enable_thinking branch */
} EngChat;

/* Work out which family this model's template belongs to, and resolve the control
 * tokens it needs. Returns 0 even when the family is NONE -- that is a finding, not an
 * error -- so callers should test `family` rather than the return value.
 *
 * Requires BOTH the template marker and the control tokens to resolve. A template that
 * mentions ChatML while the vocabulary lacks the tokens would otherwise produce a prompt
 * full of unmatched text. */
int eng_chat_detect(EngChat *c, const Gguf *g, const EngBpe *b);

/* Build a single-turn prompt. `system` may be NULL.
 *
 * Returns the token count, or -1 if it does not fit in `cap` or the family is NONE --
 * refusing rather than emitting a partial prompt, since a truncated turn structure is
 * worse than none.
 *
 * `think` non-zero lets the model reason before answering; zero emits the empty
 * <think></think> block that suppresses it. See the header. */
int eng_chat_build(const EngChat *c, const EngBpe *b,
                   const char *system, const char *user,
                   int think, int *out, int cap);

const char *eng_chat_family_name(EngChatFamily f);

#ifdef __cplusplus
}
#endif

#endif /* ENG_CHAT_H */
