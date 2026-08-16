/* SPDX-License-Identifier: Apache-2.0 */
/*
 * bpe.h - byte-level BPE, loaded from a GGUF container.
 *
 * WHAT "BYTE-LEVEL" MEANS, because it is the part that surprises people
 *   GPT-2 style BPE does not operate on characters. It maps each of the 256 byte values
 *   to a printable Unicode codepoint first, runs BPE over THAT, and the vocabulary is
 *   stored in the mapped alphabet. So the token whose stored text is "Ġthe" is the byte
 *   sequence " the": U+0120 stands for byte 0x20. Any implementation that treats the
 *   vocabulary strings as literal UTF-8 will fail on every token containing a space.
 *
 *   The mapping is: bytes that are already printable ASCII and Latin-1 keep their value;
 *   the remaining 68 are shifted to U+0100 and up, in order. eng_bpe_byte_decoder()
 *   below inverts it.
 *
 * THREE STAGES, in this order
 *   1. PRE-TOKENIZE  split the input into fragments that BPE is never allowed to merge
 *                    across. Qwen3 declares tokenizer.ggml.pre = "qwen2", whose pattern
 *                    keeps contractions, letter runs, digit runs, punctuation runs and
 *                    whitespace runs apart.
 *   2. BYTE MAP      each fragment's bytes become codepoints as above.
 *   3. MERGE         repeatedly join the adjacent pair with the lowest merge rank,
 *                    until no pair is in the merge table.
 *
 *   Merging across fragment boundaries is the classic bug: it produces tokens that
 *   exist in the vocabulary and decode back to the same text, so a round-trip test
 *   passes while the ids differ from every other implementation's.
 *
 * A DOCUMENTED LIMITATION, stated rather than buried
 *   The qwen2 pre-tokenizer pattern uses the Unicode properties \p{L} and \p{N}.
 *   Implementing those exactly needs the full Unicode category tables. This
 *   implementation classifies ASCII exactly and treats every byte >= 0x80 as a letter,
 *   which is correct for CJK, Cyrillic, Greek and accented Latin -- the bulk of real
 *   input -- but wrong for non-ASCII DIGITS (Arabic-Indic, Devanagari and similar),
 *   which it will group with adjacent letters instead of splitting.
 *
 *   The effect is a different, still valid, still round-trippable tokenisation of text
 *   containing those digits. eng_bpe_ascii_exact() reports whether a given input stays
 *   inside the exactly-handled subset, so a caller that needs certainty can check
 *   rather than assume. Closing this properly means a Unicode category table and is
 *   tracked as such.
 */
#ifndef ENG_BPE_H
#define ENG_BPE_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct EngBpe EngBpe;

/* Build from an open container. Reads tokenizer.ggml.tokens, .merges, .token_type and
 * the special-token ids. Returns NULL and explains on stderr if the container has no
 * usable tokenizer. */
EngBpe *eng_bpe_from_gguf(const Gguf *g);
void    eng_bpe_free(EngBpe *b);

int  eng_bpe_vocab_size(const EngBpe *b);
int  eng_bpe_bos(const EngBpe *b);   /* -1 when absent */
int  eng_bpe_eos(const EngBpe *b);
int  eng_bpe_add_bos(const EngBpe *b);

/* The stored text of a token, in the byte-mapped alphabet, NOT UTF-8 the caller can
 * print. Use eng_bpe_decode for anything user-facing. NULL when out of range. */
const char *eng_bpe_token_text(const EngBpe *b, int id, int *len);

/* Is this token a control/special token like <|im_start|>? Those must never be produced
 * by merging ordinary text, and are matched literally before pre-tokenisation. */
int eng_bpe_is_special(const EngBpe *b, int id);

/* Look up a token by its exact stored text. -1 when absent. */
int eng_bpe_find(const EngBpe *b, const char *text, int len);

/* Encode UTF-8 text. Writes at most `cap` ids and returns how many were produced, or -1
 * on error. Does NOT add BOS: whether to is the caller's policy, and Qwen3 sets
 * add_bos_token = 0.
 *
 * `allow_special` matches registered special tokens literally in the input, which is
 * what a chat template needs; with it off, "<|im_start|>" encodes as ordinary text. */
int eng_bpe_encode(const EngBpe *b, const char *text, int len,
                   int *out, int cap, int allow_special);

/* Decode ids back to UTF-8. Returns bytes written, or -1 if they do not fit. Always
 * NUL-terminates when there is room. */
int eng_bpe_decode(const EngBpe *b, const int *ids, int n, char *out, int cap);

/* Decode a single token, appending to a buffer. The common case during generation,
 * where one token arrives at a time. */
int eng_bpe_decode_one(const EngBpe *b, int id, char *out, int cap);

/* Does this input stay inside the subset the pre-tokenizer handles exactly? See the
 * limitation note above. 1 = exact, 0 = contains non-ASCII digits. */
int eng_bpe_ascii_exact(const char *text, int len);

#ifdef __cplusplus
}
#endif

#endif /* ENG_BPE_H */
