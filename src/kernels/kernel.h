/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kernel.h - CPU kernels, selected at runtime, with numerical policy as a parameter.
 *
 * TWO PROBLEMS THIS SOLVES, and they pull in opposite directions.
 *
 * 1. ONE BINARY, MANY CPUs
 *    K3 chose its vector paths with `#if defined(__AVX2__)`, which bakes the decision
 *    into the build. A -march=native binary SIGILLs on an older machine; a portable one
 *    leaves the vector units idle on a newer one. Here both paths are COMPILED (the
 *    AVX2 file uses __attribute__((target("avx2,fma"))) so it builds even without
 *    -mavx2 on the command line) and the choice is made once, at first use, from what
 *    the CPU actually reports.
 *
 * 2. K3 AND QWEN3 WANT DIFFERENT ARITHMETIC, and neither is wrong
 *
 *    K3's entire value proposition is that its output is byte-identical whether it runs
 *    in 8 GB or 224 GB. That requires the vector path to be BIT-IDENTICAL to the scalar
 *    path -- not close, identical -- because a streamed layer and a resident layer must
 *    produce the same bits. K3 achieves it by accumulating in double with a fixed
 *    16-accumulator reduction tree that the AVX2 path reproduces lane for lane.
 *
 *    That costs real throughput: an AVX2 register holds 4 doubles against 8 floats, so
 *    exact accumulation runs at half the vector width before any other consideration.
 *
 *    Qwen3 has no such contract. Nothing about it requires bit-identity across memory
 *    configurations, and fp32 accumulation is numerically fine for a 4096-wide dot
 *    product. Forcing K3's policy on it would halve its speed for a guarantee it does
 *    not need; forcing Qwen3's policy on K3 would silently break the one claim K3 makes.
 *
 *    So the policy is a PARAMETER, chosen per backend, not a build flag:
 *
 *      ENG_NUM_EXACT  double accumulation, fixed reduction order. Every implementation
 *                     must agree BIT FOR BIT with the reference. Gated by a test that
 *                     compares raw bits, not values.
 *      ENG_NUM_FAST   fp32 accumulation, natural reduction. Implementations must agree
 *                     with the reference to a stated tolerance, and are free to differ
 *                     below it.
 *
 *    A backend declares which it needs through its capabilities. Getting this wrong in
 *    either direction is silent, which is why it is a first-class parameter rather than
 *    something a kernel decides for itself.
 *
 * WHAT IS NOT HERE
 *    K3's architecture-specific kernels -- KDA, MLA, SiTU-GLU, the attention-residual
 *    aggregation, MXFP4 -- stay in src/core/k3_ops.c and move to models/kimi/ at M7.
 *    This file carries only what more than one architecture needs.
 */
#ifndef ENG_KERNEL_H
#define ENG_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ENG_NUM_EXACT = 0,   /* bit-identical across implementations; double accumulate */
    ENG_NUM_FAST  = 1    /* fp32 accumulate; equal to the reference within tolerance */
} EngNumPolicy;

typedef enum {
    ENG_IMPL_AUTO = 0,   /* pick the best the CPU supports */
    ENG_IMPL_REF  = 1,   /* portable C, always available */
    ENG_IMPL_AVX2 = 2
} EngImpl;

/* Select the implementation. Call once at startup; ENG_IMPL_AUTO consults the CPU.
 * Requesting an implementation the CPU cannot run falls back to the reference and says
 * so on stderr rather than crashing on the first instruction. Returns what was chosen. */
EngImpl eng_kernel_select(EngImpl want);

/* What is active, and what this binary contains. */
EngImpl     eng_kernel_active(void);
const char *eng_kernel_name(EngImpl impl);
int         eng_kernel_available(EngImpl impl);

/* ---------------------------------------------------------------- primitives -- */
/*
 * Shapes follow the engine's convention: W is row-major [out][in], x is [in], y is
 * [out]. No bias anywhere -- neither K3 nor Qwen3 has one in any projection.
 */

/* y[out] = W[out][in] . x[in], fp32 weights. */
void eng_matmul_f32(float *y, const float *x, const float *W,
                    int in, int out, EngNumPolicy pol);

/* Same, bf16 weights widened inside the kernel. bf16 -> f32 is a pure 16-bit left
 * shift: bf16 IS the top half of an f32, so there is no rounding and no table. */
void eng_matmul_bf16(float *y, const float *x, const uint16_t *W,
                     int in, int out, EngNumPolicy pol);

/* Single dot product, exposed because attention scores need it outside a matmul. */
float eng_dot_f32(const float *a, const float *b, int n, EngNumPolicy pol);

/* y = w * x / sqrt(mean(x^2) + eps).
 *
 * eps is INSIDE the rsqrt, and the sum of squares accumulates in double even under
 * ENG_NUM_FAST. That is not an oversight: the sum runs over the full hidden width
 * (4096 for Qwen3, 7168 for K3) and it is a sum of squares, so every term has the same
 * sign and the error accumulates monotonically rather than cancelling. The cost is one
 * reduction per normalisation against a whole matmul, which is not measurable. */
void eng_rmsnorm(float *y, const float *x, const float *w, int n, float eps);

/* In-place softmax over n elements, max-subtracted for stability. */
void eng_softmax(float *x, int n);

/* SiLU (x * sigmoid(x)), elementwise. */
void eng_silu(float *y, const float *x, int n);

/* SwiGLU over a 2n input laid out [gate | up]: y = silu(gate) * up.
 *
 * This is Qwen3's activation and is NOT K3's. K3 uses SiTU-GLU, which caps both halves
 * with tanh and has a different gate; the two are not interchangeable and K3's stays in
 * its own backend. */
void eng_swiglu(float *y, const float *x, int n);

/* Rotary position embedding, applied in place to one head of `dim` elements at
 * position `pos`.
 *
 * PAIRING IS A CONVENTION, NOT A RULE, and the two conventions in the wild produce
 * different models from the same weights:
 *
 *   ENG_ROPE_HALVED   element i pairs with i + dim/2. What GGUF/llama.cpp and the
 *                     Hugging Face implementations use, and therefore what Qwen3 needs.
 *   ENG_ROPE_ADJACENT element 2i pairs with 2i+1. The original RoFormer formulation.
 *
 * Both rotate every element by the same angles; they differ only in which element is
 * the real part and which the imaginary. Choosing wrong gives a model that runs, emits
 * fluent text, and is wrong -- the same failure mode as K3's MXFP4 nibble order. */
typedef enum { ENG_ROPE_HALVED = 0, ENG_ROPE_ADJACENT = 1 } EngRopeStyle;

void eng_rope(float *v, int dim, int pos, float theta_base, EngRopeStyle style);

/* ------------------------------------------------------------------- report -- */

void eng_kernel_report(const char *label);

#ifdef __cplusplus
}
#endif

#endif /* ENG_KERNEL_H */
