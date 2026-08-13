/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kernel_impl.h - internal seam between the dispatcher and the per-ISA implementations.
 *
 * Not a public header. It exists so kernel.c can call the AVX2 entry points without
 * their definitions being visible to it -- which matters, because kernel.c is compiled
 * WITHOUT -mavx2 in a portable build and must not inline anything that would emit an
 * AVX2 instruction into a function the dispatcher can reach on a machine without it.
 */
#ifndef ENG_KERNEL_IMPL_H
#define ENG_KERNEL_IMPL_H

#include <stdint.h>

/* bf16 -> f32 is a pure 16-bit left shift: bf16 IS the top half of an f32. No rounding,
 * no lookup table, no exponent rebias (unlike f16, which needs all three). */
static inline float eng_bf16f(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* Reference. Normative under ENG_NUM_EXACT: every other implementation must reproduce
 * these bits, so the reduction order is contract, not detail. */
double eng_ref_dot_f32_exact (const float *a, const float *b, int n);
float  eng_ref_dot_f32_fast  (const float *a, const float *b, int n);
double eng_ref_dot_bf16_exact(const float *x, const uint16_t *w, int n);
float  eng_ref_dot_bf16_fast (const float *x, const uint16_t *w, int n);

/* AVX2. Always LINKED -- the file compiles with function-level target attributes, so it
 * builds even without -mavx2 on the command line -- but only CALLED when the CPU
 * reports support. eng_avx2_compiled() distinguishes "this binary has no AVX2 code" from
 * "this CPU cannot run it", which need different fixes. */
int    eng_avx2_compiled(void);
int    eng_cpu_has_avx2(void);
double eng_avx2_dot_f32_exact (const float *a, const float *b, int n);
float  eng_avx2_dot_f32_fast  (const float *a, const float *b, int n);
double eng_avx2_dot_bf16_exact(const float *x, const uint16_t *w, int n);
float  eng_avx2_dot_bf16_fast (const float *x, const uint16_t *w, int n);

#endif /* ENG_KERNEL_IMPL_H */
