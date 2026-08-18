# Extending the runtime

Adding a quantization format, or a CPU kernel. For adding an architecture see
[adding-a-model.md](adding-a-model.md).

Both of these are table entries rather than edits to a switch, which is the point: adding
one should touch the table and its tests, and nothing that consumes it.

---

# Adding a quantization format

## 1. Describe the layout

`src/tensor/dtype.c` holds the size table. Add an entry with the block geometry:

```c
[ENG_DT_Q5_K] = { "q5_k", F|Q|B, 256, 176, 0 },
```

**The byte figures are not derivable from the bit width and must not be guessed.** Q4_K is
144 bytes for 256 elements (4.5 bits/weight), Q3_K is 110 (3.4375) — neither is what "4
bits" or "3 bits" predicts, because the padding is real. Take each number from the format's
own block definition.

There are three layout families, and the flags say which you are:

| flag | meaning | example |
|---|---|---|
| dense | one element, one fixed size | `f32`, `bf16` |
| `BLOCKED` | fixed elements share metadata stored **inside** the block | every GGUF k-quant |
| `EXT_SCALES` | packed weights and scales are **separate tensors** | K3's MXFP4 |
| `ROW_SCALE` | scale sits inline at the head of each row | K3's draft trunk |

A single bytes-per-element number describes only the first. Stretching it over the others
is how a loader silently reads the wrong span.

## 2. Implement the ops

`src/quant/quant.h` defines `EngQuantOps`. The primary operation is a **fused dot
product**, not a dequantiser:

```c
double (*dot_row)(const void *w, const void *scales, const float *x, int n);
void   (*dequant_row)(float *out, const void *w, const void *scales, int n);
int64_t (*row_bytes)(int n);
int64_t (*scale_bytes)(int n);     /* 0 when scales are interleaved */
double  dequant_tolerance;
```

Materialising is what streaming exists to avoid. One K3 expert is 132 MB as fp32 against
17.55 MB packed, and a token touches 1,472 of them. It is not a memory-for-speed trade
either: a matrix-vector product is memory-bound, so the fused path is *faster*.

`dequant_row` exists anyway — tests need an independent path to compare against, and a
weight small enough to stay resident is sometimes cheaper materialised once.

**`scales` is NULL for interleaved formats.** Every entry point takes both pointers so the
interface covers both families; designing for one meant reworking it the moment the second
model arrived.

## 3. Declare a tolerance, and justify it

`dequant_tolerance` is per format because it depends on the mantissa width, not on taste.

MXFP4 and Q6_K declare `1e-9`: every product is exact in double — a 3-bit or 6-bit code
times an f16-derived scale needs far fewer than 53 bits — so only the group reassociation
moves the result, about 1 ULP.

Q4_K declares `1e-6`, and the reason is structural: it is an **affine** format,
`w = (d·sc)·q − (dmin·m)`. The scale is an f16 times a 6-bit integer and a second such
product is subtracted, so the fused and materialised forms group their roundings
differently. Dropping the min term entirely is the classic Q4_K bug — outputs stay finite
and plausibly scaled, and only a numerical comparison catches it.

## 4. Register and test

One line in `reg_init()` in `src/quant/quant.c`. Registration refuses a duplicate dtype
and refuses incomplete ops, so a half-added format fails at startup.

`tests/unit/test_quant.c` checks the accuracy contract by **iterating the registry**, so
your format is covered automatically. What it cannot do for you is pin the byte layout — a
wrong nibble order decodes to equally plausible weights and still emits fluent text. Add
hand-computed values, as the MXFP4 section does:

```c
w[0] = 0x21;   /* low = 1 -> 0.5 (element 0); high = 2 -> 1.0 (element 1) */
```

Then validate against something independent. `tools/gguf_ref.py` implements Q4_K and Q6_K
from the block layout in numpy, vectorised over blocks, sharing no loop structure with the
C. Across 18 rows from the real model: worst difference 0.000e+00.

---

# Adding a CPU kernel

## 1. The reference is normative

Write the portable C version first, in `src/kernels/kernel.c`. Under `ENG_NUM_EXACT` every
other implementation must reproduce its bits, so **its reduction order is part of the
contract**, not an implementation detail.

Write it in the shape a vector register reproduces. The existing kernels use 16 double
accumulators folded as `(a0+a4)+(a8+a12)`, which is exactly four `__m256d` lanes; the
k-quant kernels use four. A serial `s += a*b` chain cannot be vectorised without changing
the summation order, so starting serial means rewriting it later.

Use `fma()` rather than `+=` for the multiply-add. `_mm256_fmadd_pd` rounds once; with
`-ffp-contract=off` a separate multiply and add rounds twice, and the paths diverge in the
last bits.

## 2. The vector version

`src/kernels/kernel_avx2.c` compiles **unconditionally**, via per-function
`__attribute__((target("avx2,fma")))` rather than `-mavx2` on the translation unit. That is
what lets one binary carry both paths and choose at runtime. A file-level flag would let
the compiler hoist AVX2 into a function reachable before the CPU check.

Map lanes to accumulators explicitly:

```
v0 lane j <-> a[j]      v1 lane j <-> a[j+4]
v2 lane j <-> a[j+8]    v3 lane j <-> a[j+12]
```

so `(v0+v1)+(v2+v3)` computes the reference's fold lane-wise. Fold the tail with the
reference's own loop, in the same order.

## 3. Numerical policy

`ENG_NUM_EXACT` buys bit-identity and costs half the width — 4 doubles against 8 floats.
`ENG_NUM_FAST` accumulates in fp32 and must only match the reference within a stated
tolerance.

Take the policy as a parameter rather than deciding inside the kernel. K3 needs exactness
because its claim is that output does not depend on the memory budget; Qwen3 does not, and
should not pay for it.

## 4. Gate it on raw bits

```c
ok(memcmp(&ref, &vec, sizeof ref) == 0, "bit-identical", d);
```

**Not a tolerance.** A tolerance-based test passes on an implementation that is merely
close, and merely close is precisely the failure — it means output depends on which machine
ran it.

Test lengths that exercise the tail. The exact kernels consume 16 elements per step, so 16
and 32 have no remainder while 17, 31, 63 and 4097 do, and the tail must fold in the
reference's order or the bits diverge. Compare the **double** accumulator too, since
narrowing to float can mask a low-bit difference.

Assert the dispatch actually reached your code. Without that, the comparison may be
validating something that never runs:

```c
ok(eng_kernel_active() == ENG_IMPL_AVX2, "dispatches to AVX2 on this host",
   "otherwise the comparison above tests dead code");
```

`make test` also builds the kernel tests **without** `-mavx2` and requires the same checks
to pass — that is what proves the runtime dispatch is doing the work rather than the build
flags.

## 5. Measure it, on one binary

Provide an environment override (`ENG_KQ_SCALAR`, `ENG_LM_RESIDENT`) so the A/B runs on one
binary at one configuration with only the kernel changed.

This is not ceremony. The LM-head residency question in this project was answered wrongly
the first time because the two arms had different memory budgets, and the faster one simply
had more memory. The corrected comparison reversed the conclusion — 2.57× in the opposite
direction.

Expect less than the arithmetic suggests, and record why. Vectorising the k-quant dots gave
1.17× on compute, not 3–4×, because bit-identity forces double accumulation. That number is
in [PERFORMANCE.md](PERFORMANCE.md) with its explanation rather than quietly omitted.
