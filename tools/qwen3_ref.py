#!/usr/bin/env python3
"""qwen3_ref.py - an independent numpy forward pass for Qwen3, and the golden it emits.

WHAT THIS IS FOR

  The brief is explicit that generating text is not evidence of correctness. A
  transformer with a wrong RoPE convention, a missing QK-norm, a swapped GQA head
  mapping or a transposed projection still produces fluent, confident, wrong output.
  So the C backend is validated against this, tensor by tensor, not against its own
  plausibility.

  Written from the architecture rather than from the C: numpy whole-array operations,
  no shared code, no shared loop structure. It reads the same GGUF the engine reads and
  dequantises with tools/gguf_ref.py, which was itself validated bit-exactly against the
  C kernels at M8 -- so a disagreement here is an ARCHITECTURE difference, not a
  dequantisation one. That separation is deliberate: it means a failure points at one
  layer of the stack rather than two.

WHAT IT EMITS
  A golden file with, for a fixed token sequence:
    - the embedding rows
    - the hidden state after EVERY layer
    - a few named intermediates inside layer 0 (post-norm, Q/K/V, post-QK-norm,
      post-RoPE, attention out, MLP out)
    - the final normed hidden state
    - the full logits row and the greedy token

  Per-layer output is what makes a failure locatable. A logits-only comparison tells you
  something is wrong across 36 layers; a per-layer one tells you it started at layer 7.

MEMORY. The machine this runs on has under 4 GB. Weights are dequantised one matrix at
a time and dropped immediately; the LM head is done in row chunks, because 151,936 x
4096 in f32 is 2.5 GB on its own.
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_ref import parse, dequant_q4_k, dequant_q6_k, BLOCK  # noqa: E402

DEQ = {12: dequant_q4_k, 14: dequant_q6_k}


class Model:
    def __init__(self, path):
        self.mm, self.g = parse(path)
        self.t = {t["name"]: t for t in self.g["tensors"]}
        kv = self.g["kv"]
        self.n_layers = kv["qwen3.block_count"]
        self.d_model = kv["qwen3.embedding_length"]
        self.n_heads = kv["qwen3.attention.head_count"]
        self.n_kv = kv["qwen3.attention.head_count_kv"]
        self.head_dim = kv["qwen3.attention.key_length"]
        self.eps = kv["qwen3.attention.layer_norm_rms_epsilon"]
        self.theta = kv["qwen3.rope.freq_base"]
        self.d_ff = kv["qwen3.feed_forward_length"]
        self.vocab = self.t["token_embd.weight"]["shape"][1]

    def raw(self, name, row0=0, nrows=None):
        """Dequantise `nrows` rows of a tensor, starting at row0, as float32."""
        t = self.t[name]
        cols = t["shape"][0]
        total_rows = 1
        for d in t["shape"][1:]:
            total_rows *= d
        if nrows is None:
            nrows = total_rows
        gt = t["gtype"]
        if gt == 0:                                  # f32, stored plainly
            off = t["file_off"] + row0 * cols * 4
            buf = self.mm[off:off + nrows * cols * 4]
            return np.frombuffer(buf, dtype="<f4").reshape(nrows, cols).astype(np.float32)
        be, bb = BLOCK[gt]
        row_bytes = (cols // be) * bb
        off = t["file_off"] + row0 * row_bytes
        buf = self.mm[off:off + nrows * row_bytes]
        vals = DEQ[gt](buf, nrows * (cols // be))
        return vals.reshape(nrows, cols).astype(np.float32)


def rmsnorm(x, w, eps):
    """y = w * x / sqrt(mean(x^2) + eps). Accumulated in float64: the sum has no
    cancellation, so its error grows monotonically with width."""
    ss = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ss + eps)).astype(np.float32) * w


def rope(v, pos, theta, head_dim):
    """Rotary embedding, HALVED pairing: element i pairs with i + dim/2.

    This is what GGUF and the HF implementation use. The other convention in the wild
    (adjacent pairs, 2i with 2i+1) also preserves every pair norm, so a model built on
    the wrong one runs and is wrong."""
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    freq = 1.0 / (theta ** (2.0 * i / head_dim))
    ang = pos[:, None] * freq[None, :]          # (T, half)
    c, s = np.cos(ang), np.sin(ang)
    a, b = v[..., :half], v[..., half:]
    out = np.empty_like(v)
    out[..., :half] = a * c[:, None, :] - b * s[:, None, :]
    out[..., half:] = a * s[:, None, :] + b * c[:, None, :]
    return out.astype(np.float32)


def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp((x - m).astype(np.float64))
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


def forward(m, ids, dump):
    T = len(ids)
    pos = np.arange(T)

    emb = np.stack([m.raw("token_embd.weight", t, 1)[0] for t in ids]).astype(np.float32)
    x = emb.copy()
    dump("embedding", x)

    hd, nh, nkv = m.head_dim, m.n_heads, m.n_kv
    rep = nh // nkv                       # GQA: 4 query heads per KV head

    for L in range(m.n_layers):
        p = f"blk.{L}."
        h_in = x

        xn = rmsnorm(x, m.raw(p + "attn_norm.weight")[0], m.eps)
        if L == 0:
            dump("l0.attn_norm", xn)

        q = xn @ m.raw(p + "attn_q.weight").T          # (T, 4096)
        k = xn @ m.raw(p + "attn_k.weight").T          # (T, 1024)
        v = xn @ m.raw(p + "attn_v.weight").T          # (T, 1024)
        if L == 0:
            dump("l0.q_proj", q); dump("l0.k_proj", k); dump("l0.v_proj", v)

        q = q.reshape(T, nh, hd)
        k = k.reshape(T, nkv, hd)
        v = v.reshape(T, nkv, hd)

        # QK-norm: RMSNorm per head over head_dim, on Q and K, AFTER projection and
        # BEFORE rope. A Qwen3 signature; K3 has nothing equivalent and omitting it
        # produces a model that runs and is wrong.
        qn = m.raw(p + "attn_q_norm.weight")[0]
        kn = m.raw(p + "attn_k_norm.weight")[0]
        q = rmsnorm(q, qn, m.eps)
        k = rmsnorm(k, kn, m.eps)
        if L == 0:
            dump("l0.q_normed", q.reshape(T, -1)); dump("l0.k_normed", k.reshape(T, -1))

        q = rope(q, pos, m.theta, hd)
        k = rope(k, pos, m.theta, hd)
        if L == 0:
            dump("l0.q_roped", q.reshape(T, -1)); dump("l0.k_roped", k.reshape(T, -1))

        # Grouped-query attention, causal.
        out = np.zeros((T, nh, hd), dtype=np.float32)
        scale = 1.0 / np.sqrt(hd)
        for h in range(nh):
            kh = k[:, h // rep, :]
            vh = v[:, h // rep, :]
            sc = (q[:, h, :] @ kh.T) * scale            # (T, T)
            mask = np.triu(np.ones((T, T), dtype=bool), 1)
            sc = np.where(mask, -np.inf, sc)
            out[:, h, :] = softmax(sc) @ vh
        att = out.reshape(T, nh * hd)
        if L == 0:
            dump("l0.attn_heads", att)

        att = att @ m.raw(p + "attn_output.weight").T
        if L == 0:
            dump("l0.attn_out", att)
        x = h_in + att

        h_mid = x
        xn = rmsnorm(x, m.raw(p + "ffn_norm.weight")[0], m.eps)
        gate = xn @ m.raw(p + "ffn_gate.weight").T
        up = xn @ m.raw(p + "ffn_up.weight").T
        act = (gate / (1.0 + np.exp(-gate.astype(np.float64)))).astype(np.float32) * up
        mlp = act @ m.raw(p + "ffn_down.weight").T
        if L == 0:
            dump("l0.mlp_out", mlp)
        x = h_mid + mlp

        dump(f"layer{L}", x)
        print(f"  layer {L:2d}: |x| mean {np.abs(x).mean():.6f} "
              f"max {np.abs(x).max():.4f}", flush=True)

    x = rmsnorm(x, m.raw("output_norm.weight")[0], m.eps)
    dump("final_norm", x)

    # LM head in row chunks: 151,936 x 4096 in f32 is 2.5 GB.
    last = x[-1]
    logits = np.zeros(m.vocab, dtype=np.float32)
    CH = 8192
    for r0 in range(0, m.vocab, CH):
        n = min(CH, m.vocab - r0)
        w = m.raw("output.weight", r0, n)
        logits[r0:r0 + n] = w @ last
    dump("logits", logits[None, :])

    return int(np.argmax(logits)), logits


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path, outdir = sys.argv[1], sys.argv[2]
    ids = [int(a) for a in sys.argv[3:]] or [9707, 11, 1879, 0]
    os.makedirs(outdir, exist_ok=True)

    m = Model(path)
    print(f"qwen3: {m.n_layers} layers, d_model {m.d_model}, "
          f"heads {m.n_heads}/{m.n_kv}, head_dim {m.head_dim}, "
          f"ffn {m.d_ff}, vocab {m.vocab}")
    print(f"tokens: {ids}")

    cases = []

    def dump(name, arr):
        cases.append((name, np.ascontiguousarray(arr, dtype=np.float32)))

    tok, logits = forward(m, ids, dump)
    print(f"\ngreedy token: {tok}   logit {logits[tok]:.6f}")
    top = np.argsort(logits)[::-1][:5]
    print("top5:", [(int(i), round(float(logits[i]), 4)) for i in top])

    with open(os.path.join(outdir, "qwen3_golden.bin"), "wb") as fh:
        fh.write(b"Q3REF1\0\0")
        fh.write(struct.pack("<II", len(cases), tok))
        fh.write(struct.pack("<I", len(ids)))
        for t in ids:
            fh.write(struct.pack("<i", t))
        for name, arr in cases:
            nb = name.encode()
            fh.write(struct.pack("<I", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<II", arr.shape[0], arr.shape[1]))
            fh.write(arr.tobytes())

    meta = {"tokens": ids, "greedy": tok,
            "cases": [[n, list(a.shape)] for n, a in cases]}
    with open(os.path.join(outdir, "qwen3_golden.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    total = sum(a.nbytes for _, a in cases)
    print(f"\nwrote {len(cases)} tensors ({total/1e6:.1f} MB) to {outdir}/qwen3_golden.bin")


if __name__ == "__main__":
    main()
