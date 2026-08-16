#!/usr/bin/env python3
"""gguf_ref.py - an INDEPENDENT reference for the GGUF reader and the k-quant kernels.

WHY THIS EXISTS, and why it is not a thin wrapper over the C

  Q4_K and Q6_K are the highest-risk code in this engine for silent wrongness. A wrong
  nibble order, a wrong scale index, an unsigned read of a signed scale, or a dropped
  affine term all produce weights that are finite, plausibly scaled, and wrong -- and a
  model built on them still emits fluent text. Nothing downstream catches that.

  So this is written from the block layout directly and vectorised with numpy array
  operations, structurally unlike the C's scalar loops. Two implementations that share
  no code and no loop structure are unlikely to make the SAME mistake; one of them
  checking itself is not evidence of anything.

  It also parses the GGUF header independently, so the C reader's tensor table is
  checked against a second opinion rather than against its own arithmetic.

Usage
  python3 tools/gguf_ref.py MODEL.gguf OUT_DIR
    writes OUT_DIR/golden.bin  - dequantised reference rows
           OUT_DIR/facts.json  - structural facts for cross-checking
"""
import json
import mmap
import os
import struct
import sys

import numpy as np

# ---------------------------------------------------------------- header parse --

GT_UINT8, GT_INT8, GT_UINT16, GT_INT16 = 0, 1, 2, 3
GT_UINT32, GT_INT32, GT_FLOAT32, GT_BOOL = 4, 5, 6, 7
GT_STRING, GT_ARRAY, GT_UINT64, GT_INT64, GT_FLOAT64 = 8, 9, 10, 11, 12

_FIXED = {
    GT_UINT8: ("<B", 1), GT_INT8: ("<b", 1), GT_UINT16: ("<H", 2), GT_INT16: ("<h", 2),
    GT_UINT32: ("<I", 4), GT_INT32: ("<i", 4), GT_FLOAT32: ("<f", 4),
    GT_BOOL: ("<B", 1), GT_UINT64: ("<Q", 8), GT_INT64: ("<q", 8), GT_FLOAT64: ("<d", 8),
}

# ggml type id -> (name, elements per block, bytes per block)
GGML = {
    0: ("f32", 1, 4), 1: ("f16", 1, 2),
    8: ("q8_0", 32, 34),
    12: ("q4_k", 256, 144), 13: ("q5_k", 256, 176), 14: ("q6_k", 256, 210),
    30: ("bf16", 1, 2),
}


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def raw(self, n):
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def num(self, fmt, n):
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += n
        return v

    def u32(self):
        return self.num("<I", 4)

    def u64(self):
        return self.num("<Q", 8)

    def string(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        if t in _FIXED:
            fmt, n = _FIXED[t]
            return self.num(fmt, n)
        if t == GT_STRING:
            return self.string()
        if t == GT_ARRAY:
            at = self.u32()
            n = self.u64()
            if at in _FIXED:
                fmt, sz = _FIXED[at]
                # Skip rather than materialise: the token list is 151,936 entries.
                self.o += sz * n
                return ("<array>", at, n)
            if at == GT_STRING:
                for _ in range(n):
                    self.string()
                return ("<array>", at, n)
            raise ValueError(f"array of type {at}")
        raise ValueError(f"metadata type {t}")


def parse(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    r = Reader(mm)

    if r.raw(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()

    kv = {}
    for _ in range(n_kv):
        key = r.string()
        kv[key] = r.value(r.u32())

    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        shape = [r.u64() for _ in range(nd)]
        gtype = r.u32()
        off = r.u64()
        if gtype not in GGML:
            raise SystemExit(f"{name}: unhandled ggml type {gtype}")
        tname, be, bb = GGML[gtype]
        nelem = 1
        for d in shape:
            nelem *= d
        if nelem % be:
            raise SystemExit(f"{name}: {nelem} elements is not whole blocks of {be}")
        tensors.append(dict(name=name, shape=shape, gtype=gtype, tname=tname,
                            rel_off=off, nbytes=(nelem // be) * bb))

    align = kv.get("general.alignment", 32)
    if isinstance(align, tuple):
        align = 32
    hdr_end = r.o
    data_start = (hdr_end + align - 1) & ~(align - 1)
    for t in tensors:
        t["file_off"] = data_start + t["rel_off"]

    return mm, dict(version=version, kv=kv, tensors=tensors,
                    alignment=align, header_end=hdr_end, data_start=data_start,
                    file_size=os.path.getsize(path))


# ------------------------------------------------------------------ dequant --
#
# Written from the block layout, vectorised over blocks. Deliberately NOT structured
# like the C, which walks elements in scalar loops.

def f16(u16):
    return np.frombuffer(np.asarray(u16, dtype="<u2").tobytes(), dtype="<f2").astype(np.float64)


def dequant_q4_k(raw, nblocks):
    """Q4_K: d(f16) dmin(f16) scales[12] qs[128]. w = d*sc*q - dmin*m."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(nblocks, 144)

    d = f16(a[:, 0:2].copy().view("<u2").ravel())        # (B,)
    dmin = f16(a[:, 2:4].copy().view("<u2").ravel())
    sc12 = a[:, 4:16].astype(np.uint32)                  # (B,12)
    qs = a[:, 16:144]                                    # (B,128)

    # Unpack eight 6-bit scales and eight 6-bit mins.
    sc = np.zeros((nblocks, 8), dtype=np.float64)
    mn = np.zeros((nblocks, 8), dtype=np.float64)
    for j in range(8):
        if j < 4:
            sc[:, j] = sc12[:, j] & 63
            mn[:, j] = sc12[:, j + 4] & 63
        else:
            sc[:, j] = (sc12[:, j + 4] & 0x0F) | ((sc12[:, j - 4] >> 6) << 4)
            mn[:, j] = (sc12[:, j + 4] >> 4) | ((sc12[:, j] >> 6) << 4)

    out = np.zeros((nblocks, 256), dtype=np.float64)
    for pair in range(4):                      # four 32-byte stretches
        q = qs[:, pair * 32:(pair + 1) * 32].astype(np.float64)
        lo = q % 16.0
        hi = np.floor(q / 16.0)
        s0, m0 = sc[:, 2 * pair], mn[:, 2 * pair]
        s1, m1 = sc[:, 2 * pair + 1], mn[:, 2 * pair + 1]
        out[:, pair * 64:pair * 64 + 32] = (d * s0)[:, None] * lo - (dmin * m0)[:, None]
        out[:, pair * 64 + 32:pair * 64 + 64] = (d * s1)[:, None] * hi - (dmin * m1)[:, None]
    return out.ravel()


def dequant_q6_k(raw, nblocks):
    """Q6_K: ql[128] qh[64] scales[16](int8) d(f16). w = d*sc*(q-32)."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(nblocks, 210)

    ql_all = a[:, 0:128]
    qh_all = a[:, 128:192]
    sc_all = a[:, 192:208].astype(np.int8).astype(np.float64)   # SIGNED
    d = f16(a[:, 208:210].copy().view("<u2").ravel())

    out = np.zeros((nblocks, 256), dtype=np.float64)
    for half in range(2):
        ql = ql_all[:, half * 64:(half + 1) * 64].astype(np.int64)
        qh = qh_all[:, half * 32:(half + 1) * 32].astype(np.int64)
        sc = sc_all[:, half * 8:(half + 1) * 8]
        base = half * 128

        l = np.arange(32)
        isx = l // 16                                    # scale index within the half

        q1 = (ql[:, l] & 0x0F) | (((qh[:, l] >> 0) & 3) << 4)
        q2 = (ql[:, l + 32] & 0x0F) | (((qh[:, l] >> 2) & 3) << 4)
        q3 = (ql[:, l] >> 4) | (((qh[:, l] >> 4) & 3) << 4)
        q4 = (ql[:, l + 32] >> 4) | (((qh[:, l] >> 6) & 3) << 4)

        out[:, base + 0:base + 32] = d[:, None] * sc[:, isx + 0] * (q1 - 32)
        out[:, base + 32:base + 64] = d[:, None] * sc[:, isx + 2] * (q2 - 32)
        out[:, base + 64:base + 96] = d[:, None] * sc[:, isx + 4] * (q3 - 32)
        out[:, base + 96:base + 128] = d[:, None] * sc[:, isx + 6] * (q4 - 32)
    return out.ravel()


DEQ = {12: dequant_q4_k, 14: dequant_q6_k}
BLOCK = {12: (256, 144), 14: (256, 210)}


# --------------------------------------------------------------------- main --

def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    mm, g = parse(path)
    ts = g["tensors"]

    # Structural facts, for the C reader to be checked against.
    total = sum(t["nbytes"] for t in ts)
    by_type = {}
    for t in ts:
        e = by_type.setdefault(t["tname"], {"count": 0, "bytes": 0})
        e["count"] += 1
        e["bytes"] += t["nbytes"]

    seq = all(ts[i]["rel_off"] >= ts[i - 1]["rel_off"] for i in range(1, len(ts)))

    facts = {
        "version": g["version"],
        "n_tensors": len(ts),
        "n_kv": len(g["kv"]),
        "alignment": g["alignment"],
        "header_end": g["header_end"],
        "data_start": g["data_start"],
        "file_size": g["file_size"],
        "tensor_bytes_total": total,
        "sequential": seq,
        "by_type": by_type,
        "arch": g["kv"].get("general.architecture"),
        "block_count": g["kv"].get("qwen3.block_count"),
        "embedding_length": g["kv"].get("qwen3.embedding_length"),
        "head_count": g["kv"].get("qwen3.attention.head_count"),
        "head_count_kv": g["kv"].get("qwen3.attention.head_count_kv"),
        "rope_freq_base": g["kv"].get("qwen3.rope.freq_base"),
        "rms_eps": g["kv"].get("qwen3.attention.layer_norm_rms_epsilon"),
        "context_length": g["kv"].get("qwen3.context_length"),
    }
    # A few named tensors the C test looks up by name.
    probe_names = ["token_embd.weight", "output.weight", "output_norm.weight",
                   "blk.0.attn_q.weight", "blk.0.attn_v.weight", "blk.0.ffn_down.weight",
                   "blk.35.ffn_up.weight"]
    facts["probe"] = {t["name"]: {"shape": t["shape"], "type": t["tname"],
                                  "file_off": t["file_off"], "nbytes": t["nbytes"]}
                      for t in ts if t["name"] in probe_names}

    with open(os.path.join(outdir, "facts.json"), "w") as fh:
        json.dump(facts, fh, indent=1, sort_keys=True)

    # Golden rows: real quantised rows from real tensors, dequantised by THIS code.
    cases = []
    want = [("token_embd.weight", 12, [0, 1, 100, 151935]),
            ("blk.0.attn_q.weight", 12, [0, 1, 4095]),
            ("blk.0.ffn_gate.weight", 12, [0, 7, 12287]),
            ("output.weight", 14, [0, 1, 151935]),
            ("blk.0.attn_v.weight", 14, [0, 5, 1023]),
            ("blk.35.ffn_down.weight", 14, [0, 4095])]
    tmap = {t["name"]: t for t in ts}

    for name, gtype, rows in want:
        t = tmap.get(name)
        if t is None or t["gtype"] != gtype:
            print(f"  skip {name}: absent or type {t and t['tname']}")
            continue
        cols = t["shape"][0]
        be, bb = BLOCK[gtype]
        row_bytes = (cols // be) * bb
        nblk = cols // be
        for r in rows:
            nrows = 1
            for d in t["shape"][1:]:
                nrows *= d
            if r >= nrows:
                continue
            off = t["file_off"] + r * row_bytes
            raw = mm[off:off + row_bytes]
            vals = DEQ[gtype](raw, nblk).astype(np.float32)
            # The RAW block bytes travel with the reference values, so the k-quant
            # kernels can be validated on a machine that does not have the 5 GB model.
            # Without this the golden file would only be usable next to the file it came
            # from, which is the one place the kernels are least likely to be run.
            cases.append((name, gtype, r, vals, bytes(raw)))
            print(f"  {name} row {r}: {cols} vals, "
                  f"min {vals.min():+.5f} max {vals.max():+.5f} "
                  f"mean {vals.mean():+.6f} std {vals.std():.6f}")

    with open(os.path.join(outdir, "golden.bin"), "wb") as fh:
        fh.write(b"GGREF2\0\0")
        fh.write(struct.pack("<I", len(cases)))
        for name, gtype, row, vals, raw in cases:
            nb = name.encode()
            fh.write(struct.pack("<I", len(nb)))
            fh.write(nb)
            fh.write(struct.pack("<IQII", gtype, row, len(vals), len(raw)))
            fh.write(vals.tobytes())
            fh.write(raw)

    print(f"\nwrote {len(cases)} golden rows to {outdir}/golden.bin")
    print(f"tensors {len(ts)}, data_start {g['data_start']}, "
          f"tensor bytes {total}, sequential={seq}")


if __name__ == "__main__":
    main()
