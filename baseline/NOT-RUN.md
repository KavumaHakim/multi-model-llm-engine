# Baseline gaps on this machine

These parts of the upstream suite need artifacts that do not fit here. They are
recorded as NOT RUN so a later "all green" cannot be mistaken for full coverage.

| test | needs | why not here |
|---|---|---|
| `test_expert` | `SHARD_DIR` (released safetensors shards) | checkpoint is 1.56 TB; 24.6 GB free |
| `test_real_layer` | `SHARD_DIR` | same |
| `tools/conform_all.py` | `SHARD_DIR` + torch | same, plus torch not installed |
| `test_tok` roundtrip | `tiktoken.model` (~2.8 MB, ships with checkpoint) | not downloaded |
| `make tok` parity | `tiktoken.model` + `tokenization_kimi.py` + `pip install tiktoken` | not downloaded |
| full K3 generation | 1.56 TB checkpoint + 109 GB packed trunk | not obtainable here |

The tokenizer gap is the only one that is cheap to close (~2.8 MB from
`moonshotai/Kimi-K3`). The rest are not, and the engine's own design anticipated
this: correctness is gated by weightless tests precisely so it does not depend on
having the checkpoint.
