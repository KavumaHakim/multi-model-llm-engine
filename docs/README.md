# Documentation

## This project

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | the generic runtime and the backend seam |
| [adding-a-model.md](adding-a-model.md) | adding an architecture, written from what Qwen3 cost |
| [extending.md](extending.md) | adding a quantization format, or a CPU kernel |
| [PERFORMANCE.md](PERFORMANCE.md) | measured figures, two machines, kept apart |
| [ROADMAP.md](ROADMAP.md) | what is next, with the reasoning |
| [qwen3-model-facts.md](qwen3-model-facts.md) | Qwen3-8B as read from the GGUF, not from documentation |

## How this came about

| | |
|---|---|
| [architecture-report.md](architecture-report.md) | the audit of kimi-k3-in-c that shaped the design |
| [baseline-m0.md](baseline-m0.md) | the pre-refactor K3 baseline, and the invariants held to since |
| [README-kimi-k3.md](README-kimi-k3.md) | upstream's README, verbatim |

## Kimi K3 specifics

These predate the refactor and describe the K3 engine as upstream shipped it. The
subsystems they refer to are now generic, but the K3 *backend* still behaves as described,
and its execution path is still the original CLI.

| | |
|---|---|
| [architecture-kimi-k3.md](architecture-kimi-k3.md) | how K3 maps onto the codebase |
| [QUICKSTART.md](QUICKSTART.md) | running K3 from nothing |
| [TUNING.md](TUNING.md) | choosing a K3 memory budget |
| [API.md](API.md) | the K3 C API |
| [TESTING.md](TESTING.md) | the test suite as it stood at M0 |
| [BENCHMARKING.md](BENCHMARKING.md) | the benchmark harness |

For the current engine use `./bin/engine --help`; for the current test surface see
`scripts/verify.sh`, which is what every milestone is gated on.
