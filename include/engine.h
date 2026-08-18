/* SPDX-License-Identifier: Apache-2.0 */
/*
 * engine.h - the public interface of libengine.
 *
 * One header, one library. Everything reachable from here is what another project may
 * depend on; anything under src/ that this does not pull in is internal and may change.
 *
 * WHAT THIS IS AND IS NOT. libengine is a single-sequence, synchronous inference
 * runtime. It is suitable for embedding in a tool, a batch job, or a server that
 * serialises requests. It is NOT thread-safe for concurrent decoding: the weight cache
 * documents this explicitly, and one model handle serves one sequence at a time. Running
 * several models in one process is fine; running one model from several threads is not.
 *
 *   link with:  -lengine -lm -fopenmp -pthread
 *
 * THE SHAPE OF A PROGRAM
 *
 *     eng_hwinfo_detect(&hw);                    what this machine is
 *     b = eng_model_probe(path, NULL);           which backend claims the file
 *     b->inspect(path, &facts);                  what the model costs, no weights read
 *     eng_plan(&plan, &hw, &facts, &req);        budgets, threads, context
 *     m = b->load(&lr);                          now the weights
 *     s = b->state_create(m, plan.context);
 *     b->decode(m, s, token, pos, flags);        one position at a time
 *     b->logits(m, &n_vocab);
 *
 * The plan is computed before anything is loaded, deliberately: `inspect` and the
 * planner answer "will this run, and how" without touching a weight, so a caller can
 * decide before paying for a load.
 */
#ifndef ENGINE_H
#define ENGINE_H

/* Hardware, memory budget, execution plan. */
#include "hwinfo.h"
#include "memory.h"
#include "planner.h"

/* Tensors and data types. */
#include "dtype.h"
#include "tensor.h"

/* Storage, caching, streaming. Useful on their own: they are model-independent, and a
 * project that only wants the streamer or the weight cache can take them. */
#include "storage.h"
#include "cache.h"
#include "streamer.h"

/* Containers. */
#include "gguf.h"

/* Quantized formats, as a registry of fused dot products. */
#include "quant.h"

/* CPU kernels with runtime ISA dispatch. */
#include "kernel.h"

/* Model backends: the registry, the vtable, and the two shipped architectures. */
#include "model.h"

/* Tokenization and chat formatting. */
#include "bpe.h"
#include "chat.h"

#define ENGINE_VERSION_MAJOR 0
#define ENGINE_VERSION_MINOR 1
#define ENGINE_VERSION_PATCH 0
#define ENGINE_VERSION_STRING "0.1.0"

#endif /* ENGINE_H */
