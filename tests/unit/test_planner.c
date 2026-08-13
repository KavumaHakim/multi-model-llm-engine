/* SPDX-License-Identifier: Apache-2.0 */
/*
 * test_planner.c - hardware detection, memory partitioning, and the execution planner.
 *
 * The planner's job is to never choose something that cannot run, so most of these
 * cases are REFUSALS. A planner that always returns a plan is useless: the failure it
 * exists to prevent is the OOM killer arriving partway through a token, and the only
 * way to prevent that is to say no at plan time with the numbers attached.
 *
 * The model facts used here are the real Qwen3-8B figures read from the target GGUF
 * (docs/qwen3-model-facts.md) and the real K3 figures from its documentation, so the
 * arithmetic is exercised at the scales it will actually meet rather than at round
 * numbers chosen to make it pass.
 */
#include <stdio.h>
#include <string.h>

#include "hwinfo.h"
#include "memory.h"
#include "planner.h"

static int fails = 0, checks = 0;

static void ok(int cond, const char *what, const char *detail)
{
    checks++;
    printf("  %s  %-46s %s\n", cond ? "PASS" : "FAIL", what, detail ? detail : "");
    if (!cond) fails++;
}

static void eqi(long long got, long long want, const char *what)
{
    char d[96];
    snprintf(d, sizeof d, "got %lld want %lld", got, want);
    ok(got == want, what, d);
}

/* Qwen3-8B, measured from Qwen3-8B-Q4_K_M.gguf. */
static void qwen3_facts(EngModelFacts *m)
{
    memset(m, 0, sizeof *m);
    m->arch               = "qwen3";
    m->n_layers           = 36;
    m->total_weight_bytes = 5021830000LL;         /* 5021.83 MB of tensor data     */
    m->resident_bytes     = 860000000LL;          /* token_embd 350 MB + output 510 */
    m->max_layer_bytes    = 131000000LL;          /* ~130.4 MB per layer            */
    m->avg_layer_bytes    = 130000000LL;
    m->activation_bytes   = 16000000LL;
    m->scratch_bytes      = 32000000LL;
    /* 36 layers x 8 kv heads x 128 dim x 2 (K and V) x 4 bytes = 294,912 */
    m->kv_bytes_per_pos   = 294912;
    m->context_max        = 40960;
    m->bytes_per_weight   = 0.58;                 /* Q4_K/Q6_K mixture              */
}

/* Kimi K3, from its own documentation. */
static void k3_facts(EngModelFacts *m)
{
    memset(m, 0, sizeof *m);
    m->arch               = "kimi-k3";
    m->n_layers           = 93;
    m->total_weight_bytes = 113490000000LL;       /* trunk 108.81 GB + 4.70 GB      */
    m->resident_bytes     = 4700000000LL;
    m->max_layer_bytes    = 2340000000LL;         /* layer 0, the dense MLP         */
    m->avg_layer_bytes    = 1270000000LL;
    m->activation_bytes   = 700000000LL;
    m->scratch_bytes      = 200000000LL;
    m->kv_bytes_per_pos   = 2370000;
    m->context_max        = 4096;
    m->n_experts          = 896;
    m->topk               = 16;
    m->expert_bytes       = 17547264LL;
    m->bytes_per_weight   = 2.0;                  /* the trunk is bf16              */
}

static void hw_fake(EngHwInfo *hw, int phys, int logi, int64_t ram)
{
    memset(hw, 0, sizeof *hw);
    snprintf(hw->cpu_name, sizeof hw->cpu_name, "test cpu");
    hw->cores_physical = phys;
    hw->cores_logical  = logi;
    hw->isa = hw->isa_built = ENG_ISA_SSE2 | ENG_ISA_AVX | ENG_ISA_AVX2 | ENG_ISA_FMA;
    hw->ram_total = ram;
    hw->ram_available = ram * 4 / 5;
    hw->storage = ENG_STORE_SSD;
}

/* ------------------------------------------------------------------ hwinfo -- */

static void test_hwinfo(void)
{
    printf("== hardware detection (this machine) ==\n");
    EngHwInfo hw;
    eng_hwinfo_detect(&hw);
    eng_hwinfo_report(&hw, "  ");

    ok(hw.cores_logical >= 1, "logical cores detected", NULL);
    ok(hw.cores_physical >= 1, "physical cores detected", NULL);
    ok(hw.cores_physical <= hw.cores_logical,
       "physical <= logical", "SMT can only add logical processors");
    ok(hw.cpu_name[0] != '\0', "cpu name populated", hw.cpu_name);

    /* The build flags are known at compile time, so this is checkable exactly: if the
     * binary was compiled with AVX2 then the CPU running it must have AVX2, or we
     * would have crashed before reaching this line. */
#if defined(__AVX2__)
    ok((hw.isa & ENG_ISA_AVX2) != 0,
       "runtime AVX2 agrees with the build", "built with -mavx2 and still running");
    ok((hw.isa_built & ENG_ISA_AVX2) != 0, "built ISA records AVX2", NULL);
#else
    ok((hw.isa_built & ENG_ISA_AVX2) == 0, "built ISA correctly omits AVX2", NULL);
#endif

    char s[96];
    eng_hwinfo_isa_string(hw.isa, s, sizeof s);
    ok(s[0] != '\0', "isa string non-empty", s);
    eng_hwinfo_isa_string(0, s, sizeof s);
    ok(!strcmp(s, "(none)"), "empty isa reports (none)", s);
}

/* ------------------------------------------------------------- size parsing -- */

static void test_parse(void)
{
    printf("\n== size parsing ==\n");
    int is_auto = 0;

    eqi(eng_mem_parse_size("1024", &is_auto), 1024, "bare bytes");
    eqi(eng_mem_parse_size("2K", &is_auto), 2048, "2K");
    eqi(eng_mem_parse_size("5M", &is_auto), 5LL << 20, "5M");
    eqi(eng_mem_parse_size("6G", &is_auto), 6LL << 30, "6G");
    eqi(eng_mem_parse_size("6g", &is_auto), 6LL << 30, "lower case");
    eqi(eng_mem_parse_size("6GB", &is_auto), 6LL << 30, "trailing B");
    eqi(eng_mem_parse_size("1.5G", &is_auto), (int64_t)(1.5 * (1LL << 30)), "fractional");

    eng_mem_parse_size("auto", &is_auto);
    ok(is_auto == 1, "auto recognised", NULL);

    /* REFUSE rather than guess. A mistyped budget that silently becomes a different
     * number is how a machine gets OOM-killed. */
    ok(eng_mem_parse_size("6X", &is_auto) == -1, "unknown suffix refused", "6X");
    ok(eng_mem_parse_size("abc", &is_auto) == -1, "non-numeric refused", NULL);
    ok(eng_mem_parse_size("", &is_auto) == -1, "empty refused", NULL);
    ok(eng_mem_parse_size(NULL, &is_auto) == -1, "NULL refused", NULL);
    ok(eng_mem_parse_size("-5G", &is_auto) == -1, "negative refused", NULL);
    ok(eng_mem_parse_size("5G junk", &is_auto) == -1, "trailing junk refused", NULL);
}

/* ---------------------------------------------------------- memory planning -- */

static void test_memory(void)
{
    printf("\n== memory partitioning ==\n");
    EngMemPlan p;
    EngMemNeeds n;
    int ctx = 0;

    memset(&n, 0, sizeof n);
    n.resident = 860000000LL;
    n.activations = 16000000LL;
    n.scratch = 32000000LL;
    n.kv_per_pos = 294912;
    n.context = 2048;
    n.min_weights = 131000000LL;

    ok(eng_mem_plan(&p, 6LL << 30, &n, &ctx) == 0, "6G budget is viable", NULL);
    eng_mem_report(&p, "  ");
    ok(p.ok, "plan marked ok", NULL);
    eqi(ctx, 2048, "context granted in full");
    ok(p.reserve >= (768LL << 20), "reserve at least the floor", NULL);
    ok(p.weights >= n.min_weights, "weights above the minimum", NULL);
    {
        const int64_t sum = p.reserve + p.resident + p.activations + p.scratch +
                            p.kv_cache + p.weights;
        eqi(sum, 6LL << 30, "partition sums to the budget exactly");
    }

    /* A budget below the reserve must be refused, not clamped to nothing. */
    ok(eng_mem_plan(&p, 100LL << 20, &n, &ctx) != 0, "100M budget refused", NULL);
    ok(p.problem[0] != '\0', "refusal explains itself", p.problem);
    ok(!p.ok, "plan marked not ok", NULL);

    /* Zero budget. */
    ok(eng_mem_plan(&p, 0, &n, &ctx) != 0, "zero budget refused", NULL);

    /* CONTEXT REDUCTION, not refusal: a shorter run beats no run, and the caller is
     * told what it actually got. */
    n.context = 40960;                       /* the model's full context */
    ok(eng_mem_plan(&p, 6LL << 30, &n, &ctx) == 0, "large context still plans", NULL);
    {
        char d[96];
        snprintf(d, sizeof d, "asked 40960, got %d", ctx);
        ok(ctx > 0 && ctx < 40960, "context reduced to fit rather than refused", d);
        ok(p.kv_cache == (int64_t)ctx * n.kv_per_pos, "kv_cache matches the granted context", NULL);
    }

    /* A model whose non-streamable weights alone exceed the budget cannot run at all,
     * and must say so precisely rather than reporting a tiny weight budget. */
    n.resident = 20LL << 30;
    n.context = 512;
    ok(eng_mem_plan(&p, 6LL << 30, &n, &ctx) != 0, "resident weights over budget refused", NULL);
    ok(strstr(p.problem, "cannot stream") != NULL, "refusal names the cause", p.problem);

    eqi(eng_mem_max_context(1LL << 30, 294912), (int)((1LL << 30) / 294912),
        "max context arithmetic");
    eqi(eng_mem_max_context(1000, 0), 0, "zero kv cost yields zero context");
}

/* ----------------------------------------------------------------- planner -- */

static void test_planner(void)
{
    printf("\n== planner: Qwen3-8B ==\n");
    EngHwInfo hw;
    EngModelFacts m;
    EngPlan p;
    EngPlanRequest req;

    qwen3_facts(&m);

    /* The reference machine: 2 physical / 4 logical, 8 GB. */
    hw_fake(&hw, 2, 4, 8LL << 30);
    eng_plan_request_init(&req);
    req.memory_budget = 4LL << 30;
    req.context = 2048;

    ok(eng_plan(&p, &hw, &m, &req) == 0, "4G plan is viable", NULL);
    eng_plan_report(&p, &m, "  ");

    ok(p.streaming == 1, "4G forces streaming",
       "4.16 GB of streamable weights does not fit");
    /* THE THREAD POLICY. Q4_K averages 0.58 bytes per weight, which is compute-bound
     * enough for SMT to pay: the measured MXFP4 case reached 2.85x at 4 threads on 2
     * cores. So this model should get the LOGICAL count. */
    eqi(p.threads, 4, "quantized model gets logical cores");
    ok(p.cache_budget == 0, "dense model has no expert cache", NULL);
    ok(p.stream_budget > 0, "stream budget allocated", NULL);
    ok(p.predicted_stream_hit_rate >= 0.0, "hit rate predicted in advance", NULL);
    ok(p.n_notes > 0, "planner explains itself", NULL);

    /* A budget big enough to hold everything must NOT stream. */
    printf("\n== planner: Qwen3-8B, generous budget ==\n");
    req.memory_budget = 16LL << 30;
    ok(eng_plan(&p, &hw, &m, &req) == 0, "16G plan is viable", NULL);
    ok(p.streaming == 0, "16G holds the model resident", NULL);
    ok(p.predicted_stream_hit_rate == 1.0, "resident implies a 100% hit rate", NULL);

    /* Requiring residency at a budget that cannot hold the model must be refused. */
    req.memory_budget = 4LL << 30;
    req.force_stream = 0;
    ok(eng_plan(&p, &hw, &m, &req) != 0, "forced residency refused at 4G", NULL);
    ok(strstr(p.problem, "residency") != NULL, "refusal names the cause", p.problem);

    /* And forcing streaming at a generous budget must be honoured. */
    eng_plan_request_init(&req);
    req.memory_budget = 16LL << 30;
    req.force_stream = 1;
    ok(eng_plan(&p, &hw, &m, &req) == 0 && p.streaming == 1,
       "forced streaming honoured at 16G", NULL);

    /* An explicit thread count overrides the policy. */
    eng_plan_request_init(&req);
    req.memory_budget = 4LL << 30;
    req.threads = 1;
    ok(eng_plan(&p, &hw, &m, &req) == 0 && p.threads == 1,
       "explicit --threads overrides the policy", NULL);

    /* ---- the bandwidth-bound case ---- */
    printf("\n== planner: thread policy by arithmetic intensity ==\n");
    m.bytes_per_weight = 2.0;                 /* pretend the same model were bf16 */
    eng_plan_request_init(&req);
    req.memory_budget = 4LL << 30;
    ok(eng_plan(&p, &hw, &m, &req) == 0, "bf16 variant plans", NULL);
    eqi(p.threads, 2, "bandwidth-bound model gets PHYSICAL cores");

    /* On a machine without SMT the two policies must agree. */
    hw_fake(&hw, 4, 4, 16LL << 30);
    m.bytes_per_weight = 0.58;
    ok(eng_plan(&p, &hw, &m, &req) == 0 && p.threads == 4,
       "no SMT: both policies give the same answer", NULL);

    /* ---- MoE: the expert cache appears ---- */
    printf("\n== planner: Kimi K3 (MoE) ==\n");
    k3_facts(&m);
    hw_fake(&hw, 8, 16, 128LL << 30);
    eng_plan_request_init(&req);
    req.memory_budget = 64LL << 30;
    req.context = 512;

    ok(eng_plan(&p, &hw, &m, &req) == 0, "K3 at 64G is viable", NULL);
    eng_plan_report(&p, &m, "  ");
    ok(p.streaming == 1, "K3 streams at 64G", NULL);
    ok(p.cache_budget > 0, "MoE model gets an expert cache", NULL);
    /* STREAM BEFORE CACHE: the layer stream must get the larger share, because K3
     * measured trunk-first as 1.69x faster than cache-first at a fixed budget. */
    ok(p.stream_budget > p.cache_budget, "stream budget exceeds cache budget",
       "re-read-every-token weights repay memory more reliably");
    /* The cache must still be able to hold one token's working set. */
    ok(p.cache_budget >= (int64_t)(m.topk + 1) * m.expert_bytes,
       "expert cache holds at least topk+1", NULL);
    eqi(p.threads, 8, "K3's bf16 trunk gets physical cores");

    /* K3 at the laptop floor: 8 GB. The model's own documentation says this runs. */
    printf("\n== planner: K3 at the 8 GB floor ==\n");
    hw_fake(&hw, 2, 4, 8LL << 30);
    eng_plan_request_init(&req);
    req.memory_budget = 7LL << 30;
    req.context = 256;
    const int rc = eng_plan(&p, &hw, &m, &req);
    {
        char d[200];
        snprintf(d, sizeof d, "%s", rc == 0 ? "viable" : p.problem);
        /* Either answer is defensible; what is NOT acceptable is planning a run that
         * cannot fit. So assert the plan is self-consistent whichever way it went. */
        ok(rc != 0 || p.mem.weights >= m.max_layer_bytes,
           "8 GB plan either refuses or leaves room for a layer", d);
    }
}

int main(void)
{
    printf("hardware detection, memory partitioning, execution planner\n\n");
    test_hwinfo();
    test_parse();
    test_memory();
    test_planner();

    printf("\n%d checks, %d failures\n", checks, fails);
    if (fails) { printf("PLANNER TESTS FAILED\n"); return 1; }
    printf("PLANNER TESTS PASSED\n");
    return 0;
}
