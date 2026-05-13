#!/usr/bin/env python3
"""
Fixed e2e_benchmark — MLA-aware RadixAttention benchmark.

Key fixes vs original:
1. Two Engine() in one process → each mode in its own subprocess
2. asyncio/uvloop destroyed by Engine.__init__ → recreated after init  
3. apply_mla_eviction_patch deadlock on available_size() → inline simpler patch
4. cache_hit_rate from get_server_info() → per-request meta_info
5. Pressure via --num-system-prompts, not mem-fraction

Usage:
    # Sanity check (no eviction pressure):
    CUDA_VISIBLE_DEVICES=2 python e2e_benchmark.py \\
        --model deepseek-ai/DeepSeek-V2-Lite \\
        --num-prompts 200 --num-system-prompts 20 \\
        --results-dir results_no_pressure

    # Eviction pressure (cache demand > pool):
    CUDA_VISIBLE_DEVICES=2 python e2e_benchmark.py \\
        --model deepseek-ai/DeepSeek-V2-Lite \\
        --num-prompts 300 --num-system-prompts 150 \\
        --results-dir results_pressure
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "benchmark_results"
MIN_SAFE_MEM_FRACTION = 0.50


# ─────────────────────────────────────────────────────────────────
# Inline MLA patch — avoids the deadlock in gpu_validation.py
# ─────────────────────────────────────────────────────────────────

def apply_mla_patch_safe():
    """
    Minimal MLA-aware eviction patch that does NOT call available_size().

    The original patch in gpu_validation.py calls
    allocator.available_size() inside patched_evict(), which acquires
    a lock that is already held by the scheduler → deadlock.

    This version only uses information already available on the RadixCache
    object itself (evictable_size_, pool size from init), never touching
    the allocator during eviction.
    """
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        from mla_radix_cache import MLAModelConfig, MLAEvictionBudget
    except ImportError:
        logger.error("Cannot import mla_radix_cache from src/. Check path.")
        return False

    try:
        from sglang.srt.mem_cache.radix_cache import RadixCache
    except ImportError:
        logger.error("Cannot import SGLang RadixCache.")
        return False

    # Use DeepSeek-V2-Lite defaults
    mla_config = MLAModelConfig.deepseek_v2_lite()
    # compression_ratio ≈ 14x, target_free_ratio ≈ 0.014
    target_free_ratio = max(0.05, 0.20 / mla_config.compression_ratio)

    original_evict = RadixCache.evict

    def patched_evict(self, num_tokens):
        if getattr(self, "disable", False):
            return original_evict(self, num_tokens)

        # Use only RadixCache-internal state — no allocator call
        evictable = getattr(self, "evictable_size_", 0)
        protected = getattr(self, "protected_size_", 0)
        total_cached = evictable + protected

        # Estimate free slots from max_total_num_tokens if available
        # (set by SGLang scheduler on the cache object)
        pool_total = getattr(self, "_mla_pool_total", None)
        if pool_total is None:
            # Try to infer from allocator without calling available_size()
            alloc = getattr(self, "token_to_kv_pool_allocator", None)
            if alloc is not None:
                pool_total = getattr(alloc, "size", None) or getattr(alloc, "total_size", None)
            if pool_total is None:
                pool_total = total_cached + num_tokens  # conservative fallback
            self._mla_pool_total = pool_total

        free_estimate = max(0, pool_total - total_cached)
        total = total_cached + free_estimate
        free_ratio = free_estimate / max(1, total)

        if free_ratio > target_free_ratio:
            # Plenty of space — reduce eviction count
            scale = max(0.1, 1.0 - (free_ratio - target_free_ratio))
            adjusted = max(1, int(num_tokens * scale))
            logger.debug(
                f"MLA evict: requested={num_tokens} → adjusted={adjusted} "
                f"(free_ratio={free_ratio:.3f} > target={target_free_ratio:.3f})"
            )
            return original_evict(self, adjusted)

        return original_evict(self, num_tokens)

    RadixCache.evict = patched_evict
    logger.info(
        f"MLA eviction patch applied (safe, no allocator call). "
        f"compression_ratio={mla_config.compression_ratio:.1f}x  "
        f"target_free_ratio={target_free_ratio:.4f}"
    )
    return True


# ─────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────

def build_prompts(num_prompts: int, num_system_prompts: int, system_prompt_tokens: int):
    TOPICS = [
        "mathematics", "physics", "chemistry", "biology", "history",
        "geography", "literature", "philosophy", "economics", "computer science",
        "medicine", "law", "engineering", "astronomy", "psychology",
        "linguistics", "art", "music", "architecture", "sociology",
        "neuroscience", "climatology", "robotics", "cryptography", "genetics",
    ]
    ROLES = ["expert", "professor", "researcher", "analyst", "specialist",
             "consultant", "advisor", "tutor", "mentor", "scientist"]
    STYLES = ["concise and precise", "detailed and thorough", "simple and clear",
              "technical and rigorous", "friendly and approachable"]

    words_needed = int(system_prompt_tokens / 1.3)

    def make_system_prompt(seed: int) -> str:
        topic = TOPICS[seed % len(TOPICS)]
        role = ROLES[seed % len(ROLES)]
        style = STYLES[seed % len(STYLES)]
        filler = (
            f"You are an expert {role} specialising in {topic}. "
            f"Your communication style is {style}. "
            f"Always provide accurate, well-reasoned responses based on current knowledge. "
            f"When discussing {topic}, draw on your deep expertise and provide examples. "
            f"Support claims with evidence. Flag uncertainty explicitly. "
            f"Prioritise correctness and clarity over brevity. "
            f"You have studied {topic} for over two decades at leading institutions. "
            f"Your answers should reflect the state of the art in {topic}. "
        )
        repeat = max(1, words_needed // len(filler.split()) + 1)
        words = (filler * repeat).split()[:words_needed]
        return " ".join(words) + f" [SYS={seed:04d}]"

    system_prompts = [make_system_prompt(i) for i in range(num_system_prompts)]

    QUESTIONS = [
        "What is the capital of France?",
        "What is 17 multiplied by 23?",
        "Explain photosynthesis in one sentence.",
        "What is the speed of light in m/s?",
        "Who wrote Hamlet?",
        "What is the boiling point of water in Celsius?",
        "Name three planets in our solar system.",
        "What does CPU stand for?",
        "What is the square root of 144?",
        "In what year did World War II end?",
        "What is the chemical symbol for gold?",
        "How many bones are in the human body?",
        "What is the powerhouse of the cell?",
        "Who painted the Mona Lisa?",
        "What is the largest ocean on Earth?",
    ]

    n_shared = num_prompts * 3 // 4
    shared_prompts = [
        f"{system_prompts[i % num_system_prompts]}\n\n"
        f"User: {QUESTIONS[i % len(QUESTIONS)]} (req {i})\nAssistant:"
        for i in range(n_shared)
    ]
    n_unique = num_prompts - n_shared
    unique_prompts = [
        f"StandaloneQuery_{i*137}: Define entropy in thermodynamics briefly.\n"
        for i in range(n_unique)
    ]

    all_prompts = shared_prompts + unique_prompts
    shared_set = set(range(len(shared_prompts)))

    bytes_per_token = 576 * 2 * 27
    cache_demand_gb = (num_system_prompts * system_prompt_tokens * bytes_per_token) / (1024**3)
    logger.info(
        f"Workload: {len(shared_prompts)} shared + {len(unique_prompts)} unique | "
        f"{num_system_prompts} sys_prompts × ~{system_prompt_tokens} tok "
        f"= ~{cache_demand_gb:.2f} GB cache demand"
    )
    return all_prompts, shared_set, cache_demand_gb


# ─────────────────────────────────────────────────────────────────
# Single-mode runner
# ─────────────────────────────────────────────────────────────────

def run_single_mode(
    mode: str,
    model_path: str,
    num_prompts: int,
    mem_fraction: float,
    results_dir: str,
    num_system_prompts: int = 20,
    system_prompt_tokens: int = 800,
    max_new_tokens: int = 32,
):
    import asyncio

    try:
        from sglang import Engine
    except ImportError:
        logger.error("SGLang not installed.")
        sys.exit(1)

    logger.info(f"{'='*60}")
    logger.info(f"Mode: {mode.upper()}  mem={mem_fraction}  prompts={num_prompts}  sys={num_system_prompts}")
    logger.info(f"{'='*60}")

    all_prompts, shared_set, cache_demand_gb = build_prompts(
        num_prompts, num_system_prompts, system_prompt_tokens
    )

    # Apply patch BEFORE Engine init so RadixCache class is patched
    # before any instance is created
    if mode == "patched":
        ok = apply_mla_patch_safe()
        if not ok:
            logger.error("Patch failed, aborting patched run.")
            sys.exit(1)

    logger.info("Initialising SGLang Engine...")
    engine = Engine(
        model_path=model_path,
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction,
    )
    logger.info("Engine ready.")

    # Fix event loop destroyed by Engine.__init__
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    import sglang.srt.entrypoints.engine as _eng_mod
    _eng_mod.asyncio.get_event_loop = lambda: _loop
    logger.info("Event loop restored.")

    # Warmup
    logger.info("Warmup...")
    for p in all_prompts[:10]:
        engine.generate(p, sampling_params={"max_new_tokens": 16, "temperature": 0})
    logger.info("Warmup done.")

    # Benchmark
    logger.info(f"Benchmarking {len(all_prompts)} prompts...")
    ttfts_shared, ttfts_unique = [], []
    total_prompt_tokens = 0
    total_cached_tokens = 0
    start_total = time.perf_counter()

    for idx, prompt in enumerate(all_prompts):
        t0 = time.perf_counter()
        out = engine.generate(
            prompt,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        meta = out.get("meta_info", {})
        total_prompt_tokens += meta.get("prompt_tokens", 0)
        total_cached_tokens += meta.get("cached_tokens", 0)

        if idx in shared_set:
            ttfts_shared.append(elapsed_ms)
        else:
            ttfts_unique.append(elapsed_ms)

        if (idx + 1) % 50 == 0:
            hit = total_cached_tokens / max(1, total_prompt_tokens)
            logger.info(f"  {idx+1}/{len(all_prompts)} | cache_hit={hit:.3f}")

    total_time = time.perf_counter() - start_total
    cache_hit_rate = total_cached_tokens / max(1, total_prompt_tokens)
    logger.info(f"cache_hit={cache_hit_rate:.4f} ({total_cached_tokens}/{total_prompt_tokens})")

    engine.shutdown()
    logger.info("Engine shut down.")

    def pct(lst, p):
        if not lst:
            return 0.0
        s = sorted(lst)
        return s[min(int(len(s) * p / 100), len(s) - 1)]

    all_ttfts = ttfts_shared + ttfts_unique
    result = {
        "mode": mode,
        "model": model_path,
        "num_prompts": len(all_prompts),
        "num_system_prompts": num_system_prompts,
        "system_prompt_tokens": system_prompt_tokens,
        "cache_demand_gb": round(cache_demand_gb, 3),
        "mem_fraction": mem_fraction,
        "total_time_s": round(total_time, 3),
        "throughput_req_s": round(len(all_prompts) / total_time, 3),
        "ttft_avg_ms": round(statistics.mean(all_ttfts), 2),
        "ttft_p50_ms": round(pct(all_ttfts, 50), 2),
        "ttft_p95_ms": round(pct(all_ttfts, 95), 2),
        "ttft_p99_ms": round(pct(all_ttfts, 99), 2),
        "ttft_shared_avg_ms": round(statistics.mean(ttfts_shared), 2) if ttfts_shared else 0,
        "ttft_shared_p50_ms": round(pct(ttfts_shared, 50), 2),
        "ttft_shared_p99_ms": round(pct(ttfts_shared, 99), 2),
        "ttft_unique_avg_ms": round(statistics.mean(ttfts_unique), 2) if ttfts_unique else 0,
        "ttft_unique_p99_ms": round(pct(ttfts_unique, 99), 2),
        "cache_hit_rate": round(cache_hit_rate, 4),
        "total_cached_tokens": total_cached_tokens,
        "total_prompt_tokens": total_prompt_tokens,
    }

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"{mode}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved → {out_file}")
    logger.info(f"Throughput={result['throughput_req_s']:.2f} req/s  "
                f"TTFT={result['ttft_avg_ms']:.1f}ms  cache_hit={cache_hit_rate:.4f}")


# ─────────────────────────────────────────────────────────────────
# Comparison printer
# ─────────────────────────────────────────────────────────────────

def compare_results(results_dir: str):
    b_file = os.path.join(results_dir, "baseline.json")
    p_file = os.path.join(results_dir, "patched.json")

    if not os.path.exists(b_file) or not os.path.exists(p_file):
        logger.error(f"Missing files in {results_dir}/")
        return

    with open(b_file) as f:
        b = json.load(f)
    with open(p_file) as f:
        p = json.load(f)

    def row(label, key, fmt="{:.2f}", higher_is_better=True):
        bv = b.get(key, 0)
        pv = p.get(key, 0)
        if bv == 0:
            print(f"  {label:<38} {'N/A':>10} {fmt.format(pv):>10} {'N/A':>9}    ~")
            return
        delta = (pv - bv) / bv * 100
        d_str = f"{delta:+.1f}%"
        sym = ("✓" if delta > 1 else ("✗" if delta < -1 else "~")) if higher_is_better \
              else ("✓" if delta < -1 else ("✗" if delta > 1 else "~"))
        print(f"  {label:<38} {fmt.format(bv):>10} {fmt.format(pv):>10} {d_str:>9}    {sym}")

    w = 78
    print("=" * w)
    print("  Engine Benchmark — MLA-aware RadixAttention")
    print(f"  Model: {b.get('model','?')}")
    print(f"  Prompts: {b.get('num_prompts')}  mem: {b.get('mem_fraction')}  "
          f"sys_prompts: {b.get('num_system_prompts')}  "
          f"cache_demand: {b.get('cache_demand_gb','?')} GB")
    print("=" * w)
    print(f"  {'Metric':<38} {'Baseline':>10} {'Patched':>10} {'Change':>9}")
    print(f"  {'─'*w}")
    row("Throughput (req/s)",           "throughput_req_s",   "{:.2f}", True)
    print("  ── Overall ──────────────────────────")
    row("TTFT avg (ms)",                "ttft_avg_ms",        "{:.2f}", False)
    row("TTFT P50 (ms)",                "ttft_p50_ms",        "{:.2f}", False)
    row("TTFT P95 (ms)",                "ttft_p95_ms",        "{:.2f}", False)
    row("TTFT P99 (ms)",                "ttft_p99_ms",        "{:.2f}", False)
    print("  ── Shared-prefix (primary target) ───")
    row("TTFT shared avg (ms) ★",      "ttft_shared_avg_ms", "{:.2f}", False)
    row("TTFT shared P50 (ms)",         "ttft_shared_p50_ms", "{:.2f}", False)
    row("TTFT shared P99 (ms)",         "ttft_shared_p99_ms", "{:.2f}", False)
    print("  ── Unique requests (regression) ─────")
    row("TTFT unique avg (ms)",         "ttft_unique_avg_ms", "{:.2f}", False)
    row("TTFT unique P99 (ms)",         "ttft_unique_p99_ms", "{:.2f}", False)
    print("  ── Cache ────────────────────────────")
    row("Cache hit rate ★",             "cache_hit_rate",     "{:.4f}", True)
    row("Total time (s)",               "total_time_s",       "{:.2f}", False)

    b_hit = b.get("cache_hit_rate", 0)
    p_hit = p.get("cache_hit_rate", 0)
    demand = b.get("cache_demand_gb", 0)
    print("=" * w)
    print(f"  cache_hit: baseline={b_hit:.4f}  patched={p_hit:.4f}  "
          f"cache_demand={demand:.2f}GB")
    if abs(b_hit - p_hit) < 0.005:
        print("  → No eviction triggered. Increase --num-system-prompts.")
    elif p_hit > b_hit:
        print("  → Patch working: patched kept more prefixes cached ✓")
    else:
        print("  → Patch may be too aggressive (patched hit rate lower).")
    print("=" * w)


# ─────────────────────────────────────────────────────────────────
# Subprocess orchestrator
# ─────────────────────────────────────────────────────────────────

def run_via_subprocess(mode, model, num_prompts, mem_fraction, results_dir,
                       cuda_devices, num_system_prompts, system_prompt_tokens):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    cmd = [
        sys.executable, __file__,
        "--run-mode", mode,
        "--model", model,
        "--num-prompts", str(num_prompts),
        "--mem-fraction", str(mem_fraction),
        "--num-system-prompts", str(num_system_prompts),
        "--system-prompt-tokens", str(system_prompt_tokens),
        "--results-dir", results_dir,
    ]
    logger.info(f"Spawning [{mode}]")
    start = time.time()
    proc = subprocess.Popen(cmd, env=env)
    proc.wait()
    elapsed = time.time() - start
    if proc.returncode != 0:
        logger.error(f"[{mode}] failed (code {proc.returncode})")
        return False
    logger.info(f"[{mode}] done in {elapsed/60:.1f} min")
    return True


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    parser.add_argument("--num-system-prompts", type=int, default=20,
                        help="Distinct system prompts. Controls cache pressure. "
                             "Try 150 for H100 80GB with mem=0.85.")
    parser.add_argument("--system-prompt-tokens", type=int, default=800,
                        help="Approx tokens per system prompt (shorter = faster runs)")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--cuda-devices", default="0")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--run-mode", choices=["baseline", "patched"])
    args = parser.parse_args()

    if args.mem_fraction < MIN_SAFE_MEM_FRACTION and args.run_mode is None:
        logger.warning(f"mem-fraction {args.mem_fraction} < safe min {MIN_SAFE_MEM_FRACTION}")
        if input("Continue? [y/N] ").strip().lower() != "y":
            sys.exit(0)

    if args.run_mode:
        run_single_mode(
            mode=args.run_mode,
            model_path=args.model,
            num_prompts=args.num_prompts,
            mem_fraction=args.mem_fraction,
            results_dir=args.results_dir,
            num_system_prompts=args.num_system_prompts,
            system_prompt_tokens=args.system_prompt_tokens,
        )
        return

    if args.compare_only:
        compare_results(args.results_dir)
        return

    os.makedirs(args.results_dir, exist_ok=True)
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", args.cuda_devices)

    for mode in ["baseline", "patched"]:
        ok = run_via_subprocess(
            mode=mode, model=args.model, num_prompts=args.num_prompts,
            mem_fraction=args.mem_fraction, results_dir=args.results_dir,
            cuda_devices=cuda, num_system_prompts=args.num_system_prompts,
            system_prompt_tokens=args.system_prompt_tokens,
        )
        if not ok:
            logger.error(f"Aborting after [{mode}] failure.")
            sys.exit(1)
        logger.info("Waiting 15s for GPU memory to release...")
        time.sleep(15)

    compare_results(args.results_dir)


if __name__ == "__main__":
    main()
