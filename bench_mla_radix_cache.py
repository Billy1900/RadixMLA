"""
Benchmarks for MLA-aware RadixCache.

Compares MLA-aware cache (this project) vs baseline cache behavior
across various workload patterns:
1. System prompt reuse (chat serving)
2. Few-shot prompting (batch inference)
3. Multi-turn conversation (growing context)
4. Random (no prefix sharing — regression test)

Run: python benchmarks/bench_mla_radix_cache.py
"""

import sys
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from mla_radix_cache import (
    MLAModelConfig,
    MLARadixCache,
    LatentCacheAnalyzer,
    MLAEvictionBudget,
)


@dataclass
class BenchmarkResult:
    name: str
    num_requests: int
    hit_rate: float
    token_reuse_rate: float
    total_tokens_cached: int
    peak_cached_tokens: int
    evictions: int
    wall_time_ms: float
    ops_per_sec: float


def generate_chat_workload(
    num_requests: int,
    system_prompt_len: int = 200,
    user_msg_len: int = 50,
    num_system_prompts: int = 3,
) -> List[Tuple[List[int], torch.Tensor]]:
    """Generate chat-like workload with shared system prompts."""
    system_prompts = [
        list(range(i * 10000, i * 10000 + system_prompt_len))
        for i in range(num_system_prompts)
    ]

    workload = []
    for i in range(num_requests):
        sp = system_prompts[i % num_system_prompts]
        user = list(range(50000 + i * user_msg_len, 50000 + (i + 1) * user_msg_len))
        tokens = sp + user
        slots = torch.arange(len(tokens), dtype=torch.int64)
        workload.append((tokens, slots))

    return workload


def generate_fewshot_workload(
    num_requests: int,
    num_examples: int = 5,
    example_len: int = 100,
    query_len: int = 30,
) -> List[Tuple[List[int], torch.Tensor]]:
    """Generate few-shot prompting workload with shared examples."""
    examples = list(range(1000, 1000 + num_examples * example_len))

    workload = []
    for i in range(num_requests):
        query = list(range(80000 + i * query_len, 80000 + (i + 1) * query_len))
        tokens = examples + query
        slots = torch.arange(len(tokens), dtype=torch.int64)
        workload.append((tokens, slots))

    return workload


def generate_random_workload(
    num_requests: int,
    seq_len: int = 100,
) -> List[Tuple[List[int], torch.Tensor]]:
    """Generate random workload (no prefix sharing)."""
    workload = []
    for i in range(num_requests):
        tokens = [random.randint(0, 100000) for _ in range(seq_len)]
        slots = torch.arange(seq_len, dtype=torch.int64)
        workload.append((tokens, slots))
    return workload


def generate_multiturn_workload(
    num_conversations: int,
    turns_per_conversation: int = 5,
    system_len: int = 100,
    msg_len: int = 40,
) -> List[Tuple[List[int], torch.Tensor]]:
    """Generate multi-turn conversation workload."""
    workload = []
    for conv in range(num_conversations):
        context = list(range(conv * 10000, conv * 10000 + system_len))
        for turn in range(turns_per_conversation):
            new_msg = list(range(
                60000 + conv * 1000 + turn * msg_len,
                60000 + conv * 1000 + (turn + 1) * msg_len,
            ))
            context = context + new_msg
            slots = torch.arange(len(context), dtype=torch.int64)
            workload.append((context[:], slots))
    return workload


def run_benchmark(
    name: str,
    config: MLAModelConfig,
    workload: List[Tuple[List[int], torch.Tensor]],
    pool_size: int = 100000,
    page_size: int = 1,
) -> BenchmarkResult:
    """Run a benchmark on a given workload."""
    cache = MLARadixCache(config, pool_size=pool_size, page_size=page_size)

    start = time.perf_counter()
    for tokens, slots in workload:
        result = cache.match_prefix(tokens)
        cache.insert(tokens, slots)

    elapsed = (time.perf_counter() - start) * 1000  # ms

    stats = cache.stats
    return BenchmarkResult(
        name=name,
        num_requests=len(workload),
        hit_rate=stats.hit_rate,
        token_reuse_rate=stats.token_reuse_rate,
        total_tokens_cached=stats.cached_tokens,
        peak_cached_tokens=stats.peak_cached_tokens,
        evictions=stats.evicted_tokens,
        wall_time_ms=elapsed,
        ops_per_sec=len(workload) / (elapsed / 1000) if elapsed > 0 else 0,
    )


def run_eviction_benchmark(
    name: str,
    config: MLAModelConfig,
    workload: List[Tuple[List[int], torch.Tensor]],
    pool_size: int = 5000,  # small pool to force evictions
) -> BenchmarkResult:
    """Run benchmark with a small pool to test MLA-aware eviction."""
    cache = MLARadixCache(config, pool_size=pool_size)

    start = time.perf_counter()
    for tokens, slots in workload:
        # Check if we need to evict
        free = pool_size - cache.total_size()
        if free < len(tokens):
            cache.evict(len(tokens) - free)

        result = cache.match_prefix(tokens)
        cache.insert(tokens, slots)

    elapsed = (time.perf_counter() - start) * 1000

    stats = cache.stats
    return BenchmarkResult(
        name=name,
        num_requests=len(workload),
        hit_rate=stats.hit_rate,
        token_reuse_rate=stats.token_reuse_rate,
        total_tokens_cached=stats.cached_tokens,
        peak_cached_tokens=stats.peak_cached_tokens,
        evictions=stats.evicted_tokens,
        wall_time_ms=elapsed,
        ops_per_sec=len(workload) / (elapsed / 1000) if elapsed > 0 else 0,
    )


def print_result(r: BenchmarkResult):
    print(f"  {'Hit rate:':<25} {r.hit_rate:.3f}")
    print(f"  {'Token reuse rate:':<25} {r.token_reuse_rate:.3f}")
    print(f"  {'Cached tokens:':<25} {r.total_tokens_cached}")
    print(f"  {'Peak cached:':<25} {r.peak_cached_tokens}")
    print(f"  {'Evictions:':<25} {r.evictions}")
    print(f"  {'Wall time:':<25} {r.wall_time_ms:.1f} ms")
    print(f"  {'Throughput:':<25} {r.ops_per_sec:.0f} req/s")


def main():
    print("=" * 70)
    print("MLA-aware RadixCache Benchmark")
    print("=" * 70)

    configs = {
        "DeepSeek-V3": MLAModelConfig.deepseek_v3(),
        "DeepSeek-V2-Lite": MLAModelConfig.deepseek_v2_lite(),
    }

    for config_name, config in configs.items():
        print(f"\n{'─' * 70}")
        print(f"Model: {config_name}")
        print(f"  Latent dim: {config.latent_dim}")
        print(f"  Compression ratio: {config.compression_ratio:.1f}x")
        print(f"  Bytes/token/layer: {config.bytes_per_token_per_layer}")
        print(f"{'─' * 70}")

        # Analyzer report
        analyzer = LatentCacheAnalyzer(config)
        rec = analyzer.recommend_pool_size(
            gpu_memory_bytes=80 * 1024 ** 3,
            mem_fraction_static=0.8,
            model_weight_bytes=40 * 1024 ** 3,
        )
        print(f"\n  Memory analysis (80GB GPU, 40GB weights):")
        print(f"    MLA tokens: {rec['mla_max_tokens']:,}")
        print(f"    MHA tokens: {rec['mha_max_tokens']:,}")
        print(f"    Extra from MLA: {rec['extra_tokens_from_mla']:,}")

        # Workloads
        workloads = [
            ("Chat (shared system prompts)", generate_chat_workload(1000)),
            ("Few-shot (shared examples)", generate_fewshot_workload(1000)),
            ("Multi-turn (growing context)", generate_multiturn_workload(100, 5)),
            ("Random (no sharing)", generate_random_workload(1000)),
        ]

        print(f"\n  {'Workload':<35} {'Hit Rate':>10} {'Reuse':>10} {'Cached':>10} {'Time':>10}")
        print(f"  {'─' * 75}")

        for wl_name, wl_data in workloads:
            result = run_benchmark(wl_name, config, wl_data)
            print(
                f"  {wl_name:<35} "
                f"{result.hit_rate:>9.3f} "
                f"{result.token_reuse_rate:>9.3f} "
                f"{result.total_tokens_cached:>9,} "
                f"{result.wall_time_ms:>8.1f}ms"
            )

        # Eviction benchmark
        print(f"\n  Eviction benchmarks (small pool = 5000 tokens):")
        print(f"  {'Workload':<35} {'Hit Rate':>10} {'Evicted':>10} {'Time':>10}")
        print(f"  {'─' * 65}")

        for wl_name, wl_data in workloads:
            result = run_eviction_benchmark(
                wl_name + " (eviction)", config, wl_data, pool_size=5000
            )
            print(
                f"  {wl_name:<35} "
                f"{result.hit_rate:>9.3f} "
                f"{result.evictions:>9,} "
                f"{result.wall_time_ms:>8.1f}ms"
            )

    print(f"\n{'=' * 70}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
