# MLA-aware RadixAttention

Optimized prefix caching for Multi-Head Latent Attention (MLA) models in SGLang.

## Problem

SGLang's `RadixCache` eviction policy treats every cached token identically, regardless of its actual memory footprint. For MLA models (DeepSeek V2/V3/V4/R1), each token stores a compressed latent vector (`kv_lora_rank + qk_rope_head_dim = 576` elements) instead of full K/V heads (`128 × 320 = 40,960` elements). This ~71× compression means the cache can hold dramatically more prefix tokens, but the eviction policy doesn't exploit this — it evicts too aggressively based on MHA-era thresholds.

## What This Does

This project makes SGLang's prefix cache **MLA-aware**:

1. **Eviction Budget** — Adjusts eviction thresholds based on MLA compression ratio. Instead of maintaining 20% free space (MHA assumption), MLA can operate at 95%+ utilization since tokens are ~14-71× cheaper.

2. **Cache Pressure Metrics** — Reports effective cache pressure accounting for actual per-token memory cost, not just token count.

3. **Capacity Estimation** — Computes how many prefix tokens fit in a given GPU memory budget, accounting for MLA's latent-vector-only storage.

## Key Finding from SGLang Code Archaeology

SGLang already stores latent vectors, not expanded K/V:

```
MLATokenToKVPool.kv_buffer shape: [pool_size, 1, kv_lora_rank + qk_rope_head_dim]
                                   └─── e.g., [pool_size, 1, 576] for DeepSeek V3
```

The optimization target is the **RadixCache eviction policy**, not the storage format.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SGLang Scheduler                       │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              RadixCache (existing)                   │ │
│  │  ┌──────────────────────────────────────────────┐   │ │
│  │  │     MLAEvictionBudget (this project)          │   │ │
│  │  │  - Adjusts eviction thresholds               │   │ │
│  │  │  - Compression-ratio-aware free space target  │   │ │
│  │  └──────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
│                         │                                 │
│  ┌─────────────────────────────────────────────────────┐ │
│  │         MLATokenToKVPool (existing SGLang)           │ │
│  │  kv_buffer: [n_layers][pool_size, 1, 576]           │ │
│  │  ↑ Already stores latent vectors, not expanded K/V  │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
mla-radix-attention/
├── src/
│   ├── mla_radix_cache.py       # Core: MLARadixCache, MLAEvictionBudget, LatentCacheAnalyzer
│   └── sglang_integration.py    # SGLang patches: detect_mla_config, patch_scheduler
├── tests/
│   └── test_mla_radix_cache.py  # 35 tests (all passing)
├── benchmarks/
│   └── bench_mla_radix_cache.py # CPU workload benchmarks
├── patches/
│   └── sglang_mla_eviction.py   # SGLang PR diff generator
├── gpu_validation.py            # Phase 3: correctness validation (baseline vs patched)
├── e2e_benchmark.py             # Phase 4: end-to-end TTFT/throughput benchmark
├── launch_patched_server.py     # Patched SGLang server launcher
├── run_all.sh                   # All-in-one runner
└── README.md
```

## Quick Start

```bash
# Run tests
pip install pytest torch
python -m pytest tests/ -v

# Run benchmarks
python benchmarks/bench_mla_radix_cache.py
```

## Standalone Usage

```python
from src.mla_radix_cache import MLARadixCache, MLAModelConfig
import torch

config = MLAModelConfig.deepseek_v3()
cache = MLARadixCache(config, pool_size=100000)

# Insert
slots = torch.arange(100, dtype=torch.int64)
cache.insert(list(range(100)), slots)

# Match
result = cache.match_prefix(list(range(50)) + [999, 998])
print(f"Hit: {result.matched_len} tokens")  # 50

# Analyze
from src.mla_radix_cache import LatentCacheAnalyzer
analyzer = LatentCacheAnalyzer(config)
print(analyzer.recommend_pool_size(
    gpu_memory_bytes=80 * 1024**3,
    model_weight_bytes=40 * 1024**3,
))
```

## SGLang Integration

The minimal patch to SGLang's `radix_cache.py` adds MLA-aware eviction scoring. See `patches/sglang_mla_eviction.py` for the exact diff.

```python
# In SGLang's scheduler init:
from src.sglang_integration import detect_mla_config, patch_scheduler_for_mla

mla_config = detect_mla_config(model_config)
if mla_config:
    patch_scheduler_for_mla(scheduler, mla_config)
```

## Development Plan

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Done | Code archaeology — all key SGLang files identified |
| Phase 1 | ✅ Done | Standalone MLARadixCache with tests (35/35 passing) |
| Phase 2 | ✅ Done | SGLang integration module + patch generator |
| Phase 3 | ✅ Done | GPU validation script (correctness comparison baseline vs patched) |
| Phase 4 | ✅ Done | End-to-end benchmark (TTFT, throughput, prefix cache hit rate) |

## Running on GPU

### Prerequisites
```bash
pip install 'sglang[all]' torch
# Requires NVIDIA GPU with ≥40GB VRAM (A100/H100)
```

### Option A: All-in-one
```bash
chmod +x run_all.sh
./run_all.sh
```

### Option B: Step-by-step

```bash
# Phase 3 — Correctness validation (Engine API, no server needed)
python gpu_validation.py --mode validate --model deepseek-ai/DeepSeek-V2-Lite

# Phase 4 — End-to-end benchmark
python e2e_benchmark.py --mode engine --model deepseek-ai/DeepSeek-V2-Lite --num-prompts 200

# Phase 4 — Server-based benchmark (more realistic)
# Terminal 1: launch patched server
python launch_patched_server.py --model deepseek-ai/DeepSeek-V2-Lite --tp 1
# Terminal 2: run bench_serving
python -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 500
```

### Option C: Manual server comparison
```bash
# Baseline
python -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Lite --tp 1 --port 30000
python e2e_benchmark.py --mode server --experiment sharegpt --port 30000

# Patched
python launch_patched_server.py --model deepseek-ai/DeepSeek-V2-Lite --tp 1 --port 30000
python e2e_benchmark.py --mode server --experiment sharegpt --port 30000

# Compare
python e2e_benchmark.py --mode report
```

## License

Apache 2.0
