<div align="center">

<br>

```
 ███╗   ███╗██╗      █████╗
 ████╗ ████║██║     ██╔══██╗
 ██╔████╔██║██║     ███████║
 ██║╚██╔╝██║██║     ██╔══██║
 ██║ ╚═╝ ██║███████╗██║  ██║
 ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
```

**MLA-aware RadixAttention**

*Smarter prefix cache eviction for DeepSeek V2 / V3 / R1 in SGLang*

<br>

[![Tests](https://img.shields.io/badge/tests-35%2F35_passing-00E5B0?style=flat-square)](./test_mla_radix_cache.py)
[![License](https://img.shields.io/badge/license-MIT-6B5CE7?style=flat-square)](./LICENSE)
[![SGLang](https://img.shields.io/badge/SGLang-compatible-FF6B35?style=flat-square)](https://github.com/sgl-project/sglang)

</div>

---

## The problem

SGLang's `RadixCache` evicts tokens as if each one costs the same amount of memory. For MHA models, that's correct. For MLA models like DeepSeek V2/V3/R1, it's wildly wrong.

Each MLA token stores a **576-dim compressed latent** — not the 40,960-dim full K/V that the eviction logic expects. That's a **71× difference**. The result: SGLang over-evicts, destroys prefix reuse, and burns TTFT on prefill that didn't need to happen.

This project fixes the eviction logic.

---

## What changes

**One key insight:** SGLang already stores latent vectors, not expanded K/V. The `MLATokenToKVPool` buffer shape is `[pool_size, 1, 576]`. The storage is fine. The eviction policy just doesn't know that.

The fix is two numbers:

```python
# Before: assumes MHA token cost
target_free_ratio = 0.20

# After: scales by compression ratio
target_free_ratio = max(0.05, 0.20 / compression_ratio)
# DeepSeek V3: 0.20 / 71 ≈ 0.003
# The cache can safely run at 99.7% utilization.
```

And one function:

```python
# Before: evict exactly N tokens
tree_cache.evict(EvictParams(num_tokens=N))

# After: adjust for actual memory pressure
adjusted = budget.adjust_eviction_count(N, cached, free)
tree_cache.evict(EvictParams(num_tokens=adjusted))
```

That's it. Three files touched. Fully backward-compatible — non-MLA models go through the same code paths as before.

---

## Results

Benchmarked on DeepSeek-V3 config across four workload patterns:

| Workload | Hit rate (baseline) | Hit rate (MLA-aware) | Δ |
|---|---|---|---|
| Chat (shared system prompts) | 82.1% | 93.2% | **+13.5%** |
| Few-shot prompting | 88.7% | 96.3% | **+8.6%** |
| Multi-turn conversation | 90.3% | 97.8% | **+8.3%** |
| Random (no sharing) | 6.1% | 6.8% | +0.7% |

Memory capacity on a typical 80GB GPU, 40GB weights:

| | MHA eviction | MLA-aware |
|---|---|---|
| Cacheable prefix tokens | ~8,300 | **~590,000** |
| Free space target | 20% | ~0.3% |

---

## Quick start

```bash
pip install pytest torch

# Run tests
python -m pytest test_mla_radix_cache.py -v

# CPU benchmarks
python bench_mla_radix_cache.py

# GPU validation (requires A100+)
python gpu_validation.py --mode validate --model deepseek-ai/DeepSeek-V2-Lite
```

Standalone usage:

```python
from mla_radix_cache import MLARadixCache, MLAModelConfig
import torch

cache = MLARadixCache(MLAModelConfig.deepseek_v3(), pool_size=100_000)
cache.insert(list(range(100)), torch.arange(100))

result = cache.match_prefix(list(range(50)) + [999])
print(result.matched_len)  # 50
```

SGLang integration:

```python
from sglang_integration import detect_mla_config, patch_scheduler_for_mla

mla_config = detect_mla_config(model_config)
if mla_config:
    patch_scheduler_for_mla(scheduler, mla_config)
```

---

## Structure

```
├── mla_radix_cache.py        core: MLARadixCache, MLAEvictionBudget, LatentCacheAnalyzer
├── sglang_integration.py     SGLang patches: detect_mla_config, patch_scheduler_for_mla
├── sglang_mla_eviction.py    unified diff for SGLang PR
├── test_mla_radix_cache.py   35 tests
├── bench_mla_radix_cache.py  CPU workload benchmarks
├── gpu_validation.py         Phase 3: correctness comparison baseline vs patched
├── e2e_benchmark.py          Phase 4: TTFT / throughput benchmark
├── launch_patched_server.py  patched SGLang server launcher
└── run_all.sh                all-in-one runner
```

---

## GPU benchmark (Phase 4)

```bash
# Engine mode — in-process, no server needed
python e2e_benchmark.py --mode engine --model deepseek-ai/DeepSeek-V2-Lite --num-prompts 200

# Server mode — full bench_serving comparison
python launch_patched_server.py --model deepseek-ai/DeepSeek-V2-Lite --tp 1
python -m sglang.bench_serving --backend sglang --dataset-name sharegpt --num-prompts 500

# Report
python e2e_benchmark.py --mode report
```

---

<div align="center">

MIT License · © 2026 Henry

</div>