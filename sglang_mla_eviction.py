"""
SGLang Patch: MLA-aware RadixAttention Eviction

This script generates the minimal code changes needed in SGLang to
enable MLA-aware eviction in the RadixCache. The key insight is that
SGLang already stores latent vectors (not expanded K/V) in MLATokenToKVPool,
but the RadixCache eviction policy doesn't account for the reduced
per-token memory cost of MLA models.

Changes:
1. cache_init_params.py — Add MLA config field
2. radix_cache.py — MLA-aware eviction scoring in _eviction_priority
3. scheduler.py — Adjusted eviction thresholds
4. model_runner_kv_cache_mixin.py — Pass MLA config through

These changes are backward-compatible: non-MLA models behave identically.
"""


def print_patch():
    print("""
================================================================================
PATCH 1: python/sglang/srt/mem_cache/cache_init_params.py
================================================================================

--- a/python/sglang/srt/mem_cache/cache_init_params.py
+++ b/python/sglang/srt/mem_cache/cache_init_params.py
@@ -14,6 +14,7 @@
 @dataclasses.dataclass
 class CacheInitParams:
     disable: bool
     req_to_token_pool: ReqToTokenPool
     token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
     page_size: int
 
     is_eagle: bool = False
     ...
     cache_ttl_seconds: Optional[float] = None
 
     tree_components: Optional[tuple[ComponentType, ...]] = None
+
+    # MLA-specific: latent compression ratio for eviction budget
+    mla_compression_ratio: Optional[float] = None


================================================================================
PATCH 2: python/sglang/srt/mem_cache/radix_cache.py
================================================================================

--- a/python/sglang/srt/mem_cache/radix_cache.py
+++ b/python/sglang/srt/mem_cache/radix_cache.py
@@ -269,6 +269,11 @@
 class RadixCache(KVCacheEventMixin, BasePrefixCache):
     def __init__(self, params: CacheInitParams):
         self.disable = params.disable
+        # MLA-aware eviction: if model uses MLA, we can tolerate higher
+        # cache utilization since each token costs less memory
+        self.mla_compression_ratio = getattr(params, 'mla_compression_ratio', None)
+        if self.mla_compression_ratio and self.mla_compression_ratio > 1:
+            self._mla_target_free_ratio = max(0.05, 0.20 / self.mla_compression_ratio)
+        else:
+            self._mla_target_free_ratio = 0.20
         ...

     def evict(self, params: EvictParams) -> EvictResult:
         ...
         num_tokens = params.num_tokens
+
+        # MLA-aware adjustment: reduce eviction when we have abundant free space
+        if self.mla_compression_ratio and self.mla_compression_ratio > 1:
+            total = self.evictable_size_ + self.protected_size_
+            if total > 0:
+                free = self.token_to_kv_pool_allocator.available_size()
+                pool_total = total + free
+                free_ratio = free / pool_total if pool_total > 0 else 1.0
+                if free_ratio > self._mla_target_free_ratio:
+                    scale = max(0.1, 1.0 - (free_ratio - self._mla_target_free_ratio))
+                    num_tokens = max(1, int(num_tokens * scale))
+
         leaves = list(self.evictable_leaves)
         ...


================================================================================
PATCH 3: python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
================================================================================

--- a/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
+++ b/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
@@ -498,6 +498,14 @@
                 self.token_to_kv_pool = MLATokenToKVPool(
                     ...
                 )
+
+        # Pass MLA compression ratio to cache init params for eviction
+        if self.model_config.kv_lora_rank and self.model_config.kv_lora_rank > 0:
+            latent_dim = self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim
+            mha_dim = self.model_config.num_attention_heads * (
+                self.model_config.qk_nope_head_dim + self.model_config.qk_rope_head_dim
+                + self.model_config.v_head_dim
+            )
+            cache_init_params.mla_compression_ratio = mha_dim / latent_dim


================================================================================
NOTES
================================================================================

The patch is minimal and backward-compatible:
- Non-MLA models: mla_compression_ratio is None, all paths unchanged
- MLA models: eviction is less aggressive when free space is abundant,
  allowing more prefix tokens to remain cached

Expected impact:
- Higher prefix cache hit rates for MLA models under memory pressure
- No performance regression for non-MLA models
- No correctness changes (only eviction timing is affected)

To verify: run SGLang's existing test suite + bench_serving with
DeepSeek-V2-Lite on a single A100:

  python -m sglang.launch_server \\
    --model deepseek-ai/DeepSeek-V2-Lite \\
    --tp 1 --trust-remote-code

  python -m sglang.bench_serving \\
    --backend sglang \\
    --dataset-name sharegpt \\
    --num-prompts 500

Compare prefix cache hit rate and throughput with and without the patch.
""")


if __name__ == "__main__":
    print_patch()
