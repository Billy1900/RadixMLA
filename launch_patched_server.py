#!/usr/bin/env python3
"""
Launch SGLang server with MLA-aware eviction patch applied.

This wraps sglang.launch_server and patches evict_from_tree_cache
before the server starts accepting requests.

Usage:
    # Instead of:
    #   python -m sglang.launch_server --model deepseek-ai/DeepSeek-V2-Lite ...
    # Use:
    python launch_patched_server.py --model deepseek-ai/DeepSeek-V2-Lite --tp 1

    # All sglang.launch_server args are forwarded.
"""

import sys
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mla_patch")

# Add our src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def apply_patch():
    """Patch SGLang's eviction function before server starts."""
    try:
        from mla_radix_cache import MLAModelConfig, MLAEvictionBudget
    except ImportError:
        logger.error("Cannot import mla_radix_cache. Make sure src/ is accessible.")
        return False

    import sglang.srt.mem_cache.common as cache_common
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    # We don't know the exact model config yet (it's loaded later),
    # so we patch with a lazy detection approach
    original_evict = cache_common.evict_from_tree_cache

    _budget_cache = {}

    def patched_evict(tree_cache, num_tokens):
        if tree_cache is None or tree_cache.is_chunk_cache():
            return original_evict(tree_cache, num_tokens)

        allocator = tree_cache.token_to_kv_pool_allocator

        # Check if this is an MLA pool
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        kv_cache = getattr(allocator, "kv_cache", None) or getattr(
            tree_cache, "token_to_kv_pool", None
        )

        # Try to find the actual KV pool
        is_mla = False
        if kv_cache is not None and isinstance(kv_cache, MLATokenToKVPool):
            is_mla = True
        else:
            # Check via allocator -> pool chain
            pool = getattr(allocator, "pool", None)
            if pool is not None and isinstance(pool, MLATokenToKVPool):
                is_mla = True
                kv_cache = pool

        if not is_mla:
            return original_evict(tree_cache, num_tokens)

        # Lazy init budget
        cache_id = id(tree_cache)
        if cache_id not in _budget_cache:
            kv_lora_rank = getattr(kv_cache, "kv_lora_rank", 512)
            qk_rope_head_dim = getattr(kv_cache, "qk_rope_head_dim", 64)
            pool_size = getattr(allocator, "size", 100000)

            config = MLAModelConfig(
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                num_attention_heads=128,  # approximation
                qk_nope_head_dim=128,
                v_head_dim=128,
                num_layers=61,
            )
            _budget_cache[cache_id] = MLAEvictionBudget(config, pool_size)
            logger.info(
                f"MLA eviction budget initialized: "
                f"compression_ratio={config.compression_ratio:.1f}x, "
                f"pool_size={pool_size}"
            )

        budget = _budget_cache[cache_id]

        # Apply MLA-aware eviction
        from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
        if isinstance(allocator, SWATokenToKVPoolAllocator):
            return original_evict(tree_cache, num_tokens)

        available = allocator.available_size()
        if available < num_tokens:
            needed = num_tokens - available
            cached = tree_cache.evictable_size()
            adjusted = budget.adjust_eviction_count(needed, cached, available)
            tree_cache.evict(EvictParams(num_tokens=adjusted))

    cache_common.evict_from_tree_cache = patched_evict
    logger.info("✓ MLA eviction patch applied to sglang.srt.mem_cache.common")
    return True


def main():
    # Apply patch first
    success = apply_patch()
    if not success:
        logger.warning("MLA patch failed to apply, running vanilla SGLang")

    # Forward to sglang.launch_server
    from sglang.srt.entrypoints.http_server import launch_server
    launch_server()


if __name__ == "__main__":
    main()
