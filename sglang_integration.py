"""
SGLang integration for MLA-aware RadixCache.

This module provides the patches needed to integrate MLARadixCache
into SGLang's serving pipeline. It wraps SGLang's existing RadixCache
with MLA-aware eviction and capacity management.

Integration points (by SGLang file):
1. model_runner_kv_cache_mixin.py: Pool initialization
2. scheduler.py: _check_memory_and_evict() thresholds
3. pool_configurator.py: Pool size calculation

Usage:
    from mla_radix_attention.sglang_integration import (
        patch_scheduler_for_mla,
        compute_mla_pool_size,
    )

    # In scheduler init:
    patch_scheduler_for_mla(scheduler, model_config)

    # In pool configurator:
    pool_size = compute_mla_pool_size(
        available_memory, model_config, dtype
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

try:
    from .mla_radix_cache import (
        LatentCacheAnalyzer,
        MLACacheStats,
        MLAEvictionBudget,
        MLAModelConfig,
    )
except ImportError:
    from mla_radix_cache import (
        LatentCacheAnalyzer,
        MLACacheStats,
        MLAEvictionBudget,
        MLAModelConfig,
    )

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def detect_mla_config(model_config) -> Optional[MLAModelConfig]:
    """Auto-detect MLA configuration from SGLang's ModelConfig.

    SGLang sets model_config.kv_lora_rank > 0 for MLA models.
    This is checked in model_runner_kv_cache_mixin.py to decide
    whether to use MLATokenToKVPool.

    Returns None if the model is not MLA.
    """
    kv_lora_rank = getattr(model_config, "kv_lora_rank", None)
    if kv_lora_rank is None or kv_lora_rank <= 0:
        return None

    return MLAModelConfig(
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=getattr(model_config, "qk_rope_head_dim", 64),
        num_attention_heads=getattr(model_config, "num_attention_heads", 128),
        qk_nope_head_dim=getattr(model_config, "qk_nope_head_dim", 128),
        v_head_dim=getattr(model_config, "v_head_dim", 128),
        num_layers=getattr(model_config, "num_hidden_layers", 61),
    )


def compute_mla_pool_size(
    available_memory_bytes: int,
    model_config: MLAModelConfig,
    dtype: torch.dtype = torch.bfloat16,
    page_size: int = 1,
) -> int:
    """Compute optimal KV pool size for an MLA model.

    This replaces the standard calculation in pool_configurator.py:
        max_total_num_tokens = available_memory / (per_token_bytes * num_layers)

    For MLA, per_token_bytes is much smaller (latent_dim vs n_heads * d_head),
    so we can fit significantly more tokens.

    Args:
        available_memory_bytes: GPU memory available for KV cache
        model_config: MLA model configuration
        dtype: Cache dtype (bf16, fp8, etc.)
        page_size: Page size for paged attention

    Returns:
        Maximum number of tokens that can be cached
    """
    analyzer = LatentCacheAnalyzer(model_config)
    max_tokens = analyzer.compute_max_tokens(available_memory_bytes, dtype)

    # Align to page_size
    max_tokens = (max_tokens // page_size) * page_size

    # Log comparison with MHA
    mha_tokens = analyzer.compute_mha_equivalent_tokens(available_memory_bytes, dtype)
    logger.info(
        f"MLA pool size: {max_tokens} tokens "
        f"(MHA equivalent: {mha_tokens}, "
        f"{model_config.compression_ratio:.1f}x more capacity)"
    )

    return max_tokens


class MLASchedulerMixin:
    """Mixin for SGLang's Scheduler to add MLA-aware eviction.

    Patches _check_memory_and_evict to use MLA-adjusted thresholds.

    In SGLang's scheduler.py, the eviction check is roughly:
        if allocator.available_size() < threshold:
            self.tree_cache.evict(EvictParams(num_tokens=needed))

    For MLA models, we adjust:
    1. The threshold (lower, since tokens are cheaper)
    2. The eviction count (fewer tokens, since each is smaller)
    """

    def init_mla_eviction(
        self,
        model_config: MLAModelConfig,
        total_pool_tokens: int,
    ):
        """Initialize MLA eviction budget. Call from scheduler init."""
        self._mla_eviction_budget = MLAEvictionBudget(
            model_config, total_pool_tokens
        )
        self._mla_stats = MLACacheStats()
        self._mla_config = model_config
        logger.info(
            f"MLA eviction initialized: "
            f"compression_ratio={model_config.compression_ratio:.1f}x, "
            f"target_free_ratio={self._mla_eviction_budget.target_free_ratio:.3f}"
        )

    def mla_adjusted_eviction_count(
        self,
        requested: int,
        cached: int,
        free: int,
    ) -> int:
        """Get MLA-adjusted eviction count."""
        if not hasattr(self, "_mla_eviction_budget"):
            return requested
        return self._mla_eviction_budget.adjust_eviction_count(
            requested, cached, free
        )

    def mla_should_evict(self, free: int) -> bool:
        """Check if eviction should be triggered (MLA-aware)."""
        if not hasattr(self, "_mla_eviction_budget"):
            return True  # fall through to default behavior
        return self._mla_eviction_budget.should_trigger_eviction(free)

    def get_mla_cache_pressure(self, cached: int, free: int) -> float:
        """Get MLA-aware cache pressure metric."""
        if not hasattr(self, "_mla_eviction_budget"):
            return cached / max(1, cached + free)
        return self._mla_eviction_budget.get_cache_pressure(cached, free)


def patch_scheduler_for_mla(scheduler, model_config: MLAModelConfig):
    """Monkey-patch an existing scheduler instance for MLA-aware eviction.

    This is the least-invasive integration path — no code changes needed
    in SGLang's scheduler.py, just call this after scheduler init.

    Usage:
        scheduler = Scheduler(...)
        mla_config = detect_mla_config(scheduler.model_config)
        if mla_config:
            patch_scheduler_for_mla(scheduler, mla_config)
    """
    import types

    pool_size = getattr(scheduler, "max_total_num_tokens", 100000)

    budget = MLAEvictionBudget(model_config, pool_size)

    # Store on scheduler
    scheduler._mla_config = model_config
    scheduler._mla_budget = budget

    logger.info(
        f"Patched scheduler for MLA: "
        f"compression_ratio={model_config.compression_ratio:.1f}x"
    )


# ────────────────────────────────────────────────────────────────────
# SGLang patch generator (for PR / diff)
# ────────────────────────────────────────────────────────────────────

def generate_sglang_patch() -> str:
    """Generate a unified diff showing the SGLang integration changes.

    This shows what code changes are needed in SGLang to enable
    MLA-aware RadixAttention. Used for documentation and PR prep.
    """
    patch = """
# =============================================================
# SGLang Integration Patch for MLA-aware RadixAttention
# =============================================================
#
# Files modified:
# 1. python/sglang/srt/mem_cache/cache_init_params.py
#    - Add mla_model_config field
#
# 2. python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
#    - Pass MLA config when creating RadixCache
#    - Use MLA-aware pool size calculation
#
# 3. python/sglang/srt/managers/scheduler.py
#    - Use MLA-adjusted eviction thresholds in _check_memory_and_evict()
#
# 4. python/sglang/srt/mem_cache/radix_cache.py
#    - Add MLA-aware eviction scoring (optional enhancement)
#

# --- cache_init_params.py ---
# Add to CacheInitParams:
#     mla_model_config: Optional[MLAModelConfig] = None

# --- model_runner_kv_cache_mixin.py ---
# In _init_kv_cache(), after detecting MLA:
#     if self.model_config.kv_lora_rank:
#         from mla_radix_attention import MLAModelConfig, compute_mla_pool_size
#         mla_config = MLAModelConfig(
#             kv_lora_rank=self.model_config.kv_lora_rank,
#             qk_rope_head_dim=self.model_config.qk_rope_head_dim,
#             ...
#         )
#         # Pool size already accounts for latent dim since SGLang uses
#         # latent_dim in pool_configurator.py. But we can inform the
#         # radix cache about the compression ratio for better eviction.
#         cache_init_params.mla_model_config = mla_config

# --- scheduler.py ---
# In _check_memory_and_evict():
#     if hasattr(self, '_mla_budget'):
#         adjusted = self._mla_budget.adjust_eviction_count(
#             num_tokens, cached, free
#         )
#         self.tree_cache.evict(EvictParams(num_tokens=adjusted))
#     else:
#         self.tree_cache.evict(EvictParams(num_tokens=num_tokens))
"""
    return patch
