"""
MLA-aware RadixCache for SGLang.

Extends SGLang's RadixCache with latent-space-aware memory management.

Background (from SGLang code archaeology):
------------------------------------------
SGLang's MLATokenToKVPool already stores the compressed latent vector per token:
    kv_buffer shape: [pool_size, 1, kv_lora_rank + qk_rope_head_dim]
    e.g. DeepSeek V3: [pool_size, 1, 512 + 64] = 576 floats per token per layer

Standard MHA would store:
    k_buffer shape: [pool_size, n_heads, d_head]
    v_buffer shape: [pool_size, n_heads, d_head]
    e.g. 32 heads * 128 dim * 2 (K+V) = 8192 floats per token per layer

Compression ratio: 8192 / 576 ≈ 14.2x

However, RadixCache's eviction and capacity estimation treats every cached token
identically regardless of per-token memory cost. This means:
1. Eviction thresholds computed from MHA-era assumptions are too conservative
2. The cache cannot exploit the fact that MLA tokens are ~14x cheaper to store
3. Memory budget for prefix caching is underutilized on MLA models

This module provides:
- MLARadixCache: drop-in replacement for RadixCache with MLA-aware eviction
- LatentCacheAnalyzer: tools to compute actual memory savings and optimal budgets
- MLAEvictionPolicy: eviction strategy that factors in compression ratio
"""

from __future__ import annotations

import dataclasses
import heapq
import logging
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MLAModelConfig:
    """Configuration for an MLA model's cache characteristics.

    Extracted from the model's HuggingFace config at server startup.
    """

    kv_lora_rank: int  # DeepSeek V3: 512
    qk_rope_head_dim: int  # DeepSeek V3: 64
    num_attention_heads: int  # DeepSeek V3: 128
    qk_nope_head_dim: int  # DeepSeek V3: 128
    v_head_dim: int  # DeepSeek V3: 128
    num_layers: int  # DeepSeek V3: 61

    @property
    def latent_dim(self) -> int:
        """Per-token latent cache dimension (what's actually stored)."""
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def mha_equivalent_dim(self) -> int:
        """Per-token dim if this were standard MHA (K + V, all heads)."""
        return self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim)

    @property
    def compression_ratio(self) -> float:
        """How much smaller latent cache is vs equivalent MHA."""
        return self.mha_equivalent_dim / self.latent_dim

    @property
    def bytes_per_token_per_layer(self) -> int:
        """Bytes per token per layer in the latent pool (bf16)."""
        return self.latent_dim * 2  # bf16 = 2 bytes

    @property
    def total_bytes_per_token(self) -> int:
        """Total bytes per token across all layers."""
        return self.bytes_per_token_per_layer * self.num_layers

    @classmethod
    def from_hf_config(cls, hf_config) -> "MLAModelConfig":
        """Extract MLA parameters from a HuggingFace model config."""
        return cls(
            kv_lora_rank=getattr(hf_config, "kv_lora_rank", 512),
            qk_rope_head_dim=getattr(hf_config, "qk_rope_head_dim", 64),
            num_attention_heads=getattr(hf_config, "num_attention_heads", 128),
            qk_nope_head_dim=getattr(hf_config, "qk_nope_head_dim", 128),
            v_head_dim=getattr(hf_config, "v_head_dim", 128),
            num_layers=getattr(hf_config, "num_hidden_layers", 61),
        )

    @classmethod
    def deepseek_v3(cls) -> "MLAModelConfig":
        """Preset for DeepSeek-V3."""
        return cls(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_attention_heads=128,
            qk_nope_head_dim=128,
            v_head_dim=128,
            num_layers=61,
        )

    @classmethod
    def deepseek_v2_lite(cls) -> "MLAModelConfig":
        """Preset for DeepSeek-V2-Lite (dev/test model)."""
        return cls(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            num_attention_heads=16,
            qk_nope_head_dim=128,
            v_head_dim=128,
            num_layers=27,
        )


@dataclasses.dataclass
class MLACacheStats:
    """Runtime statistics for the MLA-aware cache."""

    # Hit/miss tracking
    total_requests: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    total_tokens_matched: int = 0
    total_tokens_computed: int = 0

    # Memory tracking
    cached_tokens: int = 0
    peak_cached_tokens: int = 0
    evicted_tokens: int = 0

    # Latent-specific
    latent_bytes_saved: int = 0  # vs MHA equivalent
    latent_bytes_used: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.prefix_hits / self.total_requests

    @property
    def token_reuse_rate(self) -> float:
        total = self.total_tokens_matched + self.total_tokens_computed
        if total == 0:
            return 0.0
        return self.total_tokens_matched / total

    @property
    def memory_efficiency(self) -> float:
        """Ratio of actual memory used vs MHA equivalent."""
        total = self.latent_bytes_used + self.latent_bytes_saved
        if total == 0:
            return 1.0
        return self.latent_bytes_used / total

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.3f}",
            "token_reuse_rate": f"{self.token_reuse_rate:.3f}",
            "cached_tokens": self.cached_tokens,
            "peak_cached_tokens": self.peak_cached_tokens,
            "evicted_tokens": self.evicted_tokens,
            "latent_bytes_used_mb": f"{self.latent_bytes_used / (1024 * 1024):.1f}",
            "latent_bytes_saved_mb": f"{self.latent_bytes_saved / (1024 * 1024):.1f}",
            "memory_efficiency": f"{self.memory_efficiency:.3f}",
        }


# ────────────────────────────────────────────────────────────────────
# Latent Cache Analyzer
# ────────────────────────────────────────────────────────────────────

class LatentCacheAnalyzer:
    """Analyzes and recommends cache configurations for MLA models.

    Given a GPU memory budget and MLA model config, computes optimal
    pool sizes that exploit the latent compression ratio.
    """

    def __init__(self, model_config: MLAModelConfig):
        self.config = model_config

    def compute_max_tokens(
        self,
        available_bytes: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Max tokens cacheable given available GPU memory.

        For MLA, this is ~compression_ratio times more than MHA.
        """
        bytes_per_element = dtype.itemsize if hasattr(dtype, 'itemsize') else 2
        bytes_per_token = (
            self.config.latent_dim * bytes_per_element * self.config.num_layers
        )
        return available_bytes // bytes_per_token

    def compute_mha_equivalent_tokens(
        self,
        available_bytes: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Max tokens if this were standard MHA (for comparison)."""
        bytes_per_element = dtype.itemsize if hasattr(dtype, 'itemsize') else 2
        bytes_per_token = (
            self.config.mha_equivalent_dim * bytes_per_element * self.config.num_layers
        )
        return available_bytes // bytes_per_token

    def compute_memory_for_tokens(
        self,
        num_tokens: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Memory in bytes needed to cache n tokens."""
        bytes_per_element = dtype.itemsize if hasattr(dtype, 'itemsize') else 2
        return (
            num_tokens
            * self.config.latent_dim
            * bytes_per_element
            * self.config.num_layers
        )

    def recommend_pool_size(
        self,
        gpu_memory_bytes: int,
        mem_fraction_static: float = 0.8,
        model_weight_bytes: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Recommend pool configuration for serving.

        Returns a dict with recommended pool sizes and expected metrics.
        """
        available = int(gpu_memory_bytes * mem_fraction_static) - model_weight_bytes
        available = max(available, 0)

        mla_tokens = self.compute_max_tokens(available, dtype)
        mha_tokens = self.compute_mha_equivalent_tokens(available, dtype)

        return {
            "available_memory_gb": available / (1024 ** 3),
            "mla_max_tokens": mla_tokens,
            "mha_max_tokens": mha_tokens,
            "compression_ratio": self.config.compression_ratio,
            "extra_tokens_from_mla": mla_tokens - mha_tokens,
            "bytes_per_token_mla": self.config.total_bytes_per_token,
            "bytes_per_token_mha": (
                self.config.mha_equivalent_dim * 2 * self.config.num_layers
            ),
            "recommended_pool_size": mla_tokens,
        }

    def estimate_prefix_cache_benefit(
        self,
        avg_shared_prefix_len: int,
        num_concurrent_requests: int,
        pool_size: int,
    ) -> dict:
        """Estimate benefit of prefix caching for a workload.

        Models the expected hit rate and compute savings given:
        - Average shared prefix length across requests
        - Number of concurrent requests
        - Available pool size
        """
        # Total prefix tokens across all requests
        total_prefix_tokens = avg_shared_prefix_len * num_concurrent_requests

        # With deduplication, unique prefixes are much smaller
        # Assume log-scale dedup (Zipf-like prefix distribution)
        import math

        estimated_unique_prefixes = int(
            avg_shared_prefix_len * math.log2(max(2, num_concurrent_requests))
        )

        # Can we fit all unique prefixes?
        fits_in_cache = estimated_unique_prefixes <= pool_size
        expected_hit_rate = 1.0 if fits_in_cache else pool_size / estimated_unique_prefixes

        # Compute savings: each hit saves prefill computation
        tokens_saved_per_request = int(avg_shared_prefix_len * expected_hit_rate)

        return {
            "total_prefix_tokens": total_prefix_tokens,
            "estimated_unique_prefixes": estimated_unique_prefixes,
            "fits_in_cache": fits_in_cache,
            "expected_hit_rate": expected_hit_rate,
            "tokens_saved_per_request": tokens_saved_per_request,
            "total_tokens_saved": tokens_saved_per_request * num_concurrent_requests,
        }


# ────────────────────────────────────────────────────────────────────
# MLA-aware eviction budget calculator
# ────────────────────────────────────────────────────────────────────

class MLAEvictionBudget:
    """Computes eviction budgets that account for MLA compression.

    In standard RadixCache, when the allocator reports N free slots needed,
    the cache evicts N tokens. But for MLA models, each token is much cheaper,
    so we can afford to keep more tokens cached. This class adjusts eviction
    thresholds accordingly.

    Integrates with SGLang's scheduler loop:
        scheduler.py: _check_memory_and_evict() calls
            radix_cache.evict(EvictParams(num_tokens=...))

    The adjustment happens at the boundary between scheduler and cache:
    instead of blindly evicting `num_tokens`, we evict
    `num_tokens * adjustment_factor` to account for the fact that MLA
    tokens cost less memory than standard tokens.
    """

    def __init__(
        self,
        model_config: MLAModelConfig,
        total_pool_tokens: int,
    ):
        self.model_config = model_config
        self.total_pool_tokens = total_pool_tokens

        # Target cache utilization — higher for MLA since tokens are cheaper
        # Standard: keep ~20% free. MLA: can keep ~5-10% free.
        self.target_free_ratio = max(0.05, 0.20 / model_config.compression_ratio)

    def adjust_eviction_count(
        self,
        requested_eviction: int,
        current_cached: int,
        current_free: int,
    ) -> int:
        """Adjust the number of tokens to evict.

        For MLA models, we can be more conservative with eviction since
        each token costs ~14x less memory. This means:
        1. We tolerate higher cache occupancy before triggering eviction
        2. When we do evict, we evict fewer tokens

        The caller (scheduler) explicitly requested eviction, so we always
        evict at least some tokens. The adjustment only reduces the count,
        never zeroes it.

        Args:
            requested_eviction: How many tokens the allocator wants freed
            current_cached: How many tokens are currently cached
            current_free: How many free slots remain

        Returns:
            Adjusted number of tokens to actually evict (always >= 1 if requested > 0)
        """
        if requested_eviction <= 0:
            return 0

        total = current_cached + current_free
        if total == 0:
            return requested_eviction

        free_ratio = current_free / total

        # If we already have enough free space for MLA's reduced footprint,
        # reduce eviction proportionally but never to zero
        if free_ratio > self.target_free_ratio:
            # Scale down: the more free space we have, the less we evict
            scale = max(0.1, 1.0 - (free_ratio - self.target_free_ratio))
            return max(1, int(requested_eviction * scale))

        return requested_eviction

    def should_trigger_eviction(
        self,
        current_free: int,
    ) -> bool:
        """Whether we should proactively evict based on MLA budget."""
        total = self.total_pool_tokens
        if total == 0:
            return False
        free_ratio = current_free / total
        return free_ratio < self.target_free_ratio

    def get_cache_pressure(
        self,
        current_cached: int,
        current_free: int,
    ) -> float:
        """Cache pressure metric [0, 1]. Higher = more pressure to evict.

        MLA models naturally have lower pressure since tokens are cheaper.
        """
        total = current_cached + current_free
        if total == 0:
            return 1.0
        used_ratio = current_cached / total
        # Scale by inverse compression ratio — MLA has less pressure per token
        effective_pressure = used_ratio / self.model_config.compression_ratio
        return min(1.0, effective_pressure)


# ────────────────────────────────────────────────────────────────────
# MLA-aware RadixCache
# ────────────────────────────────────────────────────────────────────

class TreeNode:
    """Tree node for the MLA-aware radix cache.

    Same structure as SGLang's TreeNode but with additional metadata
    for latent-space-aware eviction scoring.
    """

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children: Dict[Any, TreeNode] = defaultdict(TreeNode)
        self.parent: Optional[TreeNode] = None
        self.key = None  # RadixKey
        self.value: Optional[torch.Tensor] = None  # KV pool slot indices
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()
        self.hit_count = 0
        self.host_ref_counter = 0
        self.host_value: Optional[torch.Tensor] = None
        self.hash_value: Optional[List[str]] = None
        self.priority = priority

        # MLA-specific metadata
        self.latent_bytes: int = 0  # actual memory footprint of this node's latents
        self.access_frequency: float = 0.0  # weighted access frequency

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None

    @property
    def backuped(self):
        return self.host_value is not None

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


class MLARadixCache:
    """RadixCache with MLA-aware memory management.

    Drop-in compatible with SGLang's RadixCache interface, but with:
    1. Eviction budget adjusted for MLA compression ratio
    2. Cache pressure metrics that account for latent vector size
    3. Adaptive eviction that keeps more prefixes cached
    4. Per-node memory footprint tracking

    This is designed to be integrated into SGLang by:
    - Replacing RadixCache in scheduler.py
    - Adjusting _check_memory_and_evict() thresholds
    - Adding MLA model config to CacheInitParams

    Usage (standalone / testing):
        config = MLAModelConfig.deepseek_v3()
        cache = MLARadixCache(config, pool_size=100000)
        cache.insert(token_ids=[1,2,3,4], slot_indices=torch.tensor([10,11,12,13]))
        result = cache.match_prefix([1,2,3,5])
        # result.matched_len == 3, result.slot_indices == tensor([10,11,12])
    """

    def __init__(
        self,
        model_config: MLAModelConfig,
        pool_size: int,
        page_size: int = 1,
        eviction_policy: str = "lru",
        device: str = "cpu",
    ):
        self.model_config = model_config
        self.pool_size = pool_size
        self.page_size = page_size
        self.device = torch.device(device)

        # Core tree
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = []
        self.root_node.value = torch.tensor([], dtype=torch.int64, device=self.device)
        self.root_node.lock_ref = 1

        # Eviction
        self.eviction_budget = MLAEvictionBudget(model_config, pool_size)
        self.evictable_leaves: set = set()
        self.evictable_size_ = 0
        self.protected_size_ = 0

        # Stats
        self.stats = MLACacheStats()

        # Analyzer
        self.analyzer = LatentCacheAnalyzer(model_config)

        logger.info(
            f"MLARadixCache initialized: "
            f"compression_ratio={model_config.compression_ratio:.1f}x, "
            f"latent_dim={model_config.latent_dim}, "
            f"pool_size={pool_size}, "
            f"page_size={page_size}"
        )

    # ── Public API ──────────────────────────────────────────────

    def match_prefix(
        self,
        token_ids: List[int],
    ) -> "MatchPrefixResult":
        """Find the longest cached prefix.

        Args:
            token_ids: Token ID sequence to match.

        Returns:
            MatchPrefixResult with matched length and slot indices.
        """
        self.stats.total_requests += 1

        if len(token_ids) == 0:
            self.stats.prefix_misses += 1
            return MatchPrefixResult(
                matched_len=0,
                slot_indices=torch.tensor([], dtype=torch.int64, device=self.device),
                last_node=self.root_node,
            )

        # Page-align the lookup key
        aligned_len = (len(token_ids) // self.page_size) * self.page_size
        if aligned_len == 0:
            self.stats.prefix_misses += 1
            return MatchPrefixResult(
                matched_len=0,
                slot_indices=torch.tensor([], dtype=torch.int64, device=self.device),
                last_node=self.root_node,
            )

        lookup_key = token_ids[:aligned_len]
        values, last_node = self._match_prefix_helper(self.root_node, lookup_key)

        if values:
            slot_indices = torch.cat(values)
            matched_len = len(slot_indices)
            self.stats.prefix_hits += 1
            self.stats.total_tokens_matched += matched_len
        else:
            slot_indices = torch.tensor([], dtype=torch.int64, device=self.device)
            matched_len = 0
            self.stats.prefix_misses += 1

        return MatchPrefixResult(
            matched_len=matched_len,
            slot_indices=slot_indices,
            last_node=last_node,
        )

    def insert(
        self,
        token_ids: List[int],
        slot_indices: torch.Tensor,
        priority: int = 0,
    ) -> int:
        """Insert a token sequence into the cache.

        Args:
            token_ids: Token ID sequence.
            slot_indices: Corresponding KV pool slot indices.
            priority: Priority for eviction (higher = kept longer).

        Returns:
            Length of the prefix that was already in the cache.
        """
        # Page-align
        aligned_len = (len(token_ids) // self.page_size) * self.page_size
        if aligned_len == 0:
            return 0

        key = token_ids[:aligned_len]
        value = slot_indices[:aligned_len]

        prefix_len = self._insert_helper(self.root_node, key, value, priority)
        self.stats.cached_tokens = self._total_size()
        self.stats.peak_cached_tokens = max(
            self.stats.peak_cached_tokens, self.stats.cached_tokens
        )

        # Track latent memory usage
        new_tokens = aligned_len - prefix_len
        if new_tokens > 0:
            self.stats.latent_bytes_used += (
                new_tokens * self.model_config.total_bytes_per_token
            )
            self.stats.latent_bytes_saved += (
                new_tokens
                * (self.model_config.mha_equivalent_dim * 2 * self.model_config.num_layers
                   - self.model_config.total_bytes_per_token)
            )

        return prefix_len

    def evict(self, num_tokens: int) -> int:
        """Evict tokens from the cache using MLA-aware budget.

        The key difference from standard RadixCache: we adjust the eviction
        count based on MLA compression ratio to avoid over-evicting.

        Args:
            num_tokens: Number of tokens the allocator wants freed.

        Returns:
            Number of tokens actually evicted.
        """
        # Apply MLA-aware adjustment
        adjusted_count = self.eviction_budget.adjust_eviction_count(
            requested_eviction=num_tokens,
            current_cached=self.evictable_size_,
            current_free=self.pool_size - self.evictable_size_ - self.protected_size_,
        )

        # Build eviction heap
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self._eviction_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < adjusted_count and len(eviction_heap):
            _prio, node = heapq.heappop(eviction_heap)

            node_size = len(node.value) if node.value is not None else 0
            num_evicted += node_size
            self.stats.evicted_tokens += node_size
            self._delete_leaf(node)

            if (
                node.parent is not None
                and len(node.parent.children) == 0
                and node.parent.lock_ref == 0
            ):
                new_prio = self._eviction_priority(node.parent)
                heapq.heappush(eviction_heap, (new_prio, node.parent))

        self.stats.cached_tokens = self._total_size()
        return num_evicted

    def inc_lock_ref(self, node: TreeNode) -> int:
        """Lock a node path (prevent eviction)."""
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                node_len = len(node.key) if node.key is not None else 0
                self.evictable_size_ -= node_len
                self.protected_size_ += node_len
                delta -= node_len
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode) -> int:
        """Unlock a node path (allow eviction)."""
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                node_len = len(node.key) if node.key is not None else 0
                self.evictable_size_ += node_len
                self.protected_size_ -= node_len
                delta += node_len
            node.lock_ref -= 1
            self._update_leaf_status(node)
            node = node.parent
        return delta

    def evictable_size(self) -> int:
        return self.evictable_size_

    def protected_size(self) -> int:
        return self.protected_size_

    def total_size(self) -> int:
        return self._total_size()

    def get_stats(self) -> dict:
        """Get cache statistics including MLA-specific metrics."""
        base_stats = self.stats.to_dict()
        base_stats["cache_pressure"] = self.eviction_budget.get_cache_pressure(
            self.evictable_size_,
            self.pool_size - self.evictable_size_ - self.protected_size_,
        )
        base_stats["compression_ratio"] = self.model_config.compression_ratio
        return base_stats

    def pretty_print(self):
        """Print tree structure for debugging."""
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self._total_size()}")
        print(f"compression_ratio: {self.model_config.compression_ratio:.1f}x")
        print(f"stats: {self.stats.to_dict()}")

    # ── Internal helpers ─────────────────────────────────────────

    def _match_prefix_helper(
        self, node: TreeNode, key: List[int]
    ) -> Tuple[List[torch.Tensor], TreeNode]:
        access_time = time.monotonic()
        node.last_access_time = access_time

        value = []
        pos = 0
        while pos < len(key):
            child_key = self._child_key(key, pos)
            if child_key not in node.children:
                break

            child = node.children[child_key]
            child.last_access_time = access_time
            child.hit_count += 1

            # Match as far as we can within this child
            child_token_ids = child.key
            match_len = 0
            while (
                match_len < len(child_token_ids)
                and pos + match_len < len(key)
                and child_token_ids[match_len] == key[pos + match_len]
            ):
                match_len += 1

            # Align to page boundary
            match_len = (match_len // self.page_size) * self.page_size

            if match_len == 0:
                break

            if match_len < len(child_token_ids):
                # Partial match — split the node
                new_node = self._split_node(child, match_len, child_key)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                pos += match_len

        return value, node

    def _split_node(
        self, child: TreeNode, split_len: int, child_key: Any
    ) -> TreeNode:
        """Split a node at split_len, returning the new prefix node."""
        new_node = TreeNode(priority=child.priority)
        new_node.hit_count = child.hit_count

        # New child key for the suffix
        suffix_key = self._child_key(child.key, split_len)
        new_node.children = {suffix_key: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()

        # Update the child to be the suffix
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()

        # Update parent's child pointer
        new_node.parent.children[child_key] = new_node

        # Update MLA metadata
        new_node.latent_bytes = (
            split_len * self.model_config.total_bytes_per_token
        )
        child.latent_bytes = (
            len(child.key) * self.model_config.total_bytes_per_token
        )

        return new_node

    def _insert_helper(
        self,
        node: TreeNode,
        key: List[int],
        value: torch.Tensor,
        priority: int = 0,
    ) -> int:
        access_time = time.monotonic()
        node.last_access_time = access_time
        node.priority = max(node.priority, priority)

        if len(key) == 0:
            return 0

        total_prefix_length = 0
        pos = 0

        while pos < len(key):
            child_key = self._child_key(key, pos)
            if child_key not in node.children:
                break

            child = node.children[child_key]
            child.last_access_time = access_time

            child_token_ids = child.key
            match_len = 0
            while (
                match_len < len(child_token_ids)
                and pos + match_len < len(key)
                and child_token_ids[match_len] == key[pos + match_len]
            ):
                match_len += 1

            match_len = (match_len // self.page_size) * self.page_size

            if match_len == 0:
                break

            total_prefix_length += match_len
            pos += match_len

            if match_len < len(child_token_ids):
                new_node = self._split_node(child, match_len, child_key)
                new_node.priority = max(new_node.priority, priority)
                node = new_node
            else:
                child.priority = max(child.priority, priority)
                node = child

        # Insert remaining tokens as new node
        remaining_key = key[pos:]
        remaining_value = value[pos:]
        if len(remaining_key) > 0:
            new_node = TreeNode(priority=priority)
            new_node.parent = node
            new_node.key = remaining_key
            new_node.value = remaining_value.clone()
            new_node.latent_bytes = (
                len(remaining_key) * self.model_config.total_bytes_per_token
            )
            new_node.last_access_time = access_time
            new_node.hit_count = 1

            insert_key = self._child_key(remaining_key, 0)
            node.children[insert_key] = new_node
            self.evictable_size_ += len(remaining_key)
            self._update_leaf_status(node)
            self._update_leaf_status(new_node)

        return total_prefix_length

    def _child_key(self, token_ids: List[int], pos: int = 0) -> int:
        """Get the child key for a position in token_ids.

        For page_size=1, this is just the token id.
        For page_size>1, this is a tuple of the page tokens.
        """
        if self.page_size == 1:
            return token_ids[pos]
        else:
            end = min(pos + self.page_size, len(token_ids))
            return tuple(token_ids[pos:end])

    def _eviction_priority(self, node: TreeNode) -> float:
        """Compute eviction priority for a node.

        MLA-aware: factors in the actual memory cost of the node's latent
        vectors. Nodes with larger latent footprint get higher eviction
        priority (more likely to be evicted) when they're cold.
        """
        # Base priority: LRU (older access = smaller value = evicted first by min-heap)
        time_score = node.last_access_time

        # Frequency bonus: more hits = higher value = less likely to be evicted
        freq_bonus = node.hit_count * 0.01

        # MLA cost factor: nodes with more tokens cost more memory,
        # so they get slightly lower priority (more likely to be evicted)
        node_tokens = len(node.value) if node.value is not None else 0
        cost_factor = -node_tokens * self.model_config.total_bytes_per_token * 1e-12

        # Priority override: higher explicit priority = higher value = protected
        priority_factor = node.priority * 1e6

        return time_score + freq_bonus + cost_factor + priority_factor

    def _delete_leaf(self, node: TreeNode):
        if node.key is not None:
            child_key = self._child_key(node.key, 0)
            if child_key in node.parent.children:
                del node.parent.children[child_key]

        node_len = len(node.key) if node.key is not None else 0
        self.evictable_size_ -= node_len
        if node in self.evictable_leaves:
            self.evictable_leaves.discard(node)
        node.value = None  # mark as evicted
        self._update_leaf_status(node.parent)

    def _update_leaf_status(self, node: TreeNode):
        if node is None:
            return
        if node.evicted or node.lock_ref > 0:
            self.evictable_leaves.discard(node)
            return
        for child in node.children.values():
            if not child.evicted:
                self.evictable_leaves.discard(node)
                return
        self.evictable_leaves.add(node)

    def _total_size(self) -> int:
        total = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node.value is not None:
                total += len(node.value)
            for child in node.children.values():
                if not child.evicted:
                    stack.append(child)
        return total

    def _print_helper(self, node: TreeNode, indent: int):
        stack = [(node, indent)]
        while stack:
            current, cur_indent = stack.pop()
            key_preview = current.key[:10] if current.key else []
            key_len = len(current.key) if current.key else 0
            latent_mb = current.latent_bytes / (1024 * 1024)
            print(
                " " * cur_indent,
                f"len={key_len}",
                key_preview,
                f"r={current.lock_ref}",
                f"hits={current.hit_count}",
                f"latent={latent_mb:.3f}MB",
            )
            for child in current.children.values():
                stack.append((child, cur_indent + 2))


class MatchPrefixResult(NamedTuple):
    """Result of a prefix match operation."""

    matched_len: int
    slot_indices: torch.Tensor
    last_node: TreeNode
