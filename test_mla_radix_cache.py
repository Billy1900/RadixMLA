"""
Tests for MLA-aware RadixCache.

Run: python -m pytest tests/test_mla_radix_cache.py -v
"""

import sys
import time
from pathlib import Path

import torch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mla_radix_cache import (
    LatentCacheAnalyzer,
    MLACacheStats,
    MLAEvictionBudget,
    MLAModelConfig,
    MLARadixCache,
    MatchPrefixResult,
)


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────

@pytest.fixture
def v3_config():
    return MLAModelConfig.deepseek_v3()


@pytest.fixture
def v2lite_config():
    return MLAModelConfig.deepseek_v2_lite()


@pytest.fixture
def cache(v3_config):
    return MLARadixCache(v3_config, pool_size=10000, page_size=1)


@pytest.fixture
def paged_cache(v3_config):
    return MLARadixCache(v3_config, pool_size=10000, page_size=4)


# ────────────────────────────────────────────────────────────────────
# MLAModelConfig tests
# ────────────────────────────────────────────────────────────────────

class TestMLAModelConfig:

    def test_deepseek_v3_params(self, v3_config):
        assert v3_config.kv_lora_rank == 512
        assert v3_config.qk_rope_head_dim == 64
        assert v3_config.latent_dim == 576  # 512 + 64
        assert v3_config.num_attention_heads == 128

    def test_compression_ratio(self, v3_config):
        # latent_dim = 576
        # mha_equivalent = 128 * (128 + 64 + 128) = 128 * 320 = 40960
        ratio = v3_config.compression_ratio
        assert ratio > 10  # should be very high
        assert ratio < 100

    def test_bytes_per_token(self, v3_config):
        # bf16: 2 bytes per float
        assert v3_config.bytes_per_token_per_layer == 576 * 2  # 1152

    def test_v2_lite_config(self, v2lite_config):
        assert v2lite_config.num_attention_heads == 16
        assert v2lite_config.num_layers == 27
        assert v2lite_config.latent_dim == 576  # same latent dim

    def test_from_hf_config(self):
        class MockConfig:
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            num_attention_heads = 128
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_hidden_layers = 61

        config = MLAModelConfig.from_hf_config(MockConfig())
        assert config.kv_lora_rank == 512
        assert config.num_layers == 61


# ────────────────────────────────────────────────────────────────────
# Basic cache operations
# ────────────────────────────────────────────────────────────────────

class TestMLARadixCacheBasic:

    def test_empty_match(self, cache):
        result = cache.match_prefix([1, 2, 3])
        assert result.matched_len == 0
        assert len(result.slot_indices) == 0

    def test_insert_and_match(self, cache):
        slots = torch.arange(5, dtype=torch.int64)
        cache.insert([1, 2, 3, 4, 5], slots)

        result = cache.match_prefix([1, 2, 3, 4, 5])
        assert result.matched_len == 5
        assert torch.equal(result.slot_indices, slots)

    def test_prefix_match(self, cache):
        slots = torch.arange(5, dtype=torch.int64)
        cache.insert([1, 2, 3, 4, 5], slots)

        result = cache.match_prefix([1, 2, 3, 6, 7])
        assert result.matched_len == 3
        assert torch.equal(result.slot_indices, slots[:3])

    def test_no_match(self, cache):
        slots = torch.arange(5, dtype=torch.int64)
        cache.insert([1, 2, 3, 4, 5], slots)

        result = cache.match_prefix([6, 7, 8])
        assert result.matched_len == 0

    def test_multiple_inserts_shared_prefix(self, cache):
        cache.insert([1, 2, 3, 4, 5], torch.arange(5, dtype=torch.int64))
        cache.insert(
            [1, 2, 3, 6, 7], torch.arange(10, 15, dtype=torch.int64)
        )

        # Both should share prefix [1, 2, 3]
        r1 = cache.match_prefix([1, 2, 3, 4, 5])
        assert r1.matched_len == 5

        r2 = cache.match_prefix([1, 2, 3, 6, 7])
        assert r2.matched_len == 5

    def test_empty_input(self, cache):
        result = cache.match_prefix([])
        assert result.matched_len == 0

    def test_insert_returns_prefix_len(self, cache):
        cache.insert([1, 2, 3], torch.arange(3, dtype=torch.int64))
        prefix_len = cache.insert(
            [1, 2, 3, 4, 5], torch.arange(5, dtype=torch.int64)
        )
        assert prefix_len == 3  # first 3 were already cached

    def test_total_size(self, cache):
        cache.insert([1, 2, 3], torch.arange(3, dtype=torch.int64))
        cache.insert([1, 2, 4, 5], torch.arange(4, dtype=torch.int64))
        # Tree should have: [1,2] (shared) + [3] + [4,5] = 5 tokens total
        assert cache.total_size() >= 4  # at minimum the non-overlapping parts


# ────────────────────────────────────────────────────────────────────
# Paged cache tests
# ────────────────────────────────────────────────────────────────────

class TestMLARadixCachePaged:

    def test_page_alignment(self, paged_cache):
        # page_size=4, so 5 tokens should be aligned to 4
        slots = torch.arange(5, dtype=torch.int64)
        paged_cache.insert([1, 2, 3, 4, 5], slots)

        # Only first 4 tokens (1 page) should be cached
        result = paged_cache.match_prefix([1, 2, 3, 4, 5])
        assert result.matched_len == 4

    def test_sub_page_no_cache(self, paged_cache):
        # 3 tokens < page_size=4, nothing gets cached
        slots = torch.arange(3, dtype=torch.int64)
        paged_cache.insert([1, 2, 3], slots)

        result = paged_cache.match_prefix([1, 2, 3])
        assert result.matched_len == 0

    def test_exact_page(self, paged_cache):
        slots = torch.arange(4, dtype=torch.int64)
        paged_cache.insert([1, 2, 3, 4], slots)

        result = paged_cache.match_prefix([1, 2, 3, 4])
        assert result.matched_len == 4

    def test_multi_page(self, paged_cache):
        slots = torch.arange(8, dtype=torch.int64)
        paged_cache.insert([1, 2, 3, 4, 5, 6, 7, 8], slots)

        result = paged_cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        assert result.matched_len == 8


# ────────────────────────────────────────────────────────────────────
# Eviction tests
# ────────────────────────────────────────────────────────────────────

class TestMLARadixCacheEviction:

    def test_basic_eviction(self, cache):
        # Insert some data
        for i in range(10):
            cache.insert(
                [100 + i * 10 + j for j in range(5)],
                torch.arange(i * 5, i * 5 + 5, dtype=torch.int64),
            )

        initial_size = cache.total_size()
        assert initial_size > 0

        # Evict some tokens
        evicted = cache.evict(20)
        assert evicted > 0
        assert cache.total_size() < initial_size

    def test_eviction_respects_locks(self, cache):
        slots = torch.arange(5, dtype=torch.int64)
        cache.insert([1, 2, 3, 4, 5], slots)

        # Lock the node
        result = cache.match_prefix([1, 2, 3, 4, 5])
        cache.inc_lock_ref(result.last_node)

        # Try to evict — should not evict locked node
        initial_size = cache.total_size()
        cache.evict(100)
        assert cache.total_size() == initial_size

        # Unlock and evict
        cache.dec_lock_ref(result.last_node)
        cache.evict(100)
        assert cache.total_size() < initial_size

    def test_lru_eviction_order(self, cache):
        # Insert two sequences
        cache.insert([1, 2, 3], torch.arange(3, dtype=torch.int64))
        time.sleep(0.01)  # ensure different timestamps
        cache.insert([4, 5, 6], torch.arange(3, 6, dtype=torch.int64))

        # Access first sequence to make it recent
        cache.match_prefix([1, 2, 3])

        # Evict — should evict [4,5,6] first (older access)
        cache.evict(3)

        # [1,2,3] should still be cached
        result = cache.match_prefix([1, 2, 3])
        assert result.matched_len == 3


# ────────────────────────────────────────────────────────────────────
# MLA-aware eviction budget tests
# ────────────────────────────────────────────────────────────────────

class TestMLAEvictionBudget:

    def test_budget_reduces_eviction(self, v3_config):
        budget = MLAEvictionBudget(v3_config, total_pool_tokens=10000)

        # When we have plenty of free space, evict less
        adjusted = budget.adjust_eviction_count(
            requested_eviction=100,
            current_cached=5000,
            current_free=5000,
        )
        assert adjusted <= 100  # should evict less or equal

    def test_cache_pressure_mla_vs_mha(self, v3_config):
        budget = MLAEvictionBudget(v3_config, total_pool_tokens=10000)

        # MLA has lower pressure for the same occupancy
        pressure = budget.get_cache_pressure(8000, 2000)
        assert pressure < 1.0  # should be significantly less than 1
        # With ~14x compression, 80% utilization → ~5.6% effective pressure
        assert pressure < 0.2

    def test_target_free_ratio(self, v3_config):
        budget = MLAEvictionBudget(v3_config, total_pool_tokens=10000)
        # With 14x compression, target free ratio should be much lower than 20%
        assert budget.target_free_ratio < 0.10


# ────────────────────────────────────────────────────────────────────
# Latent cache analyzer tests
# ────────────────────────────────────────────────────────────────────

class TestLatentCacheAnalyzer:

    def test_max_tokens(self, v3_config):
        analyzer = LatentCacheAnalyzer(v3_config)

        # 1 GB of memory
        memory = 1 * 1024 ** 3
        mla_tokens = analyzer.compute_max_tokens(memory)
        mha_tokens = analyzer.compute_mha_equivalent_tokens(memory)

        assert mla_tokens > mha_tokens
        ratio = mla_tokens / max(1, mha_tokens)
        # Should be approximately compression_ratio
        assert abs(ratio - v3_config.compression_ratio) / v3_config.compression_ratio < 0.01

    def test_memory_for_tokens(self, v3_config):
        analyzer = LatentCacheAnalyzer(v3_config)

        memory = analyzer.compute_memory_for_tokens(1000)
        expected = 1000 * v3_config.latent_dim * 2 * v3_config.num_layers
        assert memory == expected

    def test_recommend_pool_size(self, v3_config):
        analyzer = LatentCacheAnalyzer(v3_config)

        result = analyzer.recommend_pool_size(
            gpu_memory_bytes=80 * 1024 ** 3,  # 80 GB
            mem_fraction_static=0.8,
            model_weight_bytes=40 * 1024 ** 3,  # 40 GB weights
        )

        assert result["mla_max_tokens"] > result["mha_max_tokens"]
        assert result["compression_ratio"] > 10
        assert result["recommended_pool_size"] > 0

    def test_prefix_cache_benefit(self, v3_config):
        analyzer = LatentCacheAnalyzer(v3_config)

        result = analyzer.estimate_prefix_cache_benefit(
            avg_shared_prefix_len=2048,
            num_concurrent_requests=100,
            pool_size=500000,
        )

        assert result["total_prefix_tokens"] == 2048 * 100
        assert result["tokens_saved_per_request"] > 0


# ────────────────────────────────────────────────────────────────────
# Cache statistics tests
# ────────────────────────────────────────────────────────────────────

class TestMLACacheStats:

    def test_stats_tracking(self, cache):
        # Insert
        cache.insert([1, 2, 3, 4, 5], torch.arange(5, dtype=torch.int64))

        # Match (hit)
        cache.match_prefix([1, 2, 3])

        # Match (miss)
        cache.match_prefix([6, 7, 8])

        stats = cache.get_stats()
        assert int(stats["total_requests"]) == 2
        assert float(stats["hit_rate"]) == 0.5
        assert int(stats["cached_tokens"]) > 0

    def test_memory_tracking(self, cache):
        cache.insert([1, 2, 3, 4, 5], torch.arange(5, dtype=torch.int64))

        stats = cache.stats
        assert stats.latent_bytes_used > 0
        assert stats.latent_bytes_saved > 0
        # Savings should be significantly more than usage (due to compression)
        assert stats.latent_bytes_saved > stats.latent_bytes_used


# ────────────────────────────────────────────────────────────────────
# Integration-style tests
# ────────────────────────────────────────────────────────────────────

class TestMLARadixCacheIntegration:

    def test_system_prompt_reuse(self, cache):
        """Simulate system prompt reuse across multiple requests."""
        system_prompt = list(range(100, 200))  # 100-token system prompt
        system_slots = torch.arange(100, dtype=torch.int64)

        # First request: cold miss
        cache.insert(system_prompt, system_slots)

        # Subsequent requests with different user messages
        for i in range(10):
            user_tokens = list(range(200 + i * 10, 210 + i * 10))
            full_prompt = system_prompt + user_tokens
            result = cache.match_prefix(full_prompt)
            assert result.matched_len == 100  # system prompt cached

        assert cache.stats.hit_rate >= 0.9

    def test_multi_turn_chat(self, cache):
        """Simulate multi-turn chat with growing context."""
        context = list(range(100, 110))  # initial system prompt
        slots = torch.arange(len(context), dtype=torch.int64)
        cache.insert(context, slots)

        for turn in range(5):
            # Add new user + assistant messages
            new_tokens = list(range(200 + turn * 20, 220 + turn * 20))
            context = context + new_tokens

            # Check prefix hit
            result = cache.match_prefix(context)
            hit_len = result.matched_len

            # Insert full sequence
            full_slots = torch.arange(len(context), dtype=torch.int64)
            prefix_len = cache.insert(context, full_slots)

            # Each turn should hit more prefix
            assert prefix_len >= hit_len

    def test_concurrent_requests_eviction(self, cache):
        """Simulate many concurrent requests competing for cache space."""
        # Fill cache with many different prefixes
        for i in range(100):
            tokens = list(range(i * 100, i * 100 + 50))
            slots = torch.arange(50, dtype=torch.int64) + i * 50
            cache.insert(tokens, slots)

        total_before = cache.total_size()
        assert total_before > 0

        # Evict half
        cache.evict(total_before // 2)
        total_after = cache.total_size()
        assert total_after < total_before
        assert total_after > 0  # didn't evict everything


# ────────────────────────────────────────────────────────────────────
# SGLang integration tests
# ────────────────────────────────────────────────────────────────────

class TestSGLangIntegration:

    def test_detect_mla_config(self):
        from sglang_integration import detect_mla_config

        class MockModelConfig:
            kv_lora_rank = 512
            qk_rope_head_dim = 64
            num_attention_heads = 128
            qk_nope_head_dim = 128
            v_head_dim = 128
            num_hidden_layers = 61

        config = detect_mla_config(MockModelConfig())
        assert config is not None
        assert config.kv_lora_rank == 512

    def test_detect_non_mla(self):
        from sglang_integration import detect_mla_config

        class MockModelConfig:
            kv_lora_rank = 0

        config = detect_mla_config(MockModelConfig())
        assert config is None

    def test_compute_mla_pool_size(self, v3_config):
        from sglang_integration import compute_mla_pool_size

        pool_size = compute_mla_pool_size(
            available_memory_bytes=10 * 1024 ** 3,  # 10 GB
            model_config=v3_config,
        )
        assert pool_size > 0
        # Should be significantly more than MHA equivalent
        analyzer = LatentCacheAnalyzer(v3_config)
        mha_tokens = analyzer.compute_mha_equivalent_tokens(10 * 1024 ** 3)
        assert pool_size > mha_tokens * 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
