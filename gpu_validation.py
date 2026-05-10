#!/usr/bin/env python3
"""
Phase 3: GPU Validation for MLA-aware RadixAttention.

This script validates that MLA-aware eviction produces CORRECT outputs
by running DeepSeek-V2-Lite with and without the patch and comparing
generated tokens.

Requirements:
    pip install sglang[all] torch
    # Enough GPU memory for DeepSeek-V2-Lite (~16B params, ~32GB in bf16)

Usage:
    # Step 1: Start baseline server (no patch)
    python gpu_validation.py --mode baseline

    # Step 2: Start patched server
    python gpu_validation.py --mode patched

    # Step 3: Compare outputs
    python gpu_validation.py --mode compare

    # Or run all-in-one with the Engine API (no server needed):
    python gpu_validation.py --mode validate
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ────────────────────────────────────────────────────────────────────
# Test prompts — designed to exercise prefix caching
# ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful, harmless, and honest AI assistant. "
    "You answer questions accurately and concisely. "
    "If you don't know something, you say so. "
    "You follow instructions carefully and format your responses clearly."
)

# Group 1: Same system prompt, different user messages → tests prefix reuse
SHARED_PREFIX_PROMPTS = [
    f"{SYSTEM_PROMPT}\n\nUser: What is the capital of France?\nAssistant:",
    f"{SYSTEM_PROMPT}\n\nUser: What is 2 + 2?\nAssistant:",
    f"{SYSTEM_PROMPT}\n\nUser: Explain photosynthesis briefly.\nAssistant:",
    f"{SYSTEM_PROMPT}\n\nUser: What color is the sky?\nAssistant:",
    f"{SYSTEM_PROMPT}\n\nUser: Who wrote Romeo and Juliet?\nAssistant:",
]

# Group 2: Unique prompts → tests no-sharing path
UNIQUE_PROMPTS = [
    "Tell me about the history of computers.",
    "What are the main ingredients in bread?",
    "How does gravity work?",
    "List three prime numbers.",
    "What is the speed of light?",
]

# Group 3: Few-shot prompts → tests multi-level prefix sharing
FEW_SHOT_PREFIX = """Classify the sentiment of each review.

Review: "This movie was amazing!" → Positive
Review: "Terrible waste of time." → Negative
Review: "It was okay, nothing special." → Neutral
Review: "Best film I've ever seen!" → Positive
Review: "I fell asleep halfway through." → Negative

"""

FEW_SHOT_PROMPTS = [
    f'{FEW_SHOT_PREFIX}Review: "The acting was superb." →',
    f'{FEW_SHOT_PREFIX}Review: "I want my money back." →',
    f'{FEW_SHOT_PREFIX}Review: "An average experience overall." →',
]

ALL_PROMPTS = SHARED_PREFIX_PROMPTS + UNIQUE_PROMPTS + FEW_SHOT_PROMPTS


# ────────────────────────────────────────────────────────────────────
# Monkey-patch for MLA-aware eviction
# ────────────────────────────────────────────────────────────────────

def apply_mla_eviction_patch(engine_or_scheduler):
    """Apply MLA-aware eviction patch to a running SGLang engine/scheduler.

    Monkey-patches RadixCache.evict() to use MLA-adjusted eviction counts.
    Compatible with sglang 0.4.x where eviction is done via RadixCache.evict(num_tokens).
    """
    try:
        from mla_radix_cache import MLAModelConfig, MLAEvictionBudget
        from sglang_integration import detect_mla_config
    except ImportError:
        from src.mla_radix_cache import MLAModelConfig, MLAEvictionBudget
        from src.sglang_integration import detect_mla_config

    from sglang.srt.mem_cache.radix_cache import RadixCache

    # Detect model config — fall back to DeepSeek-V2-Lite defaults
    model_config = None
    if hasattr(engine_or_scheduler, "model_config"):
        model_config = engine_or_scheduler.model_config

    mla_config = None
    if model_config is not None:
        mla_config = detect_mla_config(model_config)

    if mla_config is None:
        logger.warning("Could not detect MLA config, using DeepSeek-V2-Lite defaults")
        mla_config = MLAModelConfig.deepseek_v2_lite()

    logger.info(
        f"Applying MLA eviction patch: "
        f"compression_ratio={mla_config.compression_ratio:.1f}x, "
        f"latent_dim={mla_config.latent_dim}"
    )

    # Get pool size from the allocator if possible
    pool_size = 100000  # default
    if hasattr(engine_or_scheduler, "token_to_kv_pool_allocator"):
        alloc = engine_or_scheduler.token_to_kv_pool_allocator
        if hasattr(alloc, "size"):
            pool_size = alloc.size

    budget = MLAEvictionBudget(mla_config, pool_size)

    # Patch RadixCache.evict() (sglang 0.4.x API)
    original_evict = RadixCache.evict

    def patched_evict(self, num_tokens: int):
        """MLA-aware eviction wrapper for RadixCache."""
        if self.disable:
            return original_evict(self, num_tokens)

        allocator = self.token_to_kv_pool_allocator
        available = allocator.available_size()

        if available < num_tokens:
            needed = num_tokens - available
            cached = self.evictable_size()
            adjusted = budget.adjust_eviction_count(needed, cached, available)

            logger.debug(
                f"MLA eviction: requested={needed}, adjusted={adjusted}, "
                f"available={available}, cached={cached}"
            )
            return original_evict(self, adjusted)

        return original_evict(self, num_tokens)

    RadixCache.evict = patched_evict
    logger.info("MLA eviction patch applied successfully (via RadixCache.evict)")

    return mla_config


# ────────────────────────────────────────────────────────────────────
# Validation using SGLang Engine API (in-process, no server needed)
# ────────────────────────────────────────────────────────────────────

def run_engine_validation(
    model_path: str = "deepseek-ai/DeepSeek-V2-Lite",
    tp_size: int = 1,
    max_new_tokens: int = 32,
    output_dir: str = "validation_results",
):
    """Run correctness validation using SGLang's Engine API.

    Runs all prompts twice — once baseline, once patched — and
    compares generated tokens for exact match.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        import sglang as sgl
        from sglang import Engine
    except ImportError:
        logger.error(
            "SGLang not installed. Install with: pip install 'sglang[all]'"
        )
        return False

    logger.info(f"Loading model: {model_path} (tp={tp_size})")

    # ── Baseline run ──
    logger.info("=" * 60)
    logger.info("BASELINE RUN (no MLA patch)")
    logger.info("=" * 60)

    engine = Engine(
        model_path=model_path,
        tp_size=tp_size,
        trust_remote_code=True,
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    baseline_results = {}
    for i, prompt in enumerate(ALL_PROMPTS):
        output = engine.generate(
            prompt,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        )
        text = output["text"]
        baseline_results[i] = {
            "prompt": prompt[:100] + "...",
            "output": text,
            "num_tokens": len(text.split()),
        }
        logger.info(f"  [{i}] {text[:80]}...")

    # Save baseline
    with open(f"{output_dir}/baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    # Get baseline metrics
    metrics_baseline = engine.get_server_info().get("cache_stats", {})
    engine.shutdown()

    # ── Patched run ──
    logger.info("=" * 60)
    logger.info("PATCHED RUN (MLA-aware eviction)")
    logger.info("=" * 60)

    engine = Engine(
        model_path=model_path,
        tp_size=tp_size,
        trust_remote_code=True,
    )
    asyncio.set_event_loop(asyncio.new_event_loop())

    # Apply patch
    mla_config = apply_mla_eviction_patch(engine)

    patched_results = {}
    for i, prompt in enumerate(ALL_PROMPTS):
        output = engine.generate(
            prompt,
            sampling_params={"max_new_tokens": max_new_tokens, "temperature": 0},
        )
        text = output["text"]
        patched_results[i] = {
            "prompt": prompt[:100] + "...",
            "output": text,
            "num_tokens": len(text.split()),
        }
        logger.info(f"  [{i}] {text[:80]}...")

    # Save patched
    with open(f"{output_dir}/patched_results.json", "w") as f:
        json.dump(patched_results, f, indent=2)

    metrics_patched = engine.get_server_info().get("cache_stats", {})
    engine.shutdown()

    # ── Compare ──
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    all_match = True
    for i in range(len(ALL_PROMPTS)):
        b = baseline_results[i]["output"]
        p = patched_results[i]["output"]
        match = b == p
        if not match:
            all_match = False
            logger.warning(f"  [{i}] MISMATCH!")
            logger.warning(f"    Baseline: {b[:100]}")
            logger.warning(f"    Patched:  {p[:100]}")
        else:
            logger.info(f"  [{i}] ✓ Match")

    if all_match:
        logger.info("\n✅ ALL OUTPUTS MATCH — patch is correctness-safe")
    else:
        logger.warning(
            "\n⚠️  Some outputs differ. This may be expected if eviction "
            "changes the prefix cache state (different tokens in cache "
            "→ different prefix hit → different computation order → "
            "floating point non-determinism). Check if differences are "
            "semantically equivalent."
        )

    # Save comparison report
    report = {
        "model": model_path,
        "num_prompts": len(ALL_PROMPTS),
        "all_match": all_match,
        "mla_config": {
            "compression_ratio": mla_config.compression_ratio,
            "latent_dim": mla_config.latent_dim,
            "kv_lora_rank": mla_config.kv_lora_rank,
        },
        "baseline_metrics": metrics_baseline,
        "patched_metrics": metrics_patched,
    }
    with open(f"{output_dir}/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}/")
    return all_match


# ────────────────────────────────────────────────────────────────────
# HTTP-based validation (server mode)
# ────────────────────────────────────────────────────────────────────

def run_http_validation(
    base_url: str = "http://localhost:30000",
    max_new_tokens: int = 32,
    output_file: str = "validation_results/http_results.json",
):
    """Run validation against a running SGLang server."""
    import requests

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = {}

    for i, prompt in enumerate(ALL_PROMPTS):
        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("text", "")
        results[i] = {
            "prompt": prompt[:100],
            "output": text,
        }
        logger.info(f"  [{i}] {text[:80]}...")

    # Get metrics
    try:
        metrics = requests.get(f"{base_url}/get_server_info").json()
    except Exception:
        metrics = {}

    with open(output_file, "w") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)

    logger.info(f"Saved to {output_file}")
    return results


def compare_http_results(
    baseline_file: str = "validation_results/baseline_http.json",
    patched_file: str = "validation_results/patched_http.json",
):
    """Compare results from two HTTP runs."""
    with open(baseline_file) as f:
        baseline = json.load(f)["results"]
    with open(patched_file) as f:
        patched = json.load(f)["results"]

    all_match = True
    for key in baseline:
        b = baseline[key]["output"]
        p = patched[key]["output"]
        if b != p:
            all_match = False
            logger.warning(f"[{key}] MISMATCH")
            logger.warning(f"  Baseline: {b[:100]}")
            logger.warning(f"  Patched:  {p[:100]}")
        else:
            logger.info(f"[{key}] ✓ Match")

    if all_match:
        logger.info("\n✅ ALL OUTPUTS MATCH")
    else:
        logger.warning("\n⚠️  Some outputs differ")

    return all_match


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: GPU Validation for MLA-aware RadixAttention"
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "baseline", "patched", "compare"],
        default="validate",
        help=(
            "validate: run both baseline and patched in-process (Engine API); "
            "baseline/patched: run against HTTP server; "
            "compare: compare saved HTTP results"
        ),
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model path (default: DeepSeek-V2-Lite)",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--max-new-tokens", type=int, default=32, help="Max tokens to generate"
    )
    parser.add_argument(
        "--output-dir", default="validation_results", help="Output directory"
    )
    parser.add_argument(
        "--server-url", default="http://localhost:30000", help="Server URL for HTTP mode"
    )

    args = parser.parse_args()

    # Python 3.10+ requires an explicit event loop for sglang Engine
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    if args.mode == "validate":
        success = run_engine_validation(
            model_path=args.model,
            tp_size=args.tp,
            max_new_tokens=args.max_new_tokens,
            output_dir=args.output_dir,
        )
        sys.exit(0 if success else 1)

    elif args.mode == "baseline":
        run_http_validation(
            base_url=args.server_url,
            max_new_tokens=args.max_new_tokens,
            output_file=f"{args.output_dir}/baseline_http.json",
        )

    elif args.mode == "patched":
        run_http_validation(
            base_url=args.server_url,
            max_new_tokens=args.max_new_tokens,
            output_file=f"{args.output_dir}/patched_http.json",
        )

    elif args.mode == "compare":
        success = compare_http_results(
            baseline_file=f"{args.output_dir}/baseline_http.json",
            patched_file=f"{args.output_dir}/patched_http.json",
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
