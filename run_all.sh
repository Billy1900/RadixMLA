#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# MLA-aware RadixAttention — GPU Validation & Benchmark Runner
# ═══════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   - NVIDIA A100 (or equivalent with ≥40GB VRAM)
#   - pip install 'sglang[all]' torch
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh                    # run everything
#   ./run_all.sh --phase 3          # only Phase 3 (validation)
#   ./run_all.sh --phase 4          # only Phase 4 (benchmark)
#   ./run_all.sh --model <path>     # use a different model
# ═══════════════════════════════════════════════════════════════════

set -e

MODEL="${MODEL:-deepseek-ai/DeepSeek-V2-Lite}"
TP="${TP:-1}"
PHASE="${1:---all}"

echo "════════════════════════════════════════════════════════════════"
echo "  MLA-aware RadixAttention — GPU Runner"
echo "  Model: $MODEL"
echo "  TP: $TP"
echo "════════════════════════════════════════════════════════════════"

# ── Phase 0: Verify environment ──
echo ""
echo "▶ Checking environment..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import sglang; print(f'  SGLang: {sglang.__version__}')" 2>/dev/null || {
    echo "  ⚠ SGLang not installed. Run: pip install 'sglang[all]'"
    exit 1
}

# ── Phase 1: Unit tests (CPU) ──
echo ""
echo "▶ Running unit tests..."
python3 -m pytest test_mla_radix_cache.py -v --tb=short
echo "  ✓ Unit tests passed"

# ── Phase 2: CPU benchmarks ──
echo ""
echo "▶ Running CPU benchmarks..."
python3 bench_mla_radix_cache.py
echo "  ✓ CPU benchmarks done"

if [ "$PHASE" = "--phase" ] && [ "$2" = "3" ]; then
    PHASE="3"
elif [ "$PHASE" = "--phase" ] && [ "$2" = "4" ]; then
    PHASE="4"
else
    PHASE="all"
fi

# ── Phase 3: GPU validation ──
if [ "$PHASE" = "all" ] || [ "$PHASE" = "3" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Phase 3: GPU Correctness Validation"
    echo "════════════════════════════════════════════════════════════════"
    python3 gpu_validation.py --mode validate --model "$MODEL" --tp "$TP"
    echo "  ✓ Phase 3 complete"
fi

# ── Phase 4: E2E benchmark ──
if [ "$PHASE" = "all" ] || [ "$PHASE" = "4" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  Phase 4: End-to-End Benchmark"
    echo "════════════════════════════════════════════════════════════════"
    python3 e2e_benchmark.py --mode engine --model "$MODEL" --tp "$TP" --num-prompts 200
    echo "  ✓ Phase 4 complete"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All done! Results in:"
echo "    validation_results/"
echo "    benchmark_results/"
echo "════════════════════════════════════════════════════════════════"
