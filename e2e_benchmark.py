#!/usr/bin/env python3
"""
Phase 4: End-to-End Benchmark for MLA-aware RadixAttention.

Measures the real impact of MLA-aware eviction on SGLang serving:
- Prefix cache hit rate
- TTFT (time to first token)
- Throughput (tokens/s)
- Memory utilization

Experiments:
1. High prefix reuse: ShareGPT dataset (shared system prompts)
2. Synthetic high-sharing: 1000 requests sharing a 2048-token system prompt
3. Synthetic no-sharing: random prompts (regression test)
4. Memory pressure: small pool forcing evictions

Usage:
    # Run all experiments (requires SGLang + GPU):
    python e2e_benchmark.py --model deepseek-ai/DeepSeek-V2-Lite

    # Single experiment:
    python e2e_benchmark.py --experiment sharegpt

    # Generate report from saved results:
    python e2e_benchmark.py --mode report --results-dir benchmark_results
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    description: str

    # Workload
    dataset: str  # "sharegpt", "random", "synthetic_shared"
    num_prompts: int = 500
    random_input_len: Optional[int] = None
    random_output_len: Optional[int] = None

    # Server
    model: str = "deepseek-ai/DeepSeek-V2-Lite"
    tp_size: int = 1
    mem_fraction_static: float = 0.85
    extra_server_args: List[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    config: Dict[str, Any]
    mode: str  # "baseline" or "patched"

    # Performance
    total_time_s: float = 0.0
    request_throughput: float = 0.0  # req/s
    input_throughput: float = 0.0   # tokens/s
    output_throughput: float = 0.0  # tokens/s

    # Latency (ms)
    ttft_avg_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_avg_ms: float = 0.0  # time per output token

    # Cache metrics
    prefix_cache_hit_rate: float = 0.0
    num_evictions: int = 0
    cache_tokens_used: int = 0

    # Raw bench_serving output
    raw_output: str = ""


# ────────────────────────────────────────────────────────────────────
# Experiment definitions
# ────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "sharegpt": ExperimentConfig(
        name="sharegpt",
        description="ShareGPT dataset — natural chat workload with shared system prompts",
        dataset="sharegpt",
        num_prompts=500,
    ),

    "high_sharing": ExperimentConfig(
        name="high_sharing",
        description="Synthetic workload: 500 requests with 1024-token shared prefix + 128 unique",
        dataset="random",
        num_prompts=500,
        random_input_len=1024,
        random_output_len=128,
        extra_server_args=["--random-range-ratio", "1.0"],
    ),

    "no_sharing": ExperimentConfig(
        name="no_sharing",
        description="Regression test: random prompts with no prefix sharing",
        dataset="random",
        num_prompts=300,
        random_input_len=256,
        random_output_len=64,
    ),

    "memory_pressure": ExperimentConfig(
        name="memory_pressure",
        description="Memory pressure: reduced pool to force evictions",
        dataset="sharegpt",
        num_prompts=500,
        mem_fraction_static=0.5,  # Lower → more evictions
    ),

    "long_prefix": ExperimentConfig(
        name="long_prefix",
        description="Long prefix reuse: 2048-token shared prefix",
        dataset="random",
        num_prompts=200,
        random_input_len=2048,
        random_output_len=64,
        extra_server_args=["--random-range-ratio", "1.0"],
    ),
}


# ────────────────────────────────────────────────────────────────────
# Server management
# ────────────────────────────────────────────────────────────────────

def start_server(
    config: ExperimentConfig,
    port: int = 30000,
    patched: bool = False,
    log_file: Optional[str] = None,
) -> subprocess.Popen:
    """Start an SGLang server process."""

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", config.model,
        "--tp", str(config.tp_size),
        "--port", str(port),
        "--trust-remote-code",
        "--mem-fraction-static", str(config.mem_fraction_static),
        "--log-level", "info",
    ]

    if patched:
        # Set env var to signal the MLA patch should be applied
        os.environ["SGLANG_MLA_EVICTION_PATCH"] = "1"
    else:
        os.environ.pop("SGLANG_MLA_EVICTION_PATCH", None)

    cmd.extend(config.extra_server_args)

    logger.info(f"Starting server: {' '.join(cmd)}")

    log_handle = None
    if log_file:
        log_handle = open(log_file, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_handle or subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    wait_for_server(port, timeout=300)

    return proc


def wait_for_server(port: int, timeout: int = 300):
    """Wait until the SGLang server is ready."""
    import requests

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                logger.info(f"Server ready on port {port}")
                return
        except Exception:
            pass
        time.sleep(2)

    raise TimeoutError(f"Server on port {port} not ready after {timeout}s")


def stop_server(proc: subprocess.Popen):
    """Stop a server process."""
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("Server stopped")


# ────────────────────────────────────────────────────────────────────
# Benchmark runner
# ────────────────────────────────────────────────────────────────────

def run_bench_serving(
    config: ExperimentConfig,
    port: int = 30000,
) -> str:
    """Run sglang.bench_serving and return raw output."""

    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--num-prompts", str(config.num_prompts),
    ]

    if config.dataset == "sharegpt":
        cmd.extend(["--dataset-name", "sharegpt"])
    elif config.dataset == "random":
        cmd.extend(["--dataset-name", "random"])
        if config.random_input_len:
            cmd.extend(["--random-input", str(config.random_input_len)])
        if config.random_output_len:
            cmd.extend(["--random-output", str(config.random_output_len)])

    logger.info(f"Running benchmark: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = result.stdout + result.stderr
    logger.info(f"Benchmark output:\n{output[-500:]}")

    return output


def parse_bench_output(raw_output: str) -> Dict[str, float]:
    """Parse sglang.bench_serving output for key metrics."""
    metrics = {}

    for line in raw_output.split("\n"):
        line = line.strip()

        # Request throughput
        if "request throughput" in line.lower():
            try:
                val = float(line.split(":")[-1].strip().split()[0])
                metrics["request_throughput"] = val
            except (ValueError, IndexError):
                pass

        # Input throughput
        if "input token throughput" in line.lower() or (
            "input throughput" in line.lower() and "token" in line.lower()
        ):
            try:
                val = float(line.split(":")[-1].strip().split()[0])
                metrics["input_throughput"] = val
            except (ValueError, IndexError):
                pass

        # Output throughput
        if "output token throughput" in line.lower() or (
            "output throughput" in line.lower() and "token" in line.lower()
        ):
            try:
                val = float(line.split(":")[-1].strip().split()[0])
                metrics["output_throughput"] = val
            except (ValueError, IndexError):
                pass

        # TTFT
        if "ttft" in line.lower() or "time to first token" in line.lower():
            if "avg" in line.lower() or "mean" in line.lower():
                try:
                    val = float(line.split(":")[-1].strip().split()[0])
                    metrics["ttft_avg_ms"] = val * 1000 if val < 10 else val
                except (ValueError, IndexError):
                    pass
            if "p50" in line.lower() or "median" in line.lower():
                try:
                    val = float(line.split(":")[-1].strip().split()[0])
                    metrics["ttft_p50_ms"] = val * 1000 if val < 10 else val
                except (ValueError, IndexError):
                    pass
            if "p99" in line.lower():
                try:
                    val = float(line.split(":")[-1].strip().split()[0])
                    metrics["ttft_p99_ms"] = val * 1000 if val < 10 else val
                except (ValueError, IndexError):
                    pass

        # Total time
        if "total time" in line.lower() or "duration" in line.lower():
            try:
                val = float(line.split(":")[-1].strip().split()[0])
                metrics["total_time_s"] = val
            except (ValueError, IndexError):
                pass

    return metrics


def get_server_cache_metrics(port: int = 30000) -> Dict[str, Any]:
    """Get cache metrics from running server."""
    import requests

    try:
        resp = requests.get(f"http://localhost:{port}/get_server_info", timeout=5)
        info = resp.json()
        return {
            k: v
            for k, v in info.items()
            if "cache" in k.lower() or "evict" in k.lower() or "prefix" in k.lower()
        }
    except Exception as e:
        logger.warning(f"Could not get server metrics: {e}")
        return {}


# ────────────────────────────────────────────────────────────────────
# Experiment runner
# ────────────────────────────────────────────────────────────────────

def run_experiment(
    config: ExperimentConfig,
    results_dir: str = "benchmark_results",
    port: int = 30000,
) -> Dict[str, ExperimentResult]:
    """Run a single experiment: baseline + patched."""

    os.makedirs(results_dir, exist_ok=True)
    results = {}

    for mode in ["baseline", "patched"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {config.name} | Mode: {mode}")
        logger.info(f"Description: {config.description}")
        logger.info(f"{'='*60}")

        is_patched = mode == "patched"

        # Start server
        log_file = f"{results_dir}/{config.name}_{mode}_server.log"
        try:
            proc = start_server(config, port=port, patched=is_patched, log_file=log_file)
        except TimeoutError:
            logger.error(f"Server failed to start for {config.name}/{mode}")
            continue

        try:
            # Warmup
            logger.info("Warmup run...")
            warmup_config = ExperimentConfig(
                name="warmup",
                description="warmup",
                dataset=config.dataset,
                num_prompts=min(50, config.num_prompts),
                random_input_len=config.random_input_len,
                random_output_len=config.random_output_len,
            )
            run_bench_serving(warmup_config, port=port)
            time.sleep(2)

            # Actual benchmark
            logger.info("Benchmark run...")
            raw_output = run_bench_serving(config, port=port)

            # Get metrics
            perf_metrics = parse_bench_output(raw_output)
            cache_metrics = get_server_cache_metrics(port)

            result = ExperimentResult(
                config=asdict(config) if hasattr(config, "__dataclass_fields__") else vars(config),
                mode=mode,
                total_time_s=perf_metrics.get("total_time_s", 0),
                request_throughput=perf_metrics.get("request_throughput", 0),
                input_throughput=perf_metrics.get("input_throughput", 0),
                output_throughput=perf_metrics.get("output_throughput", 0),
                ttft_avg_ms=perf_metrics.get("ttft_avg_ms", 0),
                ttft_p50_ms=perf_metrics.get("ttft_p50_ms", 0),
                ttft_p99_ms=perf_metrics.get("ttft_p99_ms", 0),
                prefix_cache_hit_rate=cache_metrics.get("prefix_cache_hit_rate", 0),
                num_evictions=cache_metrics.get("num_evictions", 0),
                cache_tokens_used=cache_metrics.get("cache_tokens_used", 0),
                raw_output=raw_output,
            )

            results[mode] = result

            # Save individual result
            result_file = f"{results_dir}/{config.name}_{mode}.json"
            with open(result_file, "w") as f:
                # Don't serialize raw_output in JSON (too large)
                save_data = {k: v for k, v in vars(result).items() if k != "raw_output"}
                json.dump(save_data, f, indent=2, default=str)

        finally:
            stop_server(proc)
            time.sleep(5)

    return results


# ────────────────────────────────────────────────────────────────────
# Report generation
# ────────────────────────────────────────────────────────────────────

def generate_report(
    results_dir: str = "benchmark_results",
    output_file: Optional[str] = None,
):
    """Generate a comparison report from saved results."""

    experiments = {}

    for f in Path(results_dir).glob("*.json"):
        if f.name.endswith("_report.json"):
            continue
        with open(f) as fh:
            data = json.load(fh)

        # Extract experiment name and mode from filename
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2:
            exp_name, mode = parts
            if exp_name not in experiments:
                experiments[exp_name] = {}
            experiments[exp_name][mode] = data

    if not experiments:
        logger.warning("No results found in %s", results_dir)
        return

    # Build report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MLA-aware RadixAttention — End-to-End Benchmark Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    summary_data = []

    for exp_name, modes in sorted(experiments.items()):
        report_lines.append(f"\n{'─'*80}")
        report_lines.append(f"Experiment: {exp_name}")

        if "baseline" in modes:
            desc = modes["baseline"].get("config", {}).get("description", "")
            report_lines.append(f"Description: {desc}")

        report_lines.append(f"{'─'*80}")

        baseline = modes.get("baseline", {})
        patched = modes.get("patched", {})

        if baseline and patched:
            # Comparison table
            metrics = [
                ("Request throughput (req/s)", "request_throughput", "{:.2f}"),
                ("Input throughput (tok/s)", "input_throughput", "{:.0f}"),
                ("Output throughput (tok/s)", "output_throughput", "{:.0f}"),
                ("TTFT avg (ms)", "ttft_avg_ms", "{:.1f}"),
                ("TTFT P50 (ms)", "ttft_p50_ms", "{:.1f}"),
                ("TTFT P99 (ms)", "ttft_p99_ms", "{:.1f}"),
                ("Prefix cache hit rate", "prefix_cache_hit_rate", "{:.3f}"),
                ("Evictions", "num_evictions", "{:d}"),
            ]

            report_lines.append(
                f"  {'Metric':<35} {'Baseline':>12} {'Patched':>12} {'Δ':>10}"
            )
            report_lines.append(f"  {'─'*70}")

            row = {"experiment": exp_name}
            for label, key, fmt in metrics:
                b_val = baseline.get(key, 0)
                p_val = patched.get(key, 0)

                b_str = fmt.format(b_val) if b_val else "N/A"
                p_str = fmt.format(p_val) if p_val else "N/A"

                if isinstance(b_val, (int, float)) and isinstance(p_val, (int, float)) and b_val:
                    if "throughput" in key:
                        delta = ((p_val - b_val) / b_val) * 100
                        d_str = f"{delta:+.1f}%"
                    elif "ttft" in key or "tpot" in key:
                        delta = ((p_val - b_val) / b_val) * 100
                        d_str = f"{delta:+.1f}%"
                    elif "eviction" in key:
                        delta = p_val - b_val
                        d_str = f"{delta:+d}"
                    else:
                        delta = p_val - b_val
                        d_str = f"{delta:+.3f}"
                else:
                    d_str = "N/A"

                report_lines.append(
                    f"  {label:<35} {b_str:>12} {p_str:>12} {d_str:>10}"
                )
                row[key] = {"baseline": b_val, "patched": p_val}

            summary_data.append(row)
        else:
            for mode, data in modes.items():
                report_lines.append(f"\n  {mode.upper()}:")
                for k, v in data.items():
                    if k not in ("config", "raw_output"):
                        report_lines.append(f"    {k}: {v}")

    report_lines.append(f"\n{'='*80}")
    report_lines.append("END OF REPORT")
    report_lines.append(f"{'='*80}")

    report_text = "\n".join(report_lines)
    print(report_text)

    # Save
    if output_file is None:
        output_file = f"{results_dir}/benchmark_report.txt"
    with open(output_file, "w") as f:
        f.write(report_text)

    # Also save structured summary
    with open(f"{results_dir}/benchmark_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Report saved to {output_file}")


# ────────────────────────────────────────────────────────────────────
# All-in-one runner (no server — uses Engine API directly)
# ────────────────────────────────────────────────────────────────────

def run_engine_benchmark(
    model_path: str = "deepseek-ai/DeepSeek-V2-Lite",
    tp_size: int = 1,
    results_dir: str = "benchmark_results",
    num_prompts: int = 200,
    mem_fraction_static: float = 0.88,
):
    """Run benchmark in-process using SGLang Engine API.

    This is simpler than the server-based approach — no need to manage
    processes. But it doesn't use bench_serving's full workload generator.
    """
    os.makedirs(results_dir, exist_ok=True)

    try:
        from sglang import Engine
    except ImportError:
        logger.error("SGLang not installed. Install with: pip install 'sglang[all]'")
        return

    from mla_radix_cache import MLAModelConfig

    # Generate synthetic workload: shared prefix + unique suffix
    SYSTEM = (
        "You are a helpful assistant. Answer questions accurately. "
        "If unsure, say so. Keep answers concise. " * 10  # ~100 tokens
    )

    prompts = []
    for i in range(num_prompts):
        prompts.append(f"{SYSTEM}\n\nQuestion {i}: What is {i} times {i+1}?\nAnswer:")

    for mode in ["baseline", "patched"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Engine benchmark — {mode}")
        logger.info(f"{'='*60}")

        engine = Engine(
            model_path=model_path,
            tp_size=tp_size,
            trust_remote_code=True,
            mem_fraction_static=mem_fraction_static,
        )
        asyncio.set_event_loop(asyncio.new_event_loop())

        if mode == "patched":
            from gpu_validation import apply_mla_eviction_patch
            apply_mla_eviction_patch(engine)

        # Warmup
        for p in prompts[:10]:
            engine.generate(p, sampling_params={"max_new_tokens": 16, "temperature": 0})

        # Benchmark
        ttfts = []
        start_total = time.perf_counter()

        for p in prompts:
            t0 = time.perf_counter()
            output = engine.generate(
                p, sampling_params={"max_new_tokens": 32, "temperature": 0}
            )
            t1 = time.perf_counter()
            ttfts.append((t1 - t0) * 1000)

        total_time = time.perf_counter() - start_total

        # Get server info if available
        try:
            info = engine.get_server_info()
        except Exception:
            info = {}

        engine.shutdown()

        # Compute metrics
        import statistics

        avg_ttft = statistics.mean(ttfts)
        p50_ttft = statistics.median(ttfts)
        sorted_ttfts = sorted(ttfts)
        p99_idx = int(len(sorted_ttfts) * 0.99)
        p99_ttft = sorted_ttfts[min(p99_idx, len(sorted_ttfts) - 1)]

        result = {
            "mode": mode,
            "model": model_path,
            "num_prompts": num_prompts,
            "total_time_s": total_time,
            "request_throughput": num_prompts / total_time,
            "ttft_avg_ms": avg_ttft,
            "ttft_p50_ms": p50_ttft,
            "ttft_p99_ms": p99_ttft,
            "server_info": {
                k: v for k, v in info.items()
                if "cache" in str(k).lower() or "evict" in str(k).lower()
            },
        }

        logger.info(f"Results:")
        logger.info(f"  Throughput: {result['request_throughput']:.2f} req/s")
        logger.info(f"  TTFT avg: {avg_ttft:.1f} ms")
        logger.info(f"  TTFT P50: {p50_ttft:.1f} ms")
        logger.info(f"  TTFT P99: {p99_ttft:.1f} ms")

        with open(f"{results_dir}/engine_{mode}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        time.sleep(5)

    # Generate comparison
    _compare_engine_results(results_dir)


def _compare_engine_results(results_dir: str):
    """Compare baseline vs patched engine results."""
    try:
        with open(f"{results_dir}/engine_baseline.json") as f:
            baseline = json.load(f)
        with open(f"{results_dir}/engine_patched.json") as f:
            patched = json.load(f)
    except FileNotFoundError:
        logger.warning("Missing engine result files")
        return

    print("\n" + "=" * 70)
    print("Engine Benchmark Comparison")
    print("=" * 70)

    metrics = [
        ("Throughput (req/s)", "request_throughput"),
        ("TTFT avg (ms)", "ttft_avg_ms"),
        ("TTFT P50 (ms)", "ttft_p50_ms"),
        ("TTFT P99 (ms)", "ttft_p99_ms"),
        ("Total time (s)", "total_time_s"),
    ]

    print(f"  {'Metric':<30} {'Baseline':>12} {'Patched':>12} {'Change':>10}")
    print(f"  {'─'*65}")

    for label, key in metrics:
        b = baseline.get(key, 0)
        p = patched.get(key, 0)
        if b > 0:
            delta = ((p - b) / b) * 100
            print(f"  {label:<30} {b:>11.2f} {p:>11.2f} {delta:>+9.1f}%")
        else:
            print(f"  {label:<30} {b:>11.2f} {p:>11.2f} {'N/A':>10}")

    print("=" * 70)


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: End-to-End Benchmark for MLA-aware RadixAttention"
    )
    parser.add_argument(
        "--mode",
        choices=["server", "engine", "report"],
        default="engine",
        help=(
            "server: start/stop SGLang servers and use bench_serving; "
            "engine: in-process benchmark using Engine API; "
            "report: generate comparison report from saved results"
        ),
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Model path",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--mem-fraction", type=float, default=0.88,
                        help="mem_fraction_static for Engine (lower = smaller KV pool = more eviction pressure)")
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run (server mode only)",
    )
    parser.add_argument("--results-dir", default="benchmark_results")
    parser.add_argument("--port", type=int, default=30000)

    args = parser.parse_args()

    # Python 3.10+ requires an explicit event loop for sglang Engine
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    if args.mode == "report":
        generate_report(results_dir=args.results_dir)

    elif args.mode == "engine":
        run_engine_benchmark(
            model_path=args.model,
            tp_size=args.tp,
            results_dir=args.results_dir,
            num_prompts=args.num_prompts,
            mem_fraction_static=args.mem_fraction,
        )

    elif args.mode == "server":
        exp_names = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

        for exp_name in exp_names:
            config = EXPERIMENTS[exp_name]
            config.model = args.model
            config.tp_size = args.tp

            try:
                run_experiment(config, results_dir=args.results_dir, port=args.port)
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}", exc_info=True)

        # Generate final report
        generate_report(results_dir=args.results_dir)


if __name__ == "__main__":
    main()
