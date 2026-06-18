"""
Compare LLMCompass simulation results with HuggingFace Transformers-based
GPU execution latency.

Reads:
  1. LLMCompass simulation CSV from change_core_atlas2_for_realGPUcomparison.py
  2. HF-based GPU benchmark CSV from benchmark_real_gpu_hf.py

Outputs: comparison_report_hf.md
"""

import csv
import os
import re
import subprocess

# --- Paths ---
LLMCOMPASS_CSV = "./inference_LUT_with_masked_ICCAD/H100/Llama-3.1-8B/core_size_results_ar_valkyrie.csv"
REAL_GPU_CSV = "./hf_gpu_benchmark_results.csv"
REPORT_OUTPUT = "./comparison_report_hf.md"

# Model config
N_LAYERS = 32
MODEL_NAME = "Llama-3.1-8B"
SEQ_LENGTH = 8192
NATIVE_NO_SOFTMAX = "raw no_QK_AV_softmax total"
LEGACY_SOFTMAX_REMOVED = "recomputed from no_QK_AV log with softmax removed"
RECOMPUTED_TOTAL = "recomputed no-attention total"
LATENCY_TOLERANCE_S = 1e-10


def parse_llmcompass_csv(filepath):
    """Parse LLMCompass auto-regression results CSV."""
    results = {}
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for parts in reader:
            if not parts:
                continue
            parts = [p.strip() for p in parts]
            name = parts[0]
            match = re.match(r"Batch(\d+)_Input(\d+)_Output(\d+)", name)
            if not match:
                continue
            batch_size = int(match.group(1))
            input_seq_length = int(match.group(2))
            output_seq_length = int(match.group(3))

            per_layer_latency = float(parts[2])
            total_latency = float(parts[3])

            qkv = float(parts[4])
            q_mul_k = float(parts[5])
            a_mul_v = float(parts[6])
            h_matmul0 = float(parts[7])
            h1_matmul1 = float(parts[8])
            h2_matmul2 = float(parts[9])
            softmax = float(parts[10])
            layernorm1 = float(parts[11])
            layernorm2 = float(parts[12])
            gelu = float(parts[13])
            allreduce1 = float(parts[14])
            allreduce2 = float(parts[15])

            matmul_total = qkv + h_matmul0 + h1_matmul1 + h2_matmul2
            norm_total = layernorm1 + layernorm2
            layer_total_recomputed = matmul_total + norm_total + gelu
            native_no_qk_av_total = layer_total_recomputed + softmax

            if abs(per_layer_latency - layer_total_recomputed) <= LATENCY_TOLERANCE_S:
                source_total_mode = NATIVE_NO_SOFTMAX
            elif abs(per_layer_latency - native_no_qk_av_total) <= LATENCY_TOLERANCE_S:
                source_total_mode = LEGACY_SOFTMAX_REMOVED
            else:
                source_total_mode = RECOMPUTED_TOTAL

            results[batch_size] = {
                "name": name,
                "input_seq_length": input_seq_length,
                "output_seq_length": output_seq_length,
                "per_layer_latency": per_layer_latency,
                "total_latency": total_latency,
                "qkv": qkv,
                "q_mul_k": q_mul_k,
                "a_mul_v": a_mul_v,
                "h_matmul0": h_matmul0,
                "h1_matmul1": h1_matmul1,
                "h2_matmul2": h2_matmul2,
                "softmax": softmax,
                "layernorm": layernorm1,
                "gelu": gelu,
                "allreduce": allreduce1,
                "matmul_total": matmul_total,
                "norm_total": norm_total,
                "layer_total_no_attn": layer_total_recomputed,
                "native_no_qk_av_total": native_no_qk_av_total,
                "source_total_mode": source_total_mode,
            }

    return results


def parse_real_gpu_csv(filepath):
    """Parse HF-based GPU benchmark results CSV."""
    results = {}
    metadata = {
        "gpu_names": set(),
        "measurement_methods": set(),
        "warmup_iters": set(),
        "measure_iters": set(),
        "benchmark_types": set(),
    }
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_size = int(row["batch_size"])
            if row.get("gpu_name"):
                metadata["gpu_names"].add(row["gpu_name"])
            if row.get("measurement_method"):
                metadata["measurement_methods"].add(row["measurement_method"])
            if row.get("warmup_iters"):
                metadata["warmup_iters"].add(row["warmup_iters"])
            if row.get("measure_iters"):
                metadata["measure_iters"].add(row["measure_iters"])
            if row.get("benchmark_type"):
                metadata["benchmark_types"].add(row["benchmark_type"])
            results[batch_size] = {
                "qkv": float(row["qkv_s"]),
                "q_proj": float(row["q_proj_s"]),
                "k_proj": float(row["k_proj_s"]),
                "v_proj": float(row["v_proj_s"]),
                "h_matmul0": float(row["h_matmul0_s"]),
                "h1_matmul1": float(row["h_matmul1_s"]),
                "h2_matmul2": float(row["h_matmul2_s"]),
                "layernorm": float(row["layernorm_s"]),
                "gelu": float(row["gelu_s"]),
                "matmul_total": float(row["matmul_total_s"]),
                "norm_total": float(row["norm_total_s"]),
                "layer_total": float(row["layer_total_s"]),
                "model_total": float(row["model_total_s"]),
            }
    return results, metadata


def detect_gpu_name_from_nvidia_smi():
    """Best-effort GPU name detection for legacy benchmark CSVs without metadata."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        capture_output=True,
    )
    if query.returncode != 0:
        return "GPU name unavailable"
    names = sorted({line.strip() for line in query.stdout.splitlines() if line.strip()})
    if not names:
        return "GPU name unavailable"
    if len(names) == 1:
        return names[0]
    return ", ".join(names)


def metadata_label(values, fallback):
    clean = sorted(str(v) for v in values if str(v))
    if not clean:
        return fallback
    return clean[0] if len(clean) == 1 else ", ".join(clean)


def fmt_us(val_s):
    """Format seconds as microseconds string."""
    return f"{val_s * 1e6:.2f}"


def fmt_ms(val_s):
    """Format seconds as milliseconds string."""
    return f"{val_s * 1e3:.4f}"


def pct_error(sim, real):
    """Compute percentage error: (sim - real) / real * 100."""
    if real == 0:
        return float('inf')
    return (sim - real) / real * 100


def generate_report(llmcompass, real_gpu, real_gpu_metadata):
    """Generate markdown comparison report."""
    batch_sizes = sorted(set(llmcompass.keys()) & set(real_gpu.keys()))
    source_modes = sorted({llmcompass[bs]["source_total_mode"] for bs in batch_sizes})
    source_mode_label = metadata_label(source_modes, RECOMPUTED_TOTAL)
    gpu_name = metadata_label(real_gpu_metadata["gpu_names"], detect_gpu_name_from_nvidia_smi())
    measurement_method = metadata_label(
        real_gpu_metadata["measurement_methods"],
        "legacy wall-clock synchronization timing",
    )
    warmup_iters = metadata_label(real_gpu_metadata["warmup_iters"], "50")
    measure_iters = metadata_label(real_gpu_metadata["measure_iters"], "200")
    benchmark_type = metadata_label(real_gpu_metadata.get("benchmark_types", set()), "huggingface_transformers")

    lines = []
    lines.append("# LLMCompass vs HuggingFace Transformers GPU Execution Latency Comparison")
    lines.append("")
    lines.append("## Motivation")
    lines.append("")
    lines.append("> The raw CUDA kernel benchmark (benchmark_real_gpu.py) showed that real GPU")
    lines.append("> execution was significantly faster than LLMCompass predictions (~55-68% overestimation).")
    lines.append("> This is likely because LLMCompass's tuning parameters are calibrated for a specific")
    lines.append("> software stack overhead level. By using HuggingFace Transformers model layers")
    lines.append("> (nn.Linear, LlamaRMSNorm, SiLU) instead of raw CUDA operations, we add realistic")
    lines.append("> software stack overhead that may better match LLMCompass's assumptions.")
    lines.append(">")
    lines.append("> Key differences from raw benchmark:")
    lines.append("> - **Linear layers**: HF nn.Linear (F.linear with weight transpose + module dispatch) vs raw torch.matmul")
    lines.append("> - **LayerNorm**: LlamaRMSNorm (multi-op Python: cast→pow→mean→rsqrt→mul→cast) vs F.layer_norm (fused kernel)")
    lines.append("> - **Activation**: nn.SiLU vs F.gelu(approximate='tanh')")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"**Model:** {MODEL_NAME}")
    lines.append(f"**Sequence Length (KV cache context):** {SEQ_LENGTH}")
    lines.append(f"**Number of Layers:** {N_LAYERS}")
    lines.append("**Mode:** Decode (auto-regression, seq_len=1 token)")
    lines.append("**Excluded operations:** QxK, Softmax, AxV (multi-head attention)")
    lines.append("**LLMCompass config:** H100 (GH100_80GB), heuristic-GPU, no_QK_AV_softmax target")
    lines.append(f"**LLMCompass reported total source:** {source_mode_label}")
    lines.append(f"**Real GPU:** {gpu_name}")
    lines.append(f"**Benchmark type:** {benchmark_type}")
    lines.append(f"**Real GPU timing:** {measurement_method}")
    lines.append("")

    # ================================================================
    # Table 1: Per-operator breakdown (per layer, in microseconds)
    # ================================================================
    lines.append("## 1. Per-Operator Latency Breakdown (per layer, microseconds)")
    lines.append("")

    op_names = ["qkv", "h_matmul0", "h1_matmul1", "h2_matmul2", "layernorm", "gelu"]
    op_labels = [
        "QKV Proj (HF nn.Linear)",
        "Output Proj W0 (HF o_proj)",
        "FFN Up W1 (HF gate_proj)",
        "FFN Down W2 (HF down_proj)",
        "LayerNorm (LlamaRMSNorm)",
        "Activation (SiLU)",
    ]

    for op_name, op_label in zip(op_names, op_labels):
        lines.append(f"### {op_label}")
        lines.append("")
        lines.append("| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |")
        lines.append("|:----------:|:---------------:|:-----------:|:---------:|")
        for bs in batch_sizes:
            sim_val = llmcompass[bs][op_name]
            real_val = real_gpu[bs][op_name]
            err = pct_error(sim_val, real_val)
            lines.append(f"| {bs} | {fmt_us(sim_val)} | {fmt_us(real_val)} | {err:+.1f}% |")
        lines.append("")

    # ================================================================
    # Table 2: Aggregate comparison (per layer)
    # ================================================================
    lines.append("## 2. Aggregate Latency Per Layer (microseconds)")
    lines.append("")
    lines.append("| Batch Size | Component | LLMCompass (us) | HF GPU (us) | Error (%) |")
    lines.append("|:----------:|:---------:|:---------------:|:-----------:|:---------:|")
    for bs in batch_sizes:
        s = llmcompass[bs]
        r = real_gpu[bs]
        for comp, label in [("matmul_total", "GEMM Total"), ("norm_total", "Norm Total"), ("gelu", "Activation")]:
            sv = s[comp]
            rv = r[comp]
            err = pct_error(sv, rv)
            lines.append(f"| {bs} | {label} | {fmt_us(sv)} | {fmt_us(rv)} | {err:+.1f}% |")
        # Layer total
        sv = s["layer_total_no_attn"]
        rv = r["layer_total"]
        err = pct_error(sv, rv)
        lines.append(f"| **{bs}** | **Layer Total** | **{fmt_us(sv)}** | **{fmt_us(rv)}** | **{err:+.1f}%** |")
    lines.append("")

    # ================================================================
    # Table 3: Total model latency (all layers, in milliseconds)
    # ================================================================
    lines.append("## 3. Total Model Latency (all layers, milliseconds)")
    lines.append("")
    lines.append("| Batch Size | LLMCompass (ms) | HF GPU (ms) | Error (%) |")
    lines.append("|:----------:|:---------------:|:-----------:|:---------:|")
    for bs in batch_sizes:
        sim_total = llmcompass[bs]["layer_total_no_attn"] * N_LAYERS
        real_total = real_gpu[bs]["model_total"]
        err = pct_error(sim_total, real_total)
        lines.append(f"| {bs} | {fmt_ms(sim_total)} | {fmt_ms(real_total)} | {err:+.1f}% |")
    lines.append("")

    # ================================================================
    # Summary statistics
    # ================================================================
    lines.append("## 4. Summary Statistics")
    lines.append("")

    layer_errors = []
    for bs in batch_sizes:
        sv = llmcompass[bs]["layer_total_no_attn"]
        rv = real_gpu[bs]["layer_total"]
        layer_errors.append(abs(pct_error(sv, rv)))

    avg_err = sum(layer_errors) / len(layer_errors)
    max_err = max(layer_errors)
    min_err = min(layer_errors)

    lines.append(f"- **Mean absolute error (per-layer total):** {avg_err:.1f}%")
    lines.append(f"- **Min absolute error:** {min_err:.1f}%")
    lines.append(f"- **Max absolute error:** {max_err:.1f}%")
    lines.append("")

    # ================================================================
    # Comparison with raw benchmark
    # ================================================================
    lines.append("## 5. Comparison with Raw CUDA Benchmark")
    lines.append("")
    lines.append("| Metric | Raw CUDA Benchmark | HF Transformers Benchmark |")
    lines.append("|:------:|:------------------:|:-------------------------:|")
    lines.append(f"| Mean error (layer total) | ~64.9% (from prior report) | {avg_err:.1f}% |")
    lines.append(f"| Min error | ~54.6% | {min_err:.1f}% |")
    lines.append(f"| Max error | ~68.2% | {max_err:.1f}% |")
    lines.append(f"| Linear layers | torch.matmul | HF nn.Linear (F.linear) |")
    lines.append(f"| Normalization | F.layer_norm | LlamaRMSNorm (Python multi-op) |")
    lines.append(f"| Activation | F.gelu(tanh) | nn.SiLU |")
    lines.append("")

    lines.append("### Notes")
    lines.append("")
    lines.append("- LLMCompass latency is from cycle-accurate simulation targeting H100 (GH100_80GB).")
    if "H100" not in gpu_name:
        lines.append(f"- Real GPU latency was measured on {gpu_name}, not H100. "
                      "Results will differ due to GPU architecture; re-run on H100 for a fair comparison.")
    lines.append("- The comparison total is `QKV + W0 + W1 + W2 + 2*LayerNorm + Activation`; QxK, Softmax, and AxV are excluded from both sides.")
    lines.append(f"- Each operation was measured with {warmup_iters} warmup iterations and {measure_iters} measurement iterations (median of CUDA events).")
    lines.append("- LlamaRMSNorm performs: cast_fp32 → x.pow(2).mean() → rsqrt → multiply → cast_back, which is multiple CUDA kernels vs the fused F.layer_norm.")
    lines.append("- Llama-3.1-8B uses SiLU activation (not GeLU). FFN Up uses gate_proj (one of two [4096,14336] projections in Llama's gated MLP).")
    lines.append("- Positive error means LLMCompass overestimates; negative means it underestimates.")
    lines.append("")

    return "\n".join(lines)


def main():
    if not os.path.exists(LLMCOMPASS_CSV):
        print(f"ERROR: LLMCompass CSV not found at {LLMCOMPASS_CSV}")
        print("Run change_core_atlas2_for_realGPUcomparison.py first.")
        return

    if not os.path.exists(REAL_GPU_CSV):
        print(f"ERROR: HF GPU benchmark CSV not found at {REAL_GPU_CSV}")
        print("Run benchmark_real_gpu_hf.py first.")
        return

    llmcompass = parse_llmcompass_csv(LLMCOMPASS_CSV)
    real_gpu, real_gpu_metadata = parse_real_gpu_csv(REAL_GPU_CSV)

    print(f"LLMCompass batch sizes: {sorted(llmcompass.keys())}")
    print(f"HF GPU batch sizes: {sorted(real_gpu.keys())}")

    report = generate_report(llmcompass, real_gpu, real_gpu_metadata)

    with open(REPORT_OUTPUT, "w") as f:
        f.write(report)

    print(f"\nReport written to {REPORT_OUTPUT}")
    print("\n" + report)


if __name__ == "__main__":
    main()
