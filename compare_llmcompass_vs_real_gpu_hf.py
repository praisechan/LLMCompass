"""
Compare LLMCompass simulation results with HuggingFace Transformers-based
GPU execution latency. Supports multiple models and input lengths.

Usage:
  python compare_llmcompass_vs_real_gpu_hf.py --model Llama-3.1-8B
  python compare_llmcompass_vs_real_gpu_hf.py --model Qwen-2.5-14B
  python compare_llmcompass_vs_real_gpu_hf.py --model Qwen-2.5-32B
"""

import csv
import os
import re
import argparse
import subprocess

# Model config (n_layers needed for total model latency)
MODEL_N_LAYERS = {
    "Llama-3.1-8B": 32,
    "Qwen-2.5-14B": 48,
    "Qwen-2.5-32B": 64,
}

LATENCY_TOLERANCE_S = 1e-10


def parse_llmcompass_csv(filepath):
    """Parse LLMCompass auto-regression results CSV.

    Returns dict keyed by (input_seq_length, batch_size).
    """
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

            matmul_total = qkv + h_matmul0 + h1_matmul1 + h2_matmul2
            norm_total = layernorm1 + layernorm2
            layer_total_recomputed = matmul_total + norm_total + gelu

            results[(input_seq_length, batch_size)] = {
                "name": name,
                "input_seq_length": input_seq_length,
                "qkv": qkv,
                "h_matmul0": h_matmul0,
                "h1_matmul1": h1_matmul1,
                "h2_matmul2": h2_matmul2,
                "layernorm": layernorm1,
                "gelu": gelu,
                "matmul_total": matmul_total,
                "norm_total": norm_total,
                "layer_total_no_attn": layer_total_recomputed,
            }

    return results


def parse_real_gpu_csv(filepath):
    """Parse HF-based GPU benchmark results CSV. Returns dict keyed by batch_size."""
    results = {}
    metadata = {
        "gpu_names": set(),
        "measurement_methods": set(),
        "warmup_iters": set(),
        "measure_iters": set(),
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
            results[batch_size] = {
                "qkv": float(row["qkv_s"]),
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
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        check=False, text=True, capture_output=True,
    )
    if query.returncode != 0:
        return "GPU name unavailable"
    names = sorted({line.strip() for line in query.stdout.splitlines() if line.strip()})
    return names[0] if len(names) == 1 else ", ".join(names) if names else "GPU name unavailable"


def metadata_label(values, fallback):
    clean = sorted(str(v) for v in values if str(v))
    if not clean:
        return fallback
    return clean[0] if len(clean) == 1 else ", ".join(clean)


def fmt_us(val_s):
    return f"{val_s * 1e6:.2f}"


def fmt_ms(val_s):
    return f"{val_s * 1e3:.4f}"


def pct_error(sim, real):
    if real == 0:
        return float('inf')
    return (sim - real) / real * 100


def generate_report(model_name, llmcompass, real_gpu, real_gpu_metadata):
    """Generate markdown comparison report for one model across all input lengths."""
    n_layers = MODEL_N_LAYERS[model_name]
    gpu_name = metadata_label(real_gpu_metadata["gpu_names"], detect_gpu_name_from_nvidia_smi())
    measurement_method = metadata_label(real_gpu_metadata["measurement_methods"], "cuda_events_median")
    warmup_iters = metadata_label(real_gpu_metadata["warmup_iters"], "50")
    measure_iters = metadata_label(real_gpu_metadata["measure_iters"], "200")

    # Discover available input lengths and batch sizes
    input_lengths = sorted({k[0] for k in llmcompass.keys()})
    batch_sizes_real = sorted(real_gpu.keys())

    lines = []
    lines.append(f"# LLMCompass vs HF Transformers: {model_name}")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"**Model:** {model_name}")
    lines.append(f"**Number of Layers:** {n_layers}")
    lines.append("**Mode:** Decode (auto-regression, seq_len=1 token)")
    lines.append("**Excluded operations:** QxK, Softmax, AxV (multi-head attention)")
    lines.append("**LLMCompass config:** H100 (GH100_80GB), heuristic-GPU, no_QK_AV_softmax")
    lines.append(f"**Real GPU:** {gpu_name}")
    lines.append(f"**Real GPU timing:** {measurement_method} ({warmup_iters} warmup, {measure_iters} measurement)")
    lines.append(f"**Input lengths (KV context):** {', '.join(str(l) for l in input_lengths)}")
    lines.append("")

    # Note: real GPU latency is the same for all input lengths in decode mode
    lines.append("> **Note:** In decode mode (seq_len=1), real GPU latency for GEMM/Norm/Activation")
    lines.append("> does not depend on KV cache context length. The same real GPU measurements are")
    lines.append("> compared against each input length's LLMCompass predictions.")
    lines.append("")

    all_layer_errors = []

    for input_len in input_lengths:
        batch_sizes = sorted(
            bs for bs in batch_sizes_real
            if (input_len, bs) in llmcompass
        )
        if not batch_sizes:
            continue

        lines.append(f"## Input Length: {input_len}")
        lines.append("")

        # ---- Per-operator breakdown ----
        lines.append("### Per-Operator Latency (per layer, microseconds)")
        lines.append("")

        op_names = ["qkv", "h_matmul0", "h1_matmul1", "h2_matmul2", "layernorm", "gelu"]
        op_labels = ["QKV Proj", "Output Proj (W0)", "FFN Up (W1)", "FFN Down (W2)", "LayerNorm (RMSNorm)", "Activation (SiLU)"]

        for op_name, op_label in zip(op_names, op_labels):
            lines.append(f"#### {op_label}")
            lines.append("")
            lines.append("| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |")
            lines.append("|:----------:|:---------------:|:-----------:|:---------:|")
            for bs in batch_sizes:
                sim_val = llmcompass[(input_len, bs)][op_name]
                real_val = real_gpu[bs][op_name]
                err = pct_error(sim_val, real_val)
                lines.append(f"| {bs} | {fmt_us(sim_val)} | {fmt_us(real_val)} | {err:+.1f}% |")
            lines.append("")

        # ---- Aggregate per layer ----
        lines.append("### Aggregate Latency Per Layer (microseconds)")
        lines.append("")
        lines.append("| Batch Size | Component | LLMCompass (us) | HF GPU (us) | Error (%) |")
        lines.append("|:----------:|:---------:|:---------------:|:-----------:|:---------:|")
        for bs in batch_sizes:
            s = llmcompass[(input_len, bs)]
            r = real_gpu[bs]
            for comp, label in [("matmul_total", "GEMM Total"), ("norm_total", "Norm Total"), ("gelu", "Activation")]:
                sv = s[comp]
                rv = r[comp]
                err = pct_error(sv, rv)
                lines.append(f"| {bs} | {label} | {fmt_us(sv)} | {fmt_us(rv)} | {err:+.1f}% |")
            sv = s["layer_total_no_attn"]
            rv = r["layer_total"]
            err = pct_error(sv, rv)
            all_layer_errors.append(abs(err))
            lines.append(f"| **{bs}** | **Layer Total** | **{fmt_us(sv)}** | **{fmt_us(rv)}** | **{err:+.1f}%** |")
        lines.append("")

        # ---- Total model latency ----
        lines.append("### Total Model Latency (all layers, milliseconds)")
        lines.append("")
        lines.append("| Batch Size | LLMCompass (ms) | HF GPU (ms) | Error (%) |")
        lines.append("|:----------:|:---------------:|:-----------:|:---------:|")
        for bs in batch_sizes:
            sim_total = llmcompass[(input_len, bs)]["layer_total_no_attn"] * n_layers
            real_total = real_gpu[bs]["model_total"]
            err = pct_error(sim_total, real_total)
            lines.append(f"| {bs} | {fmt_ms(sim_total)} | {fmt_ms(real_total)} | {err:+.1f}% |")
        lines.append("")

    # ---- Summary across all input lengths ----
    lines.append("## Summary Statistics (across all input lengths)")
    lines.append("")
    if all_layer_errors:
        avg_err = sum(all_layer_errors) / len(all_layer_errors)
        min_err = min(all_layer_errors)
        max_err = max(all_layer_errors)
        lines.append(f"- **Mean absolute error (per-layer total):** {avg_err:.1f}%")
        lines.append(f"- **Min absolute error:** {min_err:.1f}%")
        lines.append(f"- **Max absolute error:** {max_err:.1f}%")
        lines.append(f"- **Number of (input_length, batch_size) combinations:** {len(all_layer_errors)}")
    lines.append("")

    # ---- Per input-length summary table ----
    lines.append("### Error by Input Length")
    lines.append("")
    lines.append("| Input Length | Mean Error (%) | Min Error (%) | Max Error (%) |")
    lines.append("|:----------:|:--------------:|:-------------:|:-------------:|")
    for input_len in input_lengths:
        batch_sizes = sorted(
            bs for bs in batch_sizes_real
            if (input_len, bs) in llmcompass
        )
        if not batch_sizes:
            continue
        errs = []
        for bs in batch_sizes:
            sv = llmcompass[(input_len, bs)]["layer_total_no_attn"]
            rv = real_gpu[bs]["layer_total"]
            errs.append(abs(pct_error(sv, rv)))
        lines.append(f"| {input_len} | {sum(errs)/len(errs):.1f}% | {min(errs):.1f}% | {max(errs):.1f}% |")
    lines.append("")

    lines.append("### Notes")
    lines.append("")
    lines.append("- LLMCompass latency is from cycle-accurate simulation targeting H100 (GH100_80GB).")
    if "H100" not in gpu_name:
        lines.append(f"- Real GPU: {gpu_name} (not H100). Re-run on H100 for fair comparison.")
    lines.append("- Comparison total: `QKV + W0 + W1 + W2 + 2*RMSNorm + Activation`; QxK, Softmax, AxV excluded.")
    lines.append(f"- Each op measured with {warmup_iters} warmup + {measure_iters} measurement iterations (median).")
    lines.append("- Positive error = LLMCompass overestimates; negative = underestimates.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_N_LAYERS.keys()),
                        help="Model to compare")
    parser.add_argument("--llmcompass-csv", type=str, default=None,
                        help="LLMCompass CSV path (default: auto from model name)")
    parser.add_argument("--real-gpu-csv", type=str, default=None,
                        help="Real GPU CSV path (default: hf_gpu_benchmark_results_{model}.csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Report output path (default: comparison_report_hf_{model}.md)")
    args = parser.parse_args()

    if args.llmcompass_csv is None:
        args.llmcompass_csv = f"./inference_LUT_with_masked_ICCAD/H100/{args.model}/core_size_results_ar_valkyrie.csv"
    if args.real_gpu_csv is None:
        args.real_gpu_csv = f"./hf_gpu_benchmark_results_{args.model}.csv"
    if args.output is None:
        args.output = f"./comparison_report_hf_{args.model}.md"

    if not os.path.exists(args.llmcompass_csv):
        print(f"ERROR: LLMCompass CSV not found at {args.llmcompass_csv}")
        print("Run change_core_atlas2_for_realGPUcomparison.py first.")
        return

    if not os.path.exists(args.real_gpu_csv):
        print(f"ERROR: HF GPU benchmark CSV not found at {args.real_gpu_csv}")
        print(f"Run: python benchmark_real_gpu_hf.py --model {args.model}")
        return

    llmcompass = parse_llmcompass_csv(args.llmcompass_csv)
    real_gpu, real_gpu_metadata = parse_real_gpu_csv(args.real_gpu_csv)

    input_lengths = sorted({k[0] for k in llmcompass.keys()})
    print(f"Model: {args.model}")
    print(f"LLMCompass input lengths: {input_lengths}")
    print(f"LLMCompass batch sizes: {sorted({k[1] for k in llmcompass.keys()})}")
    print(f"Real GPU batch sizes: {sorted(real_gpu.keys())}")

    report = generate_report(args.model, llmcompass, real_gpu, real_gpu_metadata)

    with open(args.output, "w") as f:
        f.write(report)

    print(f"\nReport written to {args.output}")
    print("\n" + report)


if __name__ == "__main__":
    main()
