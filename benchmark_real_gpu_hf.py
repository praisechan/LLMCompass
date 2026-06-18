"""
HuggingFace Transformers-based GPU benchmark for LLMCompass comparison.

Uses actual HuggingFace Llama model layers (nn.Linear from LlamaAttention,
LlamaRMSNorm, SiLU activation) to measure per-operator decode latency.
The HuggingFace module dispatch overhead provides more realistic latency
numbers that may better match LLMCompass cycle-accurate simulation results.

Compared to benchmark_real_gpu.py (raw CUDA kernels), this script adds:
- nn.Module __call__ -> forward dispatch overhead
- LlamaRMSNorm (multi-op Python impl: cast->pow->mean->rsqrt->mul->cast)
  vs F.layer_norm (single fused CUDA kernel)
- nn.SiLU activation vs F.gelu
- F.linear (with weight transpose) vs torch.matmul

Outputs the same CSV format as benchmark_real_gpu.py for comparison reuse.
"""

import torch
import statistics
import csv
import os
import argparse
import subprocess

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# ---- Model config: Llama-3.1-8B ----
D_MODEL = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS  # 128
ATTN_DIM = N_HEADS * HEAD_DIM  # 4096
KV_DIM = N_KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE_DIM = 14336
N_LAYERS = 32
DEVICE_COUNT = 1  # single GPU, no tensor parallelism

# Warmup and measurement iterations
WARMUP_ITERS = 50
MEASURE_ITERS = 200


def _run_command(args):
    return subprocess.run(args, check=False, text=True, capture_output=True)


def check_gpu_idle(gpu_index, max_used_memory_mb, max_gpu_util, allow_busy):
    """Print nvidia-smi state and optionally reject a busy GPU."""
    query = _run_command([
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if query.returncode != 0:
        message = query.stderr.strip() or "nvidia-smi query failed"
        if allow_busy:
            print(f"WARNING: could not verify GPU idleness: {message}")
            return None
        raise RuntimeError(f"Could not verify GPU idleness with nvidia-smi: {message}")

    raw = query.stdout.strip()
    print("nvidia-smi GPU snapshot:")
    print(raw)
    index, name, used_mb, total_mb, util = [part.strip() for part in raw.split(",")]
    used_mb = int(used_mb)
    total_mb = int(total_mb)
    util = int(util)

    apps = _run_command([
        "nvidia-smi",
        "-i",
        str(gpu_index),
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    active_apps = apps.stdout.strip()
    if active_apps:
        print("Active compute processes on selected GPU:")
        print(active_apps)
    else:
        print("Active compute processes on selected GPU: none")

    is_busy = used_mb > max_used_memory_mb or util > max_gpu_util or bool(active_apps)
    if is_busy and not allow_busy:
        raise RuntimeError(
            "Selected GPU is not idle. "
            f"memory_used={used_mb} MiB/{total_mb} MiB, utilization={util}%, "
            f"active_processes={'yes' if active_apps else 'no'}. "
            "Pick an empty GPU or pass --allow-busy-gpu."
        )

    if is_busy:
        print("WARNING: selected GPU is busy; continuing because --allow-busy-gpu was set.")

    return name


def benchmark_cuda_op(op, device):
    """Benchmark a CUDA operation with events, returning median latency in seconds."""
    # Warmup
    for _ in range(WARMUP_ITERS):
        op()
    torch.cuda.synchronize(device)

    latencies_ms = []
    for _ in range(MEASURE_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        op()
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end))

    return statistics.median(latencies_ms) / 1e3


def create_llama_decoder_layer(device, dtype=torch.float16):
    """Create a single LlamaDecoderLayer with random weights (no download needed)."""
    config = LlamaConfig(
        hidden_size=D_MODEL,
        intermediate_size=INTERMEDIATE_DIM,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        rms_norm_eps=1e-5,
        hidden_act="silu",
    )
    layer = LlamaDecoderLayer(config, layer_idx=0)
    layer = layer.to(device=device, dtype=dtype)
    layer.eval()
    return layer


def run_benchmark(batch_size, device, decoder_layer):
    """Run all operator benchmarks for one batch size using HF modules."""
    b = batch_size
    # In auto-regression (decode), query token count is 1, so M = b*1 = b
    results = {}

    # Input tensors matching the raw benchmark shapes
    x_hidden = torch.randn(b, D_MODEL, dtype=torch.float16, device=device)
    x_hidden_3d = torch.randn(b, 1, D_MODEL, dtype=torch.float16, device=device)
    x_intermediate = torch.randn(b, INTERMEDIATE_DIM, dtype=torch.float16, device=device)
    x_intermediate_3d = torch.randn(b, 1, INTERMEDIATE_DIM, dtype=torch.float16, device=device)

    # Extract HF modules
    q_proj = decoder_layer.self_attn.q_proj
    k_proj = decoder_layer.self_attn.k_proj
    v_proj = decoder_layer.self_attn.v_proj
    o_proj = decoder_layer.self_attn.o_proj
    gate_proj = decoder_layer.mlp.gate_proj
    down_proj = decoder_layer.mlp.down_proj
    input_layernorm = decoder_layer.input_layernorm
    act_fn = decoder_layer.mlp.act_fn

    with torch.no_grad():
        # --- Linear layers (GEMM) through HF nn.Linear ---
        # Q_proj: [b, 4096] -> [b, 4096]  (nn.Linear forward: F.linear with weight transpose)
        q_lat = benchmark_cuda_op(lambda: q_proj(x_hidden), device)
        results["q_proj"] = q_lat

        # K_proj: [b, 4096] -> [b, 1024]
        k_lat = benchmark_cuda_op(lambda: k_proj(x_hidden), device)
        results["k_proj"] = k_lat

        # V_proj: [b, 4096] -> [b, 1024]
        v_lat = benchmark_cuda_op(lambda: v_proj(x_hidden), device)
        results["v_proj"] = v_lat

        # QKV combined
        results["qkv"] = q_lat + k_lat + v_lat

        # Output projection (W0): [b, 4096] -> [b, 4096]
        h_matmul0_lat = benchmark_cuda_op(lambda: o_proj(x_hidden), device)
        results["h_matmul0"] = h_matmul0_lat

        # FFN Up (W1): gate_proj [b, 4096] -> [b, 14336]
        # (Llama has gate_proj + up_proj; we measure gate_proj as the representative
        #  [4096, 14336] projection to match LLMCompass's single h_matmul1)
        h_matmul1_lat = benchmark_cuda_op(lambda: gate_proj(x_hidden), device)
        results["h_matmul1"] = h_matmul1_lat

        # FFN Down (W2): down_proj [b, 14336] -> [b, 4096]
        h_matmul2_lat = benchmark_cuda_op(lambda: down_proj(x_intermediate), device)
        results["h_matmul2"] = h_matmul2_lat

        # --- Non-linear layers through HF modules ---
        # LayerNorm: LlamaRMSNorm on [b, 1, 4096]
        # (Multi-op Python: cast_fp32 -> pow(2) -> mean -> rsqrt -> mul -> cast_back)
        layernorm_lat = benchmark_cuda_op(lambda: input_layernorm(x_hidden_3d), device)
        results["layernorm"] = layernorm_lat

        # Activation: SiLU on [b, 1, 14336]  (Llama uses SiLU; raw benchmark uses GeLU)
        gelu_lat = benchmark_cuda_op(lambda: act_fn(x_intermediate_3d), device)
        results["gelu"] = gelu_lat

    # --- Totals per layer: QKV + W0 + W1 + W2 + 2*LayerNorm + Activation ---
    matmul_total = results["qkv"] + h_matmul0_lat + h_matmul1_lat + h_matmul2_lat
    norm_total = layernorm_lat * 2  # two RMSNorms per transformer block
    layer_total = matmul_total + norm_total + gelu_lat

    results["matmul_total"] = matmul_total
    results["norm_total"] = norm_total
    results["layer_total"] = layer_total
    results["model_total"] = layer_total * N_LAYERS

    return results


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace Transformers-based GPU benchmark for LLMCompass comparison"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index to use")
    parser.add_argument("--output", type=str, default="hf_gpu_benchmark_results.csv")
    parser.add_argument("--max-used-memory-mb", type=int, default=512)
    parser.add_argument("--max-gpu-util", type=int, default=5)
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Run even if nvidia-smi shows memory, utilization, or active processes.",
    )
    args = parser.parse_args()

    gpu_name = check_gpu_idle(
        args.gpu,
        args.max_used_memory_mb,
        args.max_gpu_util,
        args.allow_busy_gpu,
    )

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(device)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    print("\nCreating HuggingFace LlamaDecoderLayer (random weights, no download)...")
    decoder_layer = create_llama_decoder_layer(device)
    print(f"Memory after layer creation: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_length = 8192  # context length for decode

    all_results = []

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {bs}, Seq length: {seq_length}")
        print(f"{'='*60}")

        results = run_benchmark(bs, device, decoder_layer)

        row = {
            "batch_size": bs,
            "seq_length": seq_length,
            "gpu_name": gpu_name or torch.cuda.get_device_name(device),
            "measurement_method": "cuda_events_median",
            "warmup_iters": WARMUP_ITERS,
            "measure_iters": MEASURE_ITERS,
            "benchmark_type": "huggingface_transformers",
            "q_proj_s": results["q_proj"],
            "k_proj_s": results["k_proj"],
            "v_proj_s": results["v_proj"],
            "qkv_s": results["qkv"],
            "h_matmul0_s": results["h_matmul0"],
            "h_matmul1_s": results["h_matmul1"],
            "h_matmul2_s": results["h_matmul2"],
            "layernorm_s": results["layernorm"],
            "gelu_s": results["gelu"],
            "matmul_total_s": results["matmul_total"],
            "norm_total_s": results["norm_total"],
            "layer_total_s": results["layer_total"],
            "model_total_s": results["model_total"],
        }
        all_results.append(row)

        # Print summary
        print(f"  Q_proj  (HF nn.Linear):    {results['q_proj']*1e6:10.2f} us")
        print(f"  K_proj  (HF nn.Linear):    {results['k_proj']*1e6:10.2f} us")
        print(f"  V_proj  (HF nn.Linear):    {results['v_proj']*1e6:10.2f} us")
        print(f"  QKV total:                 {results['qkv']*1e6:10.2f} us")
        print(f"  H_matmul0 (HF o_proj):     {results['h_matmul0']*1e6:10.2f} us")
        print(f"  H_matmul1 (HF gate_proj):  {results['h_matmul1']*1e6:10.2f} us")
        print(f"  H_matmul2 (HF down_proj):  {results['h_matmul2']*1e6:10.2f} us")
        print(f"  LayerNorm (LlamaRMSNorm):  {results['layernorm']*1e6:10.2f} us")
        print(f"  Activation (SiLU):         {results['gelu']*1e6:10.2f} us")
        print(f"  --- Per layer total:       {results['layer_total']*1e6:10.2f} us")
        print(f"  --- Model total ({N_LAYERS} layers): {results['model_total']*1e3:10.4f} ms")

    # Write CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
