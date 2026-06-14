"""
Real GPU execution latency benchmark for LLMCompass comparison.

Measures per-layer decode (auto-regression) latency for Llama-3.1-8B.
Only measures: linear layers (Q/K/V proj, output proj, FFN up/down),
layernorm, and gelu. Excludes QxK, softmax, AxV (multi-head attention).

Sequence length = 8192 (KV cache context), batch sizes = [1,2,4,8,16,32,64,128].
Reports per-layer latency (single transformer block) and total (x n_layers).
"""

import torch
import torch.nn.functional as F
import statistics
import csv
import os
import argparse
import subprocess

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


def benchmark_matmul(M, K, N, device, dtype=torch.float16):
    """Benchmark a single matmul: [M, K] x [K, N]"""
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    return benchmark_cuda_op(lambda: torch.matmul(A, B), device)


def benchmark_layernorm(shape, device, dtype=torch.float16):
    """Benchmark layernorm on input of given shape."""
    x = torch.randn(shape, dtype=dtype, device=device)
    norm_shape = [shape[-1]]
    return benchmark_cuda_op(lambda: F.layer_norm(x, norm_shape), device)


def benchmark_gelu(shape, device, dtype=torch.float16):
    """Benchmark GELU activation on input of given shape."""
    x = torch.randn(shape, dtype=dtype, device=device)
    return benchmark_cuda_op(lambda: F.gelu(x, approximate="tanh"), device)


def run_benchmark(batch_size, device):
    """Run all operator benchmarks for one batch size. Returns dict of latencies in seconds."""
    b = batch_size
    # In auto-regression, the query token count is 1, so M = b*1 = b
    # The Matmul in LLMCompass flattens [b, 1, d] -> M = b*1 = b
    M = b  # batch_size * seq_len(=1)

    results = {}

    # --- Linear layers (GEMM) ---
    # Q_proj: [b, 4096] x [4096, 4096]
    q_proj_lat = benchmark_matmul(M, D_MODEL, ATTN_DIM // DEVICE_COUNT, device)
    results["q_proj"] = q_proj_lat

    # K_proj: [b, 4096] x [4096, 1024]
    k_proj_lat = benchmark_matmul(M, D_MODEL, KV_DIM // DEVICE_COUNT, device)
    results["k_proj"] = k_proj_lat

    # V_proj: same shape as K_proj
    v_proj_lat = benchmark_matmul(M, D_MODEL, KV_DIM // DEVICE_COUNT, device)
    results["v_proj"] = v_proj_lat

    # QKV combined: q + 2*kv (matches LLMCompass: q_latency + 2*kv_latency)
    results["qkv"] = q_proj_lat + k_proj_lat + v_proj_lat

    # H_matmul0 (output projection): [b, 4096] x [4096, 4096]
    h_matmul0_lat = benchmark_matmul(M, ATTN_DIM // DEVICE_COUNT, D_MODEL, device)
    results["h_matmul0"] = h_matmul0_lat

    # H_matmul1 (FFN up): [b, 4096] x [4096, 14336]
    h_matmul1_lat = benchmark_matmul(M, D_MODEL, INTERMEDIATE_DIM // DEVICE_COUNT, device)
    results["h_matmul1"] = h_matmul1_lat

    # H_matmul2 (FFN down): [b, 14336] x [14336, 4096]
    h_matmul2_lat = benchmark_matmul(M, INTERMEDIATE_DIM // DEVICE_COUNT, D_MODEL, device)
    results["h_matmul2"] = h_matmul2_lat

    # --- Non-linear layers ---
    # LayerNorm: input [b, 1, 4096]
    layernorm_lat = benchmark_layernorm([b, 1, D_MODEL], device)
    results["layernorm"] = layernorm_lat

    # GeLU: input [b, 1, 14336]
    gelu_lat = benchmark_gelu([b, 1, INTERMEDIATE_DIM // DEVICE_COUNT], device)
    results["gelu"] = gelu_lat

    # --- Totals per layer: QKV/W0/W1/W2 + two LayerNorms + GeLU ---
    matmul_total = results["qkv"] + h_matmul0_lat + h_matmul1_lat + h_matmul2_lat
    norm_total = layernorm_lat * 2  # two layernorms per block
    gelu_total = gelu_lat
    layer_total = matmul_total + norm_total + gelu_total

    results["matmul_total"] = matmul_total
    results["norm_total"] = norm_total
    results["layer_total"] = layer_total
    results["model_total"] = layer_total * N_LAYERS

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="GPU device index to use")
    parser.add_argument("--output", type=str, default="real_gpu_benchmark_results.csv")
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
    mem_allocated = torch.cuda.memory_allocated(device) / 1e9
    print(f"Memory allocated: {mem_allocated:.2f} GB")

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_length = 8192  # context length for decode

    all_results = []

    for bs in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Batch size: {bs}, Seq length: {seq_length}")
        print(f"{'='*60}")

        results = run_benchmark(bs, device)

        row = {
            "batch_size": bs,
            "seq_length": seq_length,
            "gpu_name": gpu_name or torch.cuda.get_device_name(device),
            "measurement_method": "cuda_events_median",
            "warmup_iters": WARMUP_ITERS,
            "measure_iters": MEASURE_ITERS,
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
        print(f"  Q_proj:    {results['q_proj']*1e6:10.2f} us")
        print(f"  K_proj:    {results['k_proj']*1e6:10.2f} us")
        print(f"  V_proj:    {results['v_proj']*1e6:10.2f} us")
        print(f"  QKV total: {results['qkv']*1e6:10.2f} us")
        print(f"  H_matmul0: {results['h_matmul0']*1e6:10.2f} us")
        print(f"  H_matmul1: {results['h_matmul1']*1e6:10.2f} us")
        print(f"  H_matmul2: {results['h_matmul2']*1e6:10.2f} us")
        print(f"  LayerNorm: {results['layernorm']*1e6:10.2f} us")
        print(f"  GeLU:      {results['gelu']*1e6:10.2f} us")
        print(f"  --- Per layer total: {results['layer_total']*1e6:10.2f} us")
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
