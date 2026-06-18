# LLMCompass vs HuggingFace Transformers GPU Execution Latency Comparison

## Motivation

> The raw CUDA kernel benchmark (benchmark_real_gpu.py) showed that real GPU
> execution was significantly faster than LLMCompass predictions (~55-68% overestimation).
> This is likely because LLMCompass's tuning parameters are calibrated for a specific
> software stack overhead level. By using HuggingFace Transformers model layers
> (nn.Linear, LlamaRMSNorm, SiLU) instead of raw CUDA operations, we add realistic
> software stack overhead that may better match LLMCompass's assumptions.
>
> Key differences from raw benchmark:
> - **Linear layers**: HF nn.Linear (F.linear with weight transpose + module dispatch) vs raw torch.matmul
> - **LayerNorm**: LlamaRMSNorm (multi-op Python: cast→pow→mean→rsqrt→mul→cast) vs F.layer_norm (fused kernel)
> - **Activation**: nn.SiLU vs F.gelu(approximate='tanh')

## Configuration

**Model:** Llama-3.1-8B
**Sequence Length (KV cache context):** 8192
**Number of Layers:** 32
**Mode:** Decode (auto-regression, seq_len=1 token)
**Excluded operations:** QxK, Softmax, AxV (multi-head attention)
**LLMCompass config:** H100 (GH100_80GB), heuristic-GPU, no_QK_AV_softmax target
**LLMCompass reported total source:** raw no_QK_AV_softmax total
**Real GPU:** NVIDIA H100 PCIe
**Benchmark type:** huggingface_transformers
**Real GPU timing:** cuda_events_median

## 1. Per-Operator Latency Breakdown (per layer, microseconds)

### QKV Proj (HF nn.Linear)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 78.13 | 79.47 | -1.7% |
| 2 | 78.49 | 79.49 | -1.3% |
| 4 | 78.51 | 78.82 | -0.4% |
| 8 | 78.54 | 77.94 | +0.8% |
| 16 | 78.66 | 77.79 | +1.1% |
| 32 | 78.97 | 85.78 | -7.9% |
| 64 | 79.54 | 93.44 | -14.9% |
| 128 | 80.43 | 79.79 | +0.8% |

### Output Proj W0 (HF o_proj)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 31.09 | 26.46 | +17.5% |
| 2 | 31.24 | 26.27 | +18.9% |
| 4 | 31.25 | 25.98 | +20.3% |
| 8 | 31.26 | 25.22 | +24.0% |
| 16 | 31.32 | 25.50 | +22.8% |
| 32 | 31.43 | 30.91 | +1.7% |
| 64 | 31.65 | 26.78 | +18.2% |
| 128 | 32.02 | 29.57 | +8.3% |

### FFN Up W1 (HF gate_proj)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 56.30 | 79.71 | -29.4% |
| 2 | 56.53 | 79.60 | -29.0% |
| 4 | 56.54 | 80.16 | -29.5% |
| 8 | 56.61 | 84.58 | -33.1% |
| 16 | 56.70 | 85.50 | -33.7% |
| 32 | 56.89 | 82.08 | -30.7% |
| 64 | 57.31 | 83.01 | -31.0% |
| 128 | 58.08 | 86.50 | -32.9% |

### FFN Down W2 (HF down_proj)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 56.30 | 83.81 | -32.8% |
| 2 | 56.54 | 84.18 | -32.8% |
| 4 | 56.58 | 84.45 | -33.0% |
| 8 | 56.65 | 89.06 | -36.4% |
| 16 | 56.83 | 85.55 | -33.6% |
| 32 | 57.13 | 87.26 | -34.5% |
| 64 | 57.74 | 91.02 | -36.6% |
| 128 | 58.84 | 89.84 | -34.5% |

### LayerNorm (LlamaRMSNorm)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 45.14 | 60.16 | -25.0% |
| 2 | 45.15 | 59.42 | -24.0% |
| 4 | 45.16 | 59.89 | -24.6% |
| 8 | 45.20 | 59.68 | -24.3% |
| 16 | 45.39 | 60.06 | -24.4% |
| 32 | 45.47 | 60.10 | -24.3% |
| 64 | 45.62 | 76.83 | -40.6% |
| 128 | 45.94 | 59.74 | -23.1% |

### Activation (SiLU)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 45.06 | 12.54 | +259.2% |
| 2 | 45.06 | 12.59 | +257.8% |
| 4 | 45.11 | 12.61 | +257.8% |
| 8 | 45.22 | 12.74 | +255.1% |
| 16 | 45.39 | 12.51 | +262.7% |
| 32 | 45.77 | 12.74 | +259.4% |
| 64 | 46.54 | 17.28 | +169.3% |
| 128 | 48.03 | 13.02 | +268.8% |

## 2. Aggregate Latency Per Layer (microseconds)

| Batch Size | Component | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------:|:---------------:|:-----------:|:---------:|
| 1 | GEMM Total | 221.82 | 269.46 | -17.7% |
| 1 | Norm Total | 90.28 | 120.32 | -25.0% |
| 1 | Activation | 45.06 | 12.54 | +259.2% |
| **1** | **Layer Total** | **357.15** | **402.32** | **-11.2%** |
| 2 | GEMM Total | 222.80 | 269.54 | -17.3% |
| 2 | Norm Total | 90.30 | 118.85 | -24.0% |
| 2 | Activation | 45.06 | 12.59 | +257.8% |
| **2** | **Layer Total** | **358.15** | **400.98** | **-10.7%** |
| 4 | GEMM Total | 222.87 | 269.41 | -17.3% |
| 4 | Norm Total | 90.33 | 119.78 | -24.6% |
| 4 | Activation | 45.11 | 12.61 | +257.8% |
| **4** | **Layer Total** | **358.30** | **401.79** | **-10.8%** |
| 8 | GEMM Total | 223.06 | 276.78 | -19.4% |
| 8 | Norm Total | 90.39 | 119.36 | -24.3% |
| 8 | Activation | 45.22 | 12.74 | +255.1% |
| **8** | **Layer Total** | **358.68** | **408.88** | **-12.3%** |
| 16 | GEMM Total | 223.52 | 274.35 | -18.5% |
| 16 | Norm Total | 90.77 | 120.13 | -24.4% |
| 16 | Activation | 45.39 | 12.51 | +262.7% |
| **16** | **Layer Total** | **359.68** | **406.99** | **-11.6%** |
| 32 | GEMM Total | 224.42 | 286.03 | -21.5% |
| 32 | Norm Total | 90.93 | 120.19 | -24.3% |
| 32 | Activation | 45.77 | 12.74 | +259.4% |
| **32** | **Layer Total** | **361.12** | **418.96** | **-13.8%** |
| 64 | GEMM Total | 226.23 | 294.26 | -23.1% |
| 64 | Norm Total | 91.25 | 153.66 | -40.6% |
| 64 | Activation | 46.54 | 17.28 | +169.3% |
| **64** | **Layer Total** | **364.02** | **465.20** | **-21.8%** |
| 128 | GEMM Total | 229.38 | 285.70 | -19.7% |
| 128 | Norm Total | 91.88 | 119.49 | -23.1% |
| 128 | Activation | 48.03 | 13.02 | +268.8% |
| **128** | **Layer Total** | **369.28** | **418.21** | **-11.7%** |

## 3. Total Model Latency (all layers, milliseconds)

| Batch Size | LLMCompass (ms) | HF GPU (ms) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 11.4289 | 12.8742 | -11.2% |
| 2 | 11.4608 | 12.8312 | -10.7% |
| 4 | 11.4657 | 12.8573 | -10.8% |
| 8 | 11.4776 | 13.0842 | -12.3% |
| 16 | 11.5096 | 13.0237 | -11.6% |
| 32 | 11.5558 | 13.4067 | -13.8% |
| 64 | 11.6485 | 14.8864 | -21.8% |
| 128 | 11.8170 | 13.3827 | -11.7% |

## 4. Summary Statistics

- **Mean absolute error (per-layer total):** 13.0%
- **Min absolute error:** 10.7%
- **Max absolute error:** 21.8%

## 5. Comparison with Raw CUDA Benchmark

| Metric | Raw CUDA Benchmark | HF Transformers Benchmark |
|:------:|:------------------:|:-------------------------:|
| Mean error (layer total) | ~64.9% (from prior report) | 13.0% |
| Min error | ~54.6% | 10.7% |
| Max error | ~68.2% | 21.8% |
| Linear layers | torch.matmul | HF nn.Linear (F.linear) |
| Normalization | F.layer_norm | LlamaRMSNorm (Python multi-op) |
| Activation | F.gelu(tanh) | nn.SiLU |

### Notes

- LLMCompass latency is from cycle-accurate simulation targeting H100 (GH100_80GB).
- The comparison total is `QKV + W0 + W1 + W2 + 2*LayerNorm + Activation`; QxK, Softmax, and AxV are excluded from both sides.
- Each operation was measured with 50 warmup iterations and 200 measurement iterations (median of CUDA events).
- LlamaRMSNorm performs: cast_fp32 → x.pow(2).mean() → rsqrt → multiply → cast_back, which is multiple CUDA kernels vs the fused F.layer_norm.
- Llama-3.1-8B uses SiLU activation (not GeLU). FFN Up uses gate_proj (one of two [4096,14336] projections in Llama's gated MLP).
- Positive error means LLMCompass overestimates; negative means it underestimates.
