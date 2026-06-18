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
**Real GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition
**Benchmark type:** huggingface_transformers
**Real GPU timing:** cuda_events_median

## 1. Per-Operator Latency Breakdown (per layer, microseconds)

### QKV Proj (HF nn.Linear)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 78.13 | 64.24 | +21.6% |
| 2 | 78.49 | 75.89 | +3.4% |
| 4 | 78.51 | 73.62 | +6.6% |
| 8 | 78.54 | 72.64 | +8.1% |
| 16 | 78.66 | 74.99 | +4.9% |
| 32 | 78.97 | 76.02 | +3.9% |
| 64 | 79.54 | 77.34 | +2.8% |
| 128 | 80.43 | 81.28 | -1.0% |

### Output Proj W0 (HF o_proj)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 31.09 | 21.28 | +46.1% |
| 2 | 31.24 | 27.39 | +14.0% |
| 4 | 31.25 | 24.51 | +27.5% |
| 8 | 31.26 | 24.48 | +27.7% |
| 16 | 31.32 | 25.25 | +24.1% |
| 32 | 31.43 | 26.70 | +17.7% |
| 64 | 31.65 | 28.29 | +11.9% |
| 128 | 32.02 | 32.45 | -1.3% |

### FFN Up W1 (HF gate_proj)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 56.30 | 35.50 | +58.6% |
| 2 | 56.53 | 39.63 | +42.6% |
| 4 | 56.54 | 36.58 | +54.6% |
| 8 | 56.61 | 38.45 | +47.2% |
| 16 | 56.70 | 38.59 | +46.9% |
| 32 | 56.89 | 48.93 | +16.3% |
| 64 | 57.31 | 46.40 | +23.5% |
| 128 | 58.08 | 72.34 | -19.7% |

### FFN Down W2 (HF down_proj)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 56.30 | 39.10 | +44.0% |
| 2 | 56.54 | 41.18 | +37.3% |
| 4 | 56.58 | 42.18 | +34.1% |
| 8 | 56.65 | 44.51 | +27.3% |
| 16 | 56.83 | 47.28 | +20.2% |
| 32 | 57.13 | 45.62 | +25.2% |
| 64 | 57.74 | 64.99 | -11.2% |
| 128 | 58.84 | 75.17 | -21.7% |

### LayerNorm (LlamaRMSNorm)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 45.14 | 67.17 | -32.8% |
| 2 | 45.15 | 66.96 | -32.6% |
| 4 | 45.16 | 67.31 | -32.9% |
| 8 | 45.20 | 67.01 | -32.6% |
| 16 | 45.39 | 66.43 | -31.7% |
| 32 | 45.47 | 66.18 | -31.3% |
| 64 | 45.62 | 65.92 | -30.8% |
| 128 | 45.94 | 66.56 | -31.0% |

### Activation (SiLU)

| Batch Size | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 45.06 | 15.52 | +190.3% |
| 2 | 45.06 | 15.68 | +187.3% |
| 4 | 45.11 | 14.83 | +204.1% |
| 8 | 45.22 | 15.52 | +191.4% |
| 16 | 45.39 | 14.91 | +204.4% |
| 32 | 45.77 | 15.62 | +193.1% |
| 64 | 46.54 | 15.94 | +192.1% |
| 128 | 48.03 | 16.16 | +197.2% |

## 2. Aggregate Latency Per Layer (microseconds)

| Batch Size | Component | LLMCompass (us) | HF GPU (us) | Error (%) |
|:----------:|:---------:|:---------------:|:-----------:|:---------:|
| 1 | GEMM Total | 221.82 | 160.13 | +38.5% |
| 1 | Norm Total | 90.28 | 134.34 | -32.8% |
| 1 | Activation | 45.06 | 15.52 | +190.3% |
| **1** | **Layer Total** | **357.15** | **309.98** | **+15.2%** |
| 2 | GEMM Total | 222.80 | 184.10 | +21.0% |
| 2 | Norm Total | 90.30 | 133.92 | -32.6% |
| 2 | Activation | 45.06 | 15.68 | +187.3% |
| **2** | **Layer Total** | **358.15** | **333.70** | **+7.3%** |
| 4 | GEMM Total | 222.87 | 176.88 | +26.0% |
| 4 | Norm Total | 90.33 | 134.62 | -32.9% |
| 4 | Activation | 45.11 | 14.83 | +204.1% |
| **4** | **Layer Total** | **358.30** | **326.34** | **+9.8%** |
| 8 | GEMM Total | 223.06 | 180.08 | +23.9% |
| 8 | Norm Total | 90.39 | 134.02 | -32.6% |
| 8 | Activation | 45.22 | 15.52 | +191.4% |
| **8** | **Layer Total** | **358.68** | **329.62** | **+8.8%** |
| 16 | GEMM Total | 223.52 | 186.11 | +20.1% |
| 16 | Norm Total | 90.77 | 132.86 | -31.7% |
| 16 | Activation | 45.39 | 14.91 | +204.4% |
| **16** | **Layer Total** | **359.68** | **333.89** | **+7.7%** |
| 32 | GEMM Total | 224.42 | 197.26 | +13.8% |
| 32 | Norm Total | 90.93 | 132.35 | -31.3% |
| 32 | Activation | 45.77 | 15.62 | +193.1% |
| **32** | **Layer Total** | **361.12** | **345.23** | **+4.6%** |
| 64 | GEMM Total | 226.23 | 217.02 | +4.2% |
| 64 | Norm Total | 91.25 | 131.84 | -30.8% |
| 64 | Activation | 46.54 | 15.94 | +192.1% |
| **64** | **Layer Total** | **364.02** | **364.80** | **-0.2%** |
| 128 | GEMM Total | 229.38 | 261.23 | -12.2% |
| 128 | Norm Total | 91.88 | 133.12 | -31.0% |
| 128 | Activation | 48.03 | 16.16 | +197.2% |
| **128** | **Layer Total** | **369.28** | **410.51** | **-10.0%** |

## 3. Total Model Latency (all layers, milliseconds)

| Batch Size | LLMCompass (ms) | HF GPU (ms) | Error (%) |
|:----------:|:---------------:|:-----------:|:---------:|
| 1 | 11.4289 | 9.9195 | +15.2% |
| 2 | 11.4608 | 10.6783 | +7.3% |
| 4 | 11.4657 | 10.4428 | +9.8% |
| 8 | 11.4776 | 10.5477 | +8.8% |
| 16 | 11.5096 | 10.6844 | +7.7% |
| 32 | 11.5558 | 11.0474 | +4.6% |
| 64 | 11.6485 | 11.6736 | -0.2% |
| 128 | 11.8170 | 13.1364 | -10.0% |

## 4. Summary Statistics

- **Mean absolute error (per-layer total):** 8.0%
- **Min absolute error:** 0.2%
- **Max absolute error:** 15.2%

## 5. Comparison with Raw CUDA Benchmark

| Metric | Raw CUDA Benchmark | HF Transformers Benchmark |
|:------:|:------------------:|:-------------------------:|
| Mean error (layer total) | ~64.9% (from prior report) | 8.0% |
| Min error | ~54.6% | 0.2% |
| Max error | ~68.2% | 15.2% |
| Linear layers | torch.matmul | HF nn.Linear (F.linear) |
| Normalization | F.layer_norm | LlamaRMSNorm (Python multi-op) |
| Activation | F.gelu(tanh) | nn.SiLU |

### Notes

- LLMCompass latency is from cycle-accurate simulation targeting H100 (GH100_80GB).
- Real GPU latency was measured on NVIDIA RTX PRO 6000 Blackwell Server Edition, not H100. Results will differ due to GPU architecture; re-run on H100 for a fair comparison.
- The comparison total is `QKV + W0 + W1 + W2 + 2*LayerNorm + Activation`; QxK, Softmax, and AxV are excluded from both sides.
- Each operation was measured with 50 warmup iterations and 200 measurement iterations (median of CUDA events).
- LlamaRMSNorm performs: cast_fp32 → x.pow(2).mean() → rsqrt → multiply → cast_back, which is multiple CUDA kernels vs the fused F.layer_norm.
- Llama-3.1-8B uses SiLU activation (not GeLU). FFN Up uses gate_proj (one of two [4096,14336] projections in Llama's gated MLP).
- Positive error means LLMCompass overestimates; negative means it underestimates.
