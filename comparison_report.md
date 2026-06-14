# LLMCompass vs Real GPU Execution Latency Comparison

## Motivation

> In LLMCompass paper, they compare the LLMCompass results with real GPU execution latency. They do not mention which library they used, but I assume that they just ran multiple GEMM kernels to get latency of each linear layer and pytorch's functions for other functions such as layernorm, gelu, etc. I want to reproduce their real GPU execution latency with above simple implementation. I don't need prefill latency (model_init in LLMCompass) and only need decode latency (model_auto_regressive), and don't need multi-head attention (QxK, softmax, AxV). Except for above, compare LLMcompass result with real GPU execution latency. Maybe it is different because current simulation setting is H100 and this server have A6000 GPU, but I will use the script you made in another H100 server. This is just a session for implementation. You need to keep current LLMcompass code, and just add new files to evaluate real GPU execution and compare it with LLMcompass result. You can use change_core_atlas2_for_realGPUcomparison.py for comparison and you can modify the code if you want. Compare with sequence length 8192 and vary batch sizes. Write a final report in a markdown file format. You need to use empty GPU for precise latency estimation after nvidia-smi.

## Configuration

**Model:** Llama-3.1-8B
**Sequence Length (KV cache context):** 8192
**Number of Layers:** 32
**Mode:** Decode (auto-regression, seq_len=1 token)
**Excluded operations:** QxK, Softmax, AxV (multi-head attention)
**LLMCompass config:** H100 (GH100_80GB), heuristic-GPU, no_QK_AV_softmax target
**LLMCompass reported total source:** raw no_QK_AV_softmax total
**Real GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition
**Real GPU timing:** cuda_events_median

## 1. Per-Operator Latency Breakdown (per layer, microseconds)

### QKV Proj

| Batch Size | LLMCompass (us) | Real GPU (us) | Error (%) |
|:----------:|:---------------:|:-------------:|:---------:|
| 1 | 78.13 | 61.47 | +27.1% |
| 2 | 78.49 | 60.74 | +29.2% |
| 4 | 78.51 | 60.61 | +29.5% |
| 8 | 78.54 | 61.23 | +28.3% |
| 16 | 78.66 | 61.68 | +27.5% |
| 32 | 78.97 | 65.02 | +21.4% |
| 64 | 79.54 | 72.03 | +10.4% |
| 128 | 80.43 | 68.48 | +17.5% |

### Output Proj (W0)

| Batch Size | LLMCompass (us) | Real GPU (us) | Error (%) |
|:----------:|:---------------:|:-------------:|:---------:|
| 1 | 31.09 | 20.00 | +55.4% |
| 2 | 31.24 | 20.64 | +51.4% |
| 4 | 31.25 | 20.83 | +50.0% |
| 8 | 31.26 | 20.91 | +49.5% |
| 16 | 31.32 | 21.89 | +43.1% |
| 32 | 31.43 | 25.49 | +23.3% |
| 64 | 31.65 | 31.52 | +0.4% |
| 128 | 32.02 | 28.93 | +10.7% |

### FFN Up (W1)

| Batch Size | LLMCompass (us) | Real GPU (us) | Error (%) |
|:----------:|:---------------:|:-------------:|:---------:|
| 1 | 56.30 | 38.16 | +47.5% |
| 2 | 56.53 | 37.34 | +51.4% |
| 4 | 56.54 | 37.18 | +52.0% |
| 8 | 56.61 | 37.58 | +50.6% |
| 16 | 56.70 | 37.28 | +52.1% |
| 32 | 56.89 | 43.46 | +30.9% |
| 64 | 57.31 | 43.26 | +32.5% |
| 128 | 58.08 | 66.30 | -12.4% |

### FFN Down (W2)

| Batch Size | LLMCompass (us) | Real GPU (us) | Error (%) |
|:----------:|:---------------:|:-------------:|:---------:|
| 1 | 56.30 | 39.52 | +42.5% |
| 2 | 56.54 | 37.28 | +51.7% |
| 4 | 56.58 | 39.20 | +44.3% |
| 8 | 56.65 | 42.48 | +33.3% |
| 16 | 56.83 | 45.28 | +25.5% |
| 32 | 57.13 | 44.74 | +27.7% |
| 64 | 57.74 | 43.33 | +33.3% |
| 128 | 58.84 | 73.71 | -20.2% |

### LayerNorm

| Batch Size | LLMCompass (us) | Real GPU (us) | Error (%) |
|:----------:|:---------------:|:-------------:|:---------:|
| 1 | 45.14 | 17.12 | +163.7% |
| 2 | 45.15 | 17.25 | +161.8% |
| 4 | 45.16 | 17.22 | +162.3% |
| 8 | 45.20 | 17.15 | +163.5% |
| 16 | 45.39 | 17.07 | +165.9% |
| 32 | 45.47 | 17.15 | +165.1% |
| 64 | 45.62 | 17.15 | +166.0% |
| 128 | 45.94 | 17.22 | +166.8% |

### GeLU

| Batch Size | LLMCompass (us) | Real GPU (us) | Error (%) |
|:----------:|:---------------:|:-------------:|:---------:|
| 1 | 45.06 | 13.38 | +236.8% |
| 2 | 45.06 | 13.34 | +237.6% |
| 4 | 45.11 | 13.33 | +238.5% |
| 8 | 45.22 | 13.34 | +238.9% |
| 16 | 45.39 | 13.22 | +243.4% |
| 32 | 45.77 | 13.28 | +244.7% |
| 64 | 46.54 | 13.63 | +241.4% |
| 128 | 48.03 | 13.76 | +249.0% |

## 2. Aggregate Latency Per Layer (microseconds)

| Batch Size | Component | LLMCompass (us) | Real GPU (us) | Error (%) |
|:----------:|:---------:|:---------------:|:-------------:|:---------:|
| 1 | GEMM Total | 221.82 | 159.15 | +39.4% |
| 1 | Norm Total | 90.28 | 34.24 | +163.7% |
| 1 | GeLU | 45.06 | 13.38 | +236.8% |
| **1** | **Layer Total** | **357.15** | **206.77** | **+72.7%** |
| 2 | GEMM Total | 222.80 | 156.00 | +42.8% |
| 2 | Norm Total | 90.30 | 34.50 | +161.8% |
| 2 | GeLU | 45.06 | 13.34 | +237.6% |
| **2** | **Layer Total** | **358.15** | **203.84** | **+75.7%** |
| 4 | GEMM Total | 222.87 | 157.82 | +41.2% |
| 4 | Norm Total | 90.33 | 34.43 | +162.3% |
| 4 | GeLU | 45.11 | 13.33 | +238.5% |
| **4** | **Layer Total** | **358.30** | **205.58** | **+74.3%** |
| 8 | GEMM Total | 223.06 | 162.21 | +37.5% |
| 8 | Norm Total | 90.39 | 34.30 | +163.5% |
| 8 | GeLU | 45.22 | 13.34 | +238.9% |
| **8** | **Layer Total** | **358.68** | **209.86** | **+70.9%** |
| 16 | GEMM Total | 223.52 | 166.13 | +34.5% |
| 16 | Norm Total | 90.77 | 34.14 | +165.9% |
| 16 | GeLU | 45.39 | 13.22 | +243.4% |
| **16** | **Layer Total** | **359.68** | **213.49** | **+68.5%** |
| 32 | GEMM Total | 224.42 | 178.70 | +25.6% |
| 32 | Norm Total | 90.93 | 34.30 | +165.1% |
| 32 | GeLU | 45.77 | 13.28 | +244.7% |
| **32** | **Layer Total** | **361.12** | **226.29** | **+59.6%** |
| 64 | GEMM Total | 226.23 | 190.14 | +19.0% |
| 64 | Norm Total | 91.25 | 34.30 | +166.0% |
| 64 | GeLU | 46.54 | 13.63 | +241.4% |
| **64** | **Layer Total** | **364.02** | **238.08** | **+52.9%** |
| 128 | GEMM Total | 229.38 | 237.42 | -3.4% |
| 128 | Norm Total | 91.88 | 34.43 | +166.8% |
| 128 | GeLU | 48.03 | 13.76 | +249.0% |
| **128** | **Layer Total** | **369.28** | **285.62** | **+29.3%** |

## 3. Total Model Latency (all layers, milliseconds)

| Batch Size | LLMCompass (ms) | Real GPU (ms) | Error (%) |
|:----------:|:---------------:|:-------------:|:---------:|
| 1 | 11.4289 | 6.6166 | +72.7% |
| 2 | 11.4608 | 6.5229 | +75.7% |
| 4 | 11.4657 | 6.5787 | +74.3% |
| 8 | 11.4776 | 6.7154 | +70.9% |
| 16 | 11.5096 | 6.8316 | +68.5% |
| 32 | 11.5558 | 7.2412 | +59.6% |
| 64 | 11.6485 | 7.6186 | +52.9% |
| 128 | 11.8170 | 9.1397 | +29.3% |

## 4. Summary Statistics

- **Mean absolute error (per-layer total):** 63.0%
- **Min absolute error:** 29.3%
- **Max absolute error:** 75.7%

### Notes

- LLMCompass latency is from cycle-accurate simulation targeting H100 (GH100_80GB).
- Real GPU latency was not measured on H100 in this CSV. Results will differ due to GPU architecture and should be re-run on H100 for a fair comparison.
- The comparison total is `QKV + W0 + W1 + W2 + 2*LayerNorm + GeLU`; QxK, Softmax, and AxV are excluded from both sides.
- Each operation was measured with 50 warmup iterations and 200 measurement iterations (median).
- LLMCompass still logs QxK/AxV/softmax values for audit, but those values are excluded from the comparison total.
- Positive error means LLMCompass overestimates; negative means it underestimates.
