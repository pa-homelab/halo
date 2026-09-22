# Lab Hardware Inventory

Fork-local documentation for Taissa Conde's homelab (inventory 2026-09-22).

## The Rule

**Real Halo training requires an NVIDIA GPU with CUDA support and the official `halo:blackwell` or
`halo:hopper` Docker image.** The hardware below does not meet this requirement.

## Current Inventory

### ts-macbook-pro

- **CPU/GPU:** Apple M5 Pro, 48 GB unified memory
- **Role:** Local inference only (LM Studio, Qwenforge, llama.cpp with Metal)
- **Halo status:** NOT a training host — no CUDA, no `torch.cuda.is_available()`. MPS/MLX backends
  are not supported by upstream Halo. Use for reviewing model outputs, not for running the trainers.

### omarchy-samsung

- **GPU:** NVIDIA GeForce 710M (Kepler architecture, GK208M)
- **Driver:** nouveau (open-source); no `nvidia-smi`, no CUDA toolkit installed
- **VRAM:** ~1–2 GB class
- **Host RAM:** ~7.7 GB
- **Disk:** ~423 GB free
- **Docker:** Ready
- **Halo status:** NOT usable for training. The 710M predates Tensor Cores (Maxwell+), has no fp16
  fast-path, cannot run FlashAttention, and lacks the VRAM for even the smallest LoRA config.
  Installing the proprietary NVIDIA driver would not change this — the card is simply too old.

## What Would Work

To run the example configs in `examples/lab/`, you need:

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | Ampere (RTX 3090, A6000) or Ada (RTX 4090) | H100, B200 |
| VRAM | 24 GB | 48+ GB |
| Driver | nvidia-smi working, CUDA 12+ | Latest stable |
| Container | `halo:blackwell` | `halo:blackwell` |

Single-GPU LoRA/QLoRA runs from the `halo:blackwell` image with FlashAttention 2 or SDPA. DeepEP
(Expert Parallelism) requires Hopper or Blackwell.

## Inference Handoff

Until a CUDA-capable host joins the tailnet:

1. **Train elsewhere:** Rent cloud GPUs (Lambda, RunPod, vast.ai) or use a colab-style service.
2. **Serve locally:** Export the trained checkpoint and run inference on the M5 Pro via LM Studio,
   Qwenforge, or llama.cpp. GGUF export is outside Halo's scope but documented by the serving tools.

## Preflight Check

Before attempting any training, run the CUDA preflight from the container:

```bash
python scripts/preflight/cuda_check.py
```

This refuses to proceed without `torch.cuda.is_available() == True` and prints the GPU it found (or
the reason it cannot continue).
