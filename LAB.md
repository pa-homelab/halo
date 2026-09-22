# Lab Overlay

Fork-local additions for Taissa Conde's homelab. Upstream Halo is unchanged; these paths are clearly
marked as lab-only and do not affect the official Docker/Makefile/docs.

## What's Here

| Path | Purpose |
|------|---------|
| `docs/lab/HARDWARE.md` | Current hardware inventory and the CUDA requirement |
| `scripts/preflight/cuda_check.py` | Fail-fast check for `torch.cuda.is_available()` |
| `scripts/preflight/mps_check.py` | Fail-fast check for MPS availability (Mac Lab) |
| `examples/lab/` | Tiny learning configs (require a real CUDA GPU) |
| `examples/lab-mac/` | Apple Silicon SFT/LoRA configs (MPS, no CUDA) |
| `.env.lab.example` | Cache-path placeholders for large-disk mounts |

## The Gap

This tailnet has **no CUDA-capable NVIDIA host** suitable for Halo training:

- **ts-macbook-pro** (Apple M5 Pro, 48 GB unified) — inference only via LM Studio/Qwenforge; not a
  Halo training host. **Now has Mac Lab for learning/experimentation.**
- **omarchy-samsung** (GeForce 710M, Kepler, nouveau) — too old and too small; not usable.

The `examples/lab/` configs exist for the day a real GPU joins the setup. They are sized for a
small Ampere/Ada card (24 GB VRAM) and will fail immediately on the current hardware.

## Mac Lab (Apple Silicon)

> **⚠️ This is NOT production Halo training.**
>
> Mac Lab is a fork-local overlay for learning and quick experiments on Apple Silicon.
> It does not use the Halo Docker images, does not require CUDA, and is not part of
> the upstream Halo project.

A self-contained SFT + LoRA training path for Apple Silicon Macs using PyTorch MPS.
Designed for learning how LoRA fine-tuning works, validating data formats, and
quick iteration before a real CUDA GPU joins the setup.

**Requirements:**

- macOS 13+ on Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10–3.12
- 16GB+ unified memory

**Quick start:**

```bash
# Create a virtual environment (outside Docker)
python3 -m venv .venv-lab-mac
source .venv-lab-mac/bin/activate

# Install dependencies
pip install -r examples/lab-mac/requirements.txt

# Run a smoke test (~5 min on M5 Pro)
python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml
```

See [examples/lab-mac/README.md](examples/lab-mac/README.md) for details.

### What Mac Lab supports

- Qwen2.5-0.5B and 1.5B Instruct models
- LoRA via PEFT (r=16/32, attention modules)
- Chat datasets with the model's native chat template
- SDPA attention (not Flash Attention)
- Single-device MPS training

### What Mac Lab does NOT support

- Multi-GPU or distributed training
- Flash Attention (CUDA-only)
- bitsandbytes / QLoRA (CUDA-only)
- DeepEP / Expert Parallelism
- Large models (>3B parameters)
- torch.compile (unstable on MPS)

## Running the Preflight

Inside the `halo:blackwell` container (CUDA Lab):

```bash
python scripts/preflight/cuda_check.py
```

On macOS (Mac Lab):

```bash
python scripts/preflight/mps_check.py
```

## Choosing a Track

| If you have... | Use... |
|----------------|--------|
| NVIDIA GPU (any) | CUDA Lab via Docker (`examples/lab/`) |
| Apple Silicon Mac | Mac Lab for learning (`examples/lab-mac/`) |
| Neither | Cloud GPU (Lambda, RunPod, etc.) + CUDA Lab |

Mac Lab is intentionally limited. For production fine-tuning, real throughput,
and models larger than 3B, use CUDA Lab on NVIDIA hardware.

## Not Claimed

- Apple Silicon / MPS is **not** a supported training backend for upstream Halo.
- The GeForce 710M is **not** a usable GPU for any Halo workload.
- The `examples/lab/` configs do **not** run without a real CUDA device.
- Mac Lab is **not** production training — it's for learning only.

See `docs/lab/HARDWARE.md` for the full inventory and what would actually work.
