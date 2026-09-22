# Lab Example Configs

Fork-local training configs for learning and verification on small CUDA GPUs.

## Requirements

These configs **require a real NVIDIA GPU with CUDA support**:

- Ampere (RTX 3090, A6000) or Ada (RTX 4090) recommended
- Minimum 16 GB VRAM for QLoRA, 24 GB for LoRA
- Must pass `python scripts/preflight/cuda_check.py`

## What's Here

| Config | Base Model | Adapter | VRAM (approx) | Steps |
|--------|------------|---------|---------------|-------|
| `qwen3-4b-lora-lab.yaml` | Qwen3-4B | LoRA | ~12–16 GB | 50 |
| `qwen3-4b-qlora-lab.yaml` | Qwen3-4B | QLoRA (4-bit) | ~6–8 GB | 50 |

## What These Are For

These configs exist to:

1. Verify the training loop runs on your hardware
2. Learn the config format and workflow
3. Test changes to the codebase

They are **not** sized to produce a useful trained model — the step count is too low and the
data slice is tiny. For real training, use the upstream configs in `examples/sft/`.

## Usage

Inside the `halo:blackwell` container:

```bash
# Run preflight first
python scripts/preflight/cuda_check.py

# Then train
python scripts/training/sft.py examples/lab/qwen3-4b-qlora-lab.yaml
```

## What Does NOT Work

- **Apple Silicon (M1/M2/M3/M4/M5):** MPS is not a supported backend. Use the Mac for inference.
- **GeForce 710M / GT 710 / Kepler:** Too old, too little VRAM, no Tensor Cores.
- **CPU-only:** Not supported.

See `docs/lab/HARDWARE.md` for the full hardware story.
