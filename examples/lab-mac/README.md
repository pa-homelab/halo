# Mac Lab — Apple Silicon SFT/LoRA Training

> **⚠️ This is NOT White Circle Halo CUDA training.**
>
> This is a fork-local lab overlay (`pa-homelab/halo`) for learning and experimentation
> on Apple Silicon Macs using PyTorch MPS. It does not use the Halo Docker images,
> does not require CUDA or NVIDIA GPUs, and is not part of the upstream Halo project.
>
> For real distributed training on NVIDIA hardware, see the main [Halo documentation](../../human-docs/README.md).

## What This Is

A self-contained SFT + LoRA training script that runs on Apple Silicon (M1/M2/M3/M4/M5 Pro/Max/Ultra)
using PyTorch's MPS backend. Designed for:

- Learning how LoRA fine-tuning works
- Quick experiments on small models (Qwen2.5-0.5B/1.5B)
- Validating chat data formats before scaling to CUDA clusters

## Requirements

- macOS 13+ (Ventura or later) on Apple Silicon
- Python 3.10–3.12
- 16GB+ unified memory (32GB+ recommended for 1.5B models)

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv-lab-mac
source .venv-lab-mac/bin/activate

# 2. Install dependencies
pip install -r examples/lab-mac/requirements.txt

# 3. Run a smoke test (20 steps, ~3-5 min on M5 Pro)
python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml

# 4. (Optional) Override settings via CLI
python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml \
    --max_steps=50 \
    --learning_rate=5e-5
```

The default config uses conservative settings (`max_length=512`, `batch_size=1`) to avoid
OOM when macOS has other apps open. With more free memory, you can increase throughput:

```bash
# Faster settings if you have 32GB+ and few other apps running
python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml \
    --max_length=1024 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=4
```

## What's Included

| File | Purpose |
|------|---------|
| `requirements.txt` | Pinned dependencies (no CUDA, no bitsandbytes) |
| `train_sft_lora.py` | Single-file SFT + LoRA trainer for MPS |
| `qwen25-0.5b-lora.yaml` | Default config for Qwen2.5-0.5B-Instruct |
| `qwen25-1.5b-lora.yaml` | Larger model config (needs 32GB+ RAM) |

## Model Choices

| Model | Parameters | Memory | Notes |
|-------|------------|--------|-------|
| **Qwen2.5-0.5B-Instruct** | 0.5B | ~8GB | Default; fast iteration |
| Qwen2.5-1.5B-Instruct | 1.5B | ~12-14GB | Better quality; uses `max_length=512` by default |

The 0.5B model is the default for first prove-out. Edit the config or pass
`--model_name_or_path=Qwen/Qwen2.5-1.5B-Instruct` to use the larger model.

**Memory notes (validated on 48GB M5 Pro):**
- 1.5B uses `max_length=512` by default — `max_length=2048` OOMs under typical desktop load
- 0.5B with `max_length=1024 --per_device_train_batch_size=2` can OOM mid-run when other apps are open
- Close browsers/Slack/Docker before trying higher settings

## What This Does NOT Support

- **Multi-GPU** — MPS is single-device only
- **Flash Attention** — use SDPA (the default on MPS)
- **bitsandbytes / QLoRA** — requires CUDA; use standard LoRA instead
- **DeepEP / Expert Parallelism** — CUDA-only
- **torch.compile** — unstable on MPS; disabled by default
- **Large models (>3B)** — memory and speed make this impractical

## Output

Checkpoints are saved to `checkpoints/lab-mac-<model>-<timestamp>/` by default.
The final adapter can be merged into the base model using PEFT's standard merge:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "checkpoints/lab-mac-.../checkpoint-final")
merged = model.merge_and_unload()
merged.save_pretrained("my-merged-model")
```

## Troubleshooting

**"MPS not available"**
- Ensure you're on macOS 13+ with Apple Silicon
- Check: `python -c "import torch; print(torch.backends.mps.is_available())"`

**Out of memory (MPS backend out of memory)**

MPS shares unified memory with macOS and other apps. The error looks like:
```
RuntimeError: MPS backend out of memory (MPS allocated: X GiB, other allocations: Y GiB, max allowed: Z GiB)
```

Fixes:
1. **Close other apps** — browsers, Slack, Docker, etc. consume unified memory
2. **Use the conservative defaults** — the 0.5B config now ships with `max_length=512` and `batch_size=1`
3. **Reduce further if needed:**
   ```bash
   python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml \
       --max_length=256 \
       --per_device_train_batch_size=1
   ```
4. **Use the 0.5B model** — the 1.5B config needs 32GB+ with minimal other apps

The default config is tuned for a 48GB M5 Pro with typical desktop load. Machines with
16GB unified memory should stick to `max_length=256-512` and `batch_size=1`.

**Slow training**
- MPS is slower than CUDA; expect ~10–30 tok/s on M5 Pro with 0.5B
- Gradient checkpointing is enabled by default to save memory
- Shorter sequences (`max_length=512`) train faster than longer ones

## See Also

- [`LAB.md`](../../LAB.md) — overview of both CUDA and Mac lab tracks
- [PEFT Documentation](https://huggingface.co/docs/peft) — LoRA details
- [Qwen2.5 Models](https://huggingface.co/Qwen) — model cards and chat templates
