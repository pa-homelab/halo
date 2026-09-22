# Lab Overlay

Fork-local additions for Taissa Conde's homelab. Upstream Halo is unchanged; these paths are clearly
marked as lab-only and do not affect the official Docker/Makefile/docs.

## What's Here

| Path | Purpose |
|------|---------|
| `docs/lab/HARDWARE.md` | Current hardware inventory and the CUDA requirement |
| `scripts/preflight/cuda_check.py` | Fail-fast check for `torch.cuda.is_available()` |
| `examples/lab/` | Tiny learning configs (require a real CUDA GPU) |
| `.env.lab.example` | Cache-path placeholders for large-disk mounts |

## The Gap

This tailnet has **no CUDA-capable NVIDIA host** suitable for Halo training:

- **ts-macbook-pro** (Apple M5 Pro, 48 GB unified) — inference only via LM Studio/Qwenforge; not a
  Halo training host.
- **omarchy-samsung** (GeForce 710M, Kepler, nouveau) — too old and too small; not usable.

The `examples/lab/` configs exist for the day a real GPU joins the setup. They are sized for a
small Ampere/Ada card (24 GB VRAM) and will fail immediately on the current hardware.

## Running the Preflight

Inside the `halo:blackwell` container:

```bash
python scripts/preflight/cuda_check.py
```

Expected output on current hardware: an error naming the missing CUDA requirement.

## Not Claimed

- Apple Silicon / MPS is **not** a supported training backend.
- The GeForce 710M is **not** a usable GPU for any Halo workload.
- These lab configs do **not** run without a real CUDA device.

See `docs/lab/HARDWARE.md` for the full inventory and what would actually work.
