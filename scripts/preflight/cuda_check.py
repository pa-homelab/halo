#!/usr/bin/env python
"""CUDA availability preflight check.

Refuses to proceed when CUDA is not available, printing a clear error message. Use this before
attempting any Halo training run to fail fast rather than deep in the trainer.

Exit codes:
    0 — CUDA is available
    1 — CUDA is not available (training cannot proceed)
    2 — torch not importable (likely outside the Halo Docker image)

Usage:
    python scripts/preflight/cuda_check.py
    python scripts/preflight/cuda_check.py --quiet   # exit code only, no output on success
"""

import argparse
import shutil
import subprocess
import sys


def check_nvidia_smi() -> tuple[bool, str | None]:
    """Check if nvidia-smi is available and returns GPU info."""
    if not shutil.which("nvidia-smi"):
        return False, None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return False, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress output on success")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print(
            "ERROR: torch is not importable.\n\n"
            "Halo requires the official Docker image (halo:blackwell or halo:hopper) with PyTorch,\n"
            "Flash Attention, and DeepEP pre-built. Run this check inside the container:\n\n"
            "    docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \\\n"
            "        halo:blackwell python scripts/preflight/cuda_check.py\n",
            file=sys.stderr,
        )
        return 2

    if not torch.cuda.is_available():
        has_smi, smi_info = check_nvidia_smi()
        print(
            "ERROR: torch.cuda.is_available() returned False — CUDA training cannot proceed.\n",
            file=sys.stderr,
        )
        if has_smi:
            print(
                f"nvidia-smi reports: {smi_info}\n\n"
                "The GPU is visible to the host but not to PyTorch. Common causes:\n"
                "  - Container started without --gpus all\n"
                "  - NVIDIA Container Toolkit not installed or misconfigured\n"
                "  - Driver/CUDA version mismatch with the image\n",
                file=sys.stderr,
            )
        else:
            print(
                "nvidia-smi is not available or found no GPUs.\n\n"
                "Halo training requires an NVIDIA GPU with CUDA support. This host does not have one.\n"
                "  - Apple Silicon (MPS) is not a supported backend.\n"
                "  - Kepler-era GPUs (GTX 600/700, GT 710) lack the features Halo needs.\n"
                "  - CPU-only training is not supported.\n\n"
                "See docs/lab/HARDWARE.md for the hardware requirements.\n",
                file=sys.stderr,
            )
        return 1

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    if not args.quiet:
        print(f"CUDA is available: {device_count} device(s)")
        print(f"  Device 0: {device_name}")
        print(f"  Compute capability: {capability[0]}.{capability[1]}")
        print(f"  Total VRAM: {vram_gb:.1f} GB")

        if capability[0] < 7:
            print(
                "\nWARNING: Compute capability < 7.0 (pre-Volta). FlashAttention and mixed-precision\n"
                "training require Volta (V100) or newer. LoRA/QLoRA with SDPA may still work on\n"
                "Pascal (GTX 10xx, P100) but is not tested or recommended.\n"
            )
        elif capability[0] < 8:
            print(
                "\nNOTE: Volta/Turing GPU detected. FlashAttention 2 with SDPA fallback is supported.\n"
                "For best performance, use Ampere (A100, RTX 30xx) or newer.\n"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
