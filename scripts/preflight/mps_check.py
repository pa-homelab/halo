#!/usr/bin/env python3
"""
MPS Preflight Check — Verify Apple Silicon MPS availability.

This script is part of the Mac Lab (examples/lab-mac/), NOT the CUDA Halo stack.
It checks whether PyTorch MPS is available for Apple Silicon training.

Usage:
    python scripts/preflight/mps_check.py

Exit codes:
    0 — MPS is available and ready
    1 — MPS is not available (not Apple Silicon, old macOS, or missing PyTorch MPS)
"""

import sys


def main() -> int:
    print("=" * 50)
    print("Mac Lab — MPS Preflight Check")
    print("=" * 50)
    print()
    print("NOTE: This is for the Mac Lab (examples/lab-mac/),")
    print("      NOT the CUDA Halo training stack.")
    print()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed.")
        print("       pip install torch>=2.3.0")
        return 1

    print(f"PyTorch version: {torch.__version__}")

    if not hasattr(torch.backends, "mps"):
        print()
        print("ERROR: PyTorch MPS backend not found.")
        print("       Your PyTorch version may be too old.")
        print("       pip install torch>=2.3.0")
        return 1

    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()

    print(f"MPS built: {mps_built}")
    print(f"MPS available: {mps_available}")

    if not mps_built:
        print()
        print("ERROR: PyTorch was not built with MPS support.")
        print("       pip install --force-reinstall torch>=2.3.0")
        return 1

    if not mps_available:
        print()
        print("ERROR: MPS is not available on this system.")
        print()
        print("Possible causes:")
        print("  - Not running on Apple Silicon (M1/M2/M3/M4/M5)")
        print("  - macOS version too old (need 13.0+ / Ventura)")
        print("  - Running in a VM or container without GPU passthrough")
        return 1

    print()
    print("SUCCESS: MPS is available and ready for Mac Lab training.")
    print()
    print("To run a training job:")
    print("  python examples/lab-mac/train_sft_lora.py examples/lab-mac/qwen25-0.5b-lora.yaml")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
