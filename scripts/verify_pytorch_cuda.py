"""Quick check: PyTorch sees CUDA and RTX 50xx (sm_120) when using cu128 nightly."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except OSError as e:
        print("FAIL: torch is broken (partial install?). Re-run scripts/setup_pytorch_cu128.ps1")
        print(e)
        return 1

    print("torch:", torch.__version__)
    print("cuda built:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if "+cpu" in torch.__version__.lower():
        print("\nFAIL: CPU-only wheel. Install cu128 nightly (see scripts/setup_pytorch_cu128.ps1).")
        return 1

    if not torch.cuda.is_available():
        print("\nFAIL: CUDA not available.")
        print("  - RTX 5060 needs: pip install --pre torch ... --index-url https://download.pytorch.org/whl/nightly/cu128")
        print("  - Do NOT use plain: pip install torch")
        return 1

    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    arch = getattr(torch.cuda, "get_arch_list", lambda: [])()
    print("GPU:", name)
    print("compute capability:", cap)
    if arch:
        print("arch list:", arch)

    x = torch.randn(4, 4, device="cuda")
    y = x @ x
    print("matmul on cuda: OK", y.shape)

    if cap[0] >= 12 and "sm_120" not in str(arch) and "+cu128" not in torch.__version__:
        print("\nWARN: Blackwell GPU may need cu128 nightly (sm_120).")

    print("\nOK — use: --device cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
