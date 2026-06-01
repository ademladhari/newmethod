# vast.ai — Jupyter experiments

| Notebook | Experiment |
|----------|------------|
| [`moe_unfrozen_sym_v1.ipynb`](moe_unfrozen_sym_v1.ipynb) | Unfrozen backbone + symmetric 4-expert MoE (`hidden_moe_unfrozen`) |

## Quick start on vast.ai

1. Rent a GPU (**≥24 GB VRAM** for unfrozen batch **128**, ≥80 GB disk for full COCO).
2. **CUDA-only templates are OK** — they often lack `torch`. The notebook **cell 2** installs PyTorch+CUDA.
3. Start **Jupyter** from the instance.
4. Open the notebook → run **cell 2 (Install PyTorch)** before the GPU check → set Kaggle credentials → continue.

### If you only see `ModuleNotFoundError: No module named 'torch'`

Run this in Jupyter **before** anything else:

```python
import sys, subprocess
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchvision",
    "--index-url", "https://download.pytorch.org/whl/cu124",
])
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

If that fails, change `cu124` → `cu121` or `cu118`.

Runs save under `/workspace/newmethod/hidden_moe_unfrozen/runs/`.

## If you already have data on the instance

In cell 1 set:

```python
SKIP_COCO_DOWNLOAD = True
# or
LOCAL_COCO_TRAIN = "/workspace/coco100k/train"
LOCAL_COCO_VAL = "/workspace/coco100k/val"
```

## OOM

Default `BATCH_SIZE = 128` in the notebook. Lower only if OOM (try 64, then 32). Unfrozen trains encoder + backbone + MoE.
