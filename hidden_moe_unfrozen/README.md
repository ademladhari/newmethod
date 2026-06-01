# HiDDeN MoE — Unfrozen backbone branch

Copy of the dense MoE stack for experiments where the **full HiDDeN model learns**, not only the MoE head.

| | `hidden_frozen178_moe` | **`hidden_moe_unfrozen`** (this folder) |
|---|------------------------|----------------------------------------|
| Warm-start epoch-177 | yes | yes |
| Encoder + `decoder.feature_layers` | **frozen** (default) | **trainable** (default) |
| MoE router + experts | trainable | trainable |
| Use case | Routing / MoE-only adaptation | Richer features for router; may fix collapse |

Runs are written under `./runs/` inside this folder (separate from `hidden_frozen178_moe/runs/`).

## Default CLI (symmetric router, 4 experts)

- `--num-experts 4 --top-k 1`
- `--router-jitter-noise 0.0 --router-input-dropout 0.0 --expert-dropout 0.0`
- `--router-temperature-start 1.4 --router-temperature-end 1.0`
- **No** `--freeze-hidden-backbone` unless you want the old frozen behavior

## Kaggle — clone + train

```python
!rm -rf /kaggle/working/newmethod
!git clone https://github.com/ademladhari/newmethod.git /kaggle/working/newmethod
%cd /kaggle/working/newmethod/hidden_moe_unfrozen
!python train_moe.py new --help | grep -E "freeze|init-hidden"
```

```python
%cd /kaggle/working/newmethod/hidden_moe_unfrozen

!python -u train_moe.py new \
  --data-dir /kaggle/working/coco100k \
  --name moe_unfrozen_sym_t14_v1 \
  --batch-size 128 \
  --epochs 20 \
  --num-experts 4 \
  --top-k 1 \
  --balance-loss-weight 0.04 \
  --balance-loss-start-weight 0.005 \
  --balance-loss-warmup-epochs 10 \
  --router-jitter-noise 0.0 \
  --router-input-dropout 0.0 \
  --expert-dropout 0.0 \
  --router-z-loss-weight 0.001 \
  --router-temperature-start 1.4 \
  --router-temperature-end 1.0 \
  --adversarial-loss 0.0001 \
  --init-hidden-checkpoint "/kaggle/input/datasets/wings2ofice2/hiddencheckpointc/my_hidden_experiment--epoch-177.pyt" \
  --enable-fp16 \
  --router-grad-clip-norm 0.5 \
  --num-workers 6 \
  --pin-memory \
  --prefetch-factor 3 \
  --save-every 5 \
  --print-each 100
```

**Do not pass** `--freeze-hidden-backbone` for the unfrozen experiment.

**Default batch size: 128** (matches best coco100k run). Use 64 or 32 only if OOM — unfrozen uses more VRAM than frozen MoE-only.

## Epoch 10 gates (same as symmetric frozen runs)

- Val `expert_max_use` < 0.50  
- `train_val_load_l1` < 0.20  
- `validation_noisy` BER not spiking vs epoch 10  

## Continue a run

```python
import glob
%cd /kaggle/working/newmethod/hidden_moe_unfrozen
RUN = sorted(glob.glob("runs/moe_unfrozen_sym_t14_v1*"))[-1]
!python -u train_moe.py continue --folder "{RUN}" --data-dir /kaggle/working/coco100k --epochs 20 --num-workers 6 --pin-memory --prefetch-factor 3 --save-every 5 --print-each 100
```

## What warm-start loads

From the baseline `.pyt`: **encoder** weights and **decoder CNN backbone** (`decoder.layers.*` → `feature_layers`).  
MoE **router + experts** stay randomly initialized (with small expert offsets).

When unfrozen, gradients update encoder, feature layers, router, experts, and discriminator (unless `adversarial_loss=0` or `--freeze-discriminator`).
