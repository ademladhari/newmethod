# MoE experiment plan (what to run, what to skip)

Baseline reference: **`moe_unfrozen_sym_t14_v1` @ epoch 20** — clean BER 0.32%, `train_val_load_l1` 0.08, 4 experts, top-k=1, batch 128, **no** training attacks.

---

## Do not train further

| Run | Reason |
|-----|--------|
| 8-exp sparse, top-2, dense k=4 | Worse BER / stability than main |
| `exp4continue` epochs 21–60 | Overfitting; use ep-20 checkpoint only |
| `moe_sym_bal08` partial zip | Incomplete |

Archive for thesis ablations only.

---

## Required experiments (thesis)

### R0 — Clean-trained MoE (DONE)

- **Name:** `moe_unfrozen_sym_t14_v1`
- **Flags:** default (no `--apply-training-noise`)
- **Epochs:** 20
- **Report:** clean channel + 8-attack eval vs HiDDeN

### R1 — Attack-trained MoE (NEW — primary)

- **Name:** `moe_unfrozen_sym_t14_attack_v1`
- **Flags:** same as R0 + **`--apply-training-noise`**
- **Epochs:** **80** (schedule: identity epochs 1–20, attack mix ramps 21–80)
- **Warm-start:** HiDDeN epoch-177 (same as R0)
- **Stop rule:** save best checkpoint by **clean** `validation.csv` BER + `train_val_load_l1` < 0.20; do **not** use last epoch by default
- **Eval:** same 8-attack harness on **best clean-val** checkpoint; compare to R0 and HiDDeN

### R2 — Attack eval comparison (analysis only)

| Model | Training | Eval |
|-------|----------|------|
| HiDDeN | clean | 8 attacks |
| R0 MoE | clean | 8 attacks |
| R1 MoE | attack curriculum | 8 attacks |

Thesis claim changes: R0 = architectural generalisation; R1 = trained with distortion curriculum.

---

## Optional (only if GPU time left)

### R3 — Attack train, short (sanity)

- **Name:** `moe_unfrozen_sym_t14_attack_ep40_v1`
- Same as R1 but **`--epochs 40`**
- Checks whether ramp 21–40 is enough (attacks only in training for epochs 21–40)

### R4 — Attack train from R0 checkpoint (continue)

- Start from **R0 epoch-20** `.pyt`, not HiDDeN-177
- `continue --folder .../sym_t14_unfrozen... --apply-training-noise true --epochs 80`
- **Risk:** temperature schedule / LR not tuned for continue — prefer **R1 fresh** from HiDDeN-177

### R5 — COCO-200k (data scale)

- Only after R1: same flags, `--data-dir coco200k`, 20 epochs + early stop
- Compare on **fixed COCO val** for fair BER

**Skip:** 8-exp, top-2, dense k=4, longer continue on R0 without attack flag change.

---

## Code change (done in repo)

- `HiDDenMoEConfiguration.apply_training_noise`
- CLI: `train_moe.py new ... --apply-training-noise`
- Continue override: `--apply-training-noise true|false`

Training still follows `MoENoiseLayer.set_training_schedule(epoch)`:

- Epochs **1–20:** identity-only attack weights (even with flag on)
- Epochs **21–80:** linear ramp to non-identity experts
- So **R1 must run to 80** (or change schedule in code) for attacks to appear in training.

Validation unchanged:

- `validation.csv` = always clean (identity)
- `validation_noisy.csv` = sampled attacks (ramps with epoch — compare across epochs with care)

---

## Kaggle command (R1)

```bash
%cd /kaggle/working/newmethod/hidden_moe_unfrozen

python -u train_moe.py new \
  --data-dir /kaggle/working/coco100k \
  --name moe_unfrozen_sym_t14_attack_v1 \
  --batch-size 128 \
  --epochs 80 \
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
  --init-hidden-checkpoint "/kaggle/input/.../my_hidden_experiment--epoch-177.pyt" \
  --apply-training-noise \
  --enable-fp16 \
  --router-grad-clip-norm 0.5 \
  --num-workers 4 \
  --pin-memory \
  --prefetch-factor 3 \
  --save-every 5 \
  --print-each 100
```

After training: run `evaluate_moe.py` on best epoch (not necessarily 80).
