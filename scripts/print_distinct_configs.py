import json
from pathlib import Path

reg = json.loads(
    Path("results/experiments/FULL_EXPERIMENT_REGISTRY.json").read_text(encoding="utf-8")
)
fps = {}
for r in reg["all_runs_in_zips"]:
    if not r.get("has_train_csv"):
        continue
    fp = r["config_fingerprint"]
    if fp == "16da67a744be":  # empty / unreadable
        continue
    if fp not in fps or (r.get("epochs_completed") or 0) > (fps[fp].get("epochs_completed") or 0):
        fps[fp] = r

print("Distinct FULL configs (non-empty):", len(fps))
for fp, r in sorted(fps.items(), key=lambda x: (x[1]["config"].get("experiment_name", ""), x[0])):
    c = r["config"]
    print(f"\n{fp} | {c.get('experiment_name')} | ep={r.get('epochs_completed')} | {r['run_folder']}")
    keys = [
        "num_experts", "top_k", "batch_size",
        "balance_loss_weight", "balance_loss_start_weight", "balance_loss_warmup_epochs",
        "router_jitter_noise", "load_penalty_weight", "load_penalty_type",
        "router_temperature_start", "router_temperature_end",
        "adversarial_loss", "freeze_hidden_backbone", "router_z_loss_weight",
    ]
    for k in keys:
        if c.get(k) not in (None, ""):
            print(f"  {k}: {c[k]}")
