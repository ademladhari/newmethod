"""Full config audit of every MoE run in archive/zips."""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZDIR = ROOT / "results" / "archive" / "zips"
EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "FULL_EXPERIMENT_REGISTRY.json"
OUT_MD = ROOT / "results" / "FULL_EXPERIMENT_REGISTRY.md"

RUN_RE = re.compile(
    r"(?:^|/)runs/(moe[^\s/]+ \d{4}\.\d{2}\.\d{2}--\d{2}-\d{2}-\d{2})"
)
RUN_FLAT_RE = re.compile(r"^(moe[^\s/]+ \d{4}\.\d{2}\.\d{2}--\d{2}-\d{2}-\d{2})/")

CONFIG_KEYS = (
    "experiment_name",
    "num_experts",
    "top_k",
    "batch_size",
    "epochs",
    "balance_loss_weight",
    "balance_loss_start_weight",
    "balance_loss_warmup_epochs",
    "router_jitter_noise",
    "router_input_dropout",
    "expert_dropout",
    "router_z_loss_weight",
    "router_temperature_start",
    "router_temperature_end",
    "load_penalty_weight",
    "load_penalty_type",
    "adversarial_loss",
    "freeze_hidden_backbone",
    "enable_fp16",
    "router_grad_clip_norm",
    "train_folder",
    "val_folder",
)


def _norm(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    return str(v).strip().strip("'")


def _cfg_from_log(text: str) -> dict:
    cfg = {}
    for key in CONFIG_KEYS:
        m = re.search(r"'" + key + r"': ([^\n,]+)", text)
        if m:
            cfg[key] = _norm(m.group(1))
    if "train_folder" in cfg and "coco100k" in cfg["train_folder"]:
        cfg["dataset"] = "coco100k"
    return cfg


def _cfg_from_pickle(blob: bytes) -> dict:
    try:
        data = pickle.load(io.BytesIO(blob))
    except Exception:
        return {}
    out = {}
    src = vars(data) if hasattr(data, "__dict__") else {}
    for key in CONFIG_KEYS:
        if key in src and src[key] is not None:
            out[key] = _norm(src[key])
    return out


def _fingerprint(cfg: dict) -> str:
    """Hash hyperparameters only (not experiment_name or paths)."""
    parts = []
    for k in sorted(CONFIG_KEYS):
        if k in ("experiment_name", "train_folder", "val_folder", "epochs"):
            continue
        parts.append(f"{k}={cfg.get(k, '')}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _count_epochs_in_zip(zf: zipfile.ZipFile, prefix: str) -> int | None:
    for n in zf.namelist():
        if n.startswith(prefix) and n.endswith("train.csv"):
            try:
                lines = zf.read(n).decode("utf-8", errors="replace").strip().split("\n")
                return max(0, len(lines) - 1)
            except Exception:
                return None
    return None


def _scan_zip(zpath: Path) -> list[dict]:
    rows = []
    with zipfile.ZipFile(zpath) as zf:
        runs: dict[str, dict] = {}

        for name in zf.namelist():
            norm = name.replace("\\", "/")
            m = RUN_RE.search("/" + norm) or RUN_FLAT_RE.match(norm)
            if m:
                rf = m.group(1)
                r = runs.setdefault(rf, {"run_folder": rf, "zip": zpath.name})
                prefix = norm[: norm.index(rf) + len(rf) + 1]
                r["prefix"] = prefix
                if norm.endswith("train.csv"):
                    r["has_train"] = True
                if norm.endswith("validation.csv"):
                    r["has_val"] = True
                if norm.endswith("validation_noisy.csv"):
                    r["has_val_noisy"] = True
                if norm.endswith(".log"):
                    r["_log"] = zf.read(name)
                if norm.endswith("options-and-config.pickle"):
                    r["_pkl"] = zf.read(name)
                continue

            if norm.startswith("moe_run/") and norm.endswith("train.csv"):
                r = runs.setdefault("moe_run flat", {"run_folder": "moe_run flat", "zip": zpath.name, "prefix": "moe_run/"})
                r["has_train"] = True
            elif norm == "train.csv":
                r = runs.setdefault(zpath.stem, {"run_folder": zpath.stem, "zip": zpath.name, "prefix": ""})
                r["has_train"] = True
                if norm.endswith(".log") or "moe_hidden.log" in zf.namelist():
                    pass

        for n in zf.namelist():
            if n.endswith(".log") and "moe_run/" in n.replace("\\", "/"):
                r = runs.setdefault("moe_run flat", {"run_folder": "moe_run flat", "zip": zpath.name, "prefix": "moe_run/"})
                r["_log"] = zf.read(n)
                r["log_name"] = Path(n).stem

        for rf, r in runs.items():
            cfg = {}
            if "_log" in r:
                cfg.update(_cfg_from_log(r["_log"].decode("utf-8", errors="replace")))
            if "_pkl" in r:
                cfg.update({k: v for k, v in _cfg_from_pickle(r["_pkl"]).items() if k not in cfg or not cfg[k]})
            if not cfg.get("experiment_name") and " " in rf and re.search(r"\d{4}\.", rf):
                cfg["experiment_name"] = rf.rsplit(" ", 1)[0]
            elif rf == "moe_run flat" and r.get("log_name"):
                cfg["experiment_name"] = r["log_name"]
            elif "flat" not in rf and not cfg.get("experiment_name"):
                cfg["experiment_name"] = rf.split()[0] if rf else "unknown"

            prefix = r.get("prefix", "")
            epochs = _count_epochs_in_zip(zf, prefix) if r.get("has_train") else None

            rows.append({
                "run_folder": rf,
                "zip_file": zpath.name,
                "has_train_csv": bool(r.get("has_train")),
                "has_validation": bool(r.get("has_val")),
                "has_validation_noisy": bool(r.get("has_val_noisy")),
                "epochs_completed": epochs,
                "config": cfg,
                "config_fingerprint": _fingerprint(cfg),
                "prefix_in_zip": prefix,
            })
    return rows


def _installed_runs() -> list[dict]:
    out = []
    for train in EXP.rglob("train.csv"):
        d = train.parent
        cfg = {}
        logs = list(d.glob("*.log"))
        if logs:
            cfg = _cfg_from_log(logs[0].read_text(encoding="utf-8", errors="replace"))
        pkl = d / "options-and-config.pickle"
        if pkl.is_file():
            cfg.update({k: v for k, v in _cfg_from_pickle(pkl.read_bytes()).items() if k not in cfg})
        out.append({
            "installed_path": str(d.relative_to(ROOT / "results")).replace("\\", "/"),
            "folder_name": d.name,
            "epochs_completed": max(0, sum(1 for _ in open(train, encoding="utf-8")) - 1),
            "config": cfg,
            "config_fingerprint": _fingerprint(cfg),
        })
    return out


def main():
    all_rows: list[dict] = []
    for z in sorted(ZDIR.glob("*.zip")):
        try:
            all_rows.extend(_scan_zip(z))
        except zipfile.BadZipFile as e:
            all_rows.append({"zip_file": z.name, "error": str(e)})

    # Dedupe by run_folder (merge zip sources)
    by_folder: dict[str, dict] = {}
    for row in all_rows:
        if "error" in row:
            continue
        rf = row["run_folder"]
        if rf not in by_folder:
            by_folder[rf] = {**row, "zip_files": [row["zip_file"]]}
        else:
            by_folder[rf]["zip_files"].append(row["zip_file"])
            if row.get("has_train_csv"):
                by_folder[rf]["has_train_csv"] = True
            if row.get("epochs_completed"):
                by_folder[rf]["epochs_completed"] = max(
                    by_folder[rf].get("epochs_completed") or 0,
                    row.get("epochs_completed") or 0,
                )

    runs = list(by_folder.values())
    installed = _installed_runs()
    installed_fps = {r["config_fingerprint"] for r in installed}
    installed_paths = {r["installed_path"] for r in installed}

    # Group by experiment_name
    by_name: dict[str, list] = defaultdict(list)
    for r in runs:
        name = r["config"].get("experiment_name", r["run_folder"].split()[0])
        by_name[name].append(r)

    # Find same name, different fingerprint
    same_name_diff_cfg = []
    for name, group in sorted(by_name.items()):
        fps = {g["config_fingerprint"] for g in group if g.get("config")}
        if len(fps) > 1:
            same_name_diff_cfg.append({
                "experiment_name": name,
                "distinct_configs": len(fps),
                "runs": group,
            })

    # Find fingerprints in zips not installed
    zip_fps = {r["config_fingerprint"]: r for r in runs if r.get("has_train_csv") and r.get("config_fingerprint")}
    missing_install = []
    for fp, r in zip_fps.items():
        if fp not in installed_fps:
            missing_install.append(r)

    registry = {
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "zip_files": len(list(ZDIR.glob("*.zip"))),
            "unique_run_folders_in_zips": len(runs),
            "runs_with_train_csv": sum(1 for r in runs if r.get("has_train_csv")),
            "distinct_config_fingerprints_in_zips": len({r["config_fingerprint"] for r in runs if r.get("config_fingerprint")}),
            "installed_runs": len(installed),
            "same_experiment_name_different_config": len(same_name_diff_cfg),
            "full_runs_in_zips_not_installed_by_fingerprint": len(missing_install),
        },
        "same_name_different_config": same_name_diff_cfg,
        "missing_from_install": missing_install,
        "all_runs_in_zips": sorted(runs, key=lambda x: (x["config"].get("experiment_name", ""), x["run_folder"])),
        "installed_runs": installed,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    _write_md(registry)
    print(json.dumps(registry["summary"], indent=2))
    print("Wrote", OUT)
    print("Wrote", OUT_MD)
    if same_name_diff_cfg:
        print("\n=== SAME NAME, DIFFERENT CONFIG ===")
        for item in same_name_diff_cfg:
            print(item["experiment_name"], "->", item["distinct_configs"], "configs")
    if missing_install:
        print("\n=== NOT INSTALLED (distinct fingerprint) ===")
        for r in missing_install[:20]:
            print(r["run_folder"], r["config_fingerprint"], r.get("epochs_completed"))


def _write_md(reg: dict) -> str:
    lines = [
        "# Full MoE experiment registry",
        "",
        f"Generated: {reg['generated']}",
        "",
        "## Summary",
        "",
    ]
    for k, v in reg["summary"].items():
        lines.append(f"- **{k}**: {v}")
    lines.extend(["", "## Installed runs", "", "| Path | experiment_name | experts | top_k | batch | epochs | fingerprint |", "|------|-----------------|---------|-------|-------|--------|-------------|"])
    for r in reg["installed_runs"]:
        c = r["config"]
        lines.append(
            f"| `{r['installed_path']}` | {c.get('experiment_name','?')} | {c.get('num_experts','?')} | {c.get('top_k','?')} | {c.get('batch_size','?')} | {r['epochs_completed']} | `{r['config_fingerprint']}` |"
        )

    lines.extend(["", "## Every run found in archive/zips", "", "| run_folder | train? | epochs | fingerprint | balance | jitter | load_pen | freeze | zip |", "|------------|--------|--------|-------------|---------|--------|----------|--------|-----|"])
    for r in reg["all_runs_in_zips"]:
        c = r.get("config") or {}
        z = ", ".join(r.get("zip_files", [r.get("zip_file", "")])[:2])
        if len(r.get("zip_files", [])) > 2:
            z += f" (+{len(r['zip_files'])-2})"
        lines.append(
            f"| `{r['run_folder']}` | {'yes' if r.get('has_train_csv') else 'partial'} | {r.get('epochs_completed') or '-'} | `{r.get('config_fingerprint','')}` | {c.get('balance_loss_weight','-')} | {c.get('router_jitter_noise','-')} | {c.get('load_penalty_weight','-')} | {c.get('freeze_hidden_backbone','-')} | {z[:60]} |"
        )

    if reg["same_name_different_config"]:
        lines.extend(["", "## Same experiment_name, different config (important)", ""])
        for block in reg["same_name_different_config"]:
            lines.append(f"### `{block['experiment_name']}` — {block['distinct_configs']} distinct configs")
            lines.append("")
            for r in block["runs"]:
                c = r["config"]
                lines.append(f"- **{r['run_folder']}** — fp `{r['config_fingerprint']}` — ep {r.get('epochs_completed','?')}")
                lines.append(
                    f"  - batch={c.get('batch_size')} experts={c.get('num_experts')} top_k={c.get('top_k')} "
                    f"bal={c.get('balance_loss_weight')} jitter={c.get('router_jitter_noise')} "
                    f"load_pen={c.get('load_penalty_weight')} freeze={c.get('freeze_hidden_backbone')}"
                )
            lines.append("")

    if reg["missing_from_install"]:
        lines.extend(["", "## Full runs in zips but not installed (by config fingerprint)", ""])
        for r in reg["missing_from_install"]:
            c = r["config"]
            lines.append(f"- `{r['run_folder']}` fp=`{r['config_fingerprint']}` — {c.get('experiment_name')} ep={r.get('epochs_completed')}")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
