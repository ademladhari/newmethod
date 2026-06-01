"""Audit distinct runs in archive/zips vs installed experiments."""
from __future__ import annotations

import io
import pickle
import re
import zipfile
from collections import defaultdict
from pathlib import Path

ZDIR = Path(__file__).resolve().parents[1] / "results" / "archive" / "zips"
EXP = Path(__file__).resolve().parents[1] / "results" / "experiments"

RUN_RE = re.compile(
    r"(?:^|/)runs/(moe[^\s/]+ \d{4}\.\d{2}\.\d{2}--\d{2}-\d{2}-\d{2})"
)
RUN_FLAT_RE = re.compile(r"^(moe[^\s/]+ \d{4}\.\d{2}\.\d{2}--\d{2}-\d{2}-\d{2})/")


def cfg_from_log(text: str) -> dict:
    cfg = {}
    for key in (
        "num_experts",
        "top_k",
        "balance_loss_weight",
        "balance_loss_start_weight",
        "balance_loss_warmup_epochs",
        "router_jitter_noise",
        "load_penalty_weight",
        "freeze_hidden_backbone",
        "batch_size",
        "router_temperature_start",
        "router_temperature_end",
        "adversarial_loss",
    ):
        m = re.search(r"'" + key + r"': ([^\n,]+)", text)
        if m:
            cfg[key] = m.group(1).strip().strip("'")
    return cfg


def scan_zip(zpath: Path) -> list[dict]:
    rows = []
    with zipfile.ZipFile(zpath) as zf:
        runs: dict[str, dict] = {}
        for name in zf.namelist():
            norm = name.replace("\\", "/")
            m = RUN_RE.search("/" + norm) or RUN_FLAT_RE.match(norm)
            if not m:
                if norm.startswith("moe_run/") and norm.endswith("train.csv"):
                    runs["moe_run flat"] = {
                        "run_folder": "moe_run flat",
                        "has_train": True,
                    }
                elif norm == "train.csv":
                    runs[zpath.stem] = {"run_folder": zpath.stem, "has_train": True}
                continue
            rf = m.group(1)
            r = runs.setdefault(rf, {"run_folder": rf, "has_train": False})
            if norm.endswith("train.csv"):
                r["has_train"] = True
            if norm.endswith(".log") and "cfg" not in r:
                r["cfg"] = cfg_from_log(zf.read(name).decode("utf-8", errors="replace"))
            if norm.endswith("options-and-config.pickle") and "cfg" not in r:
                try:
                    data = pickle.load(io.BytesIO(zf.read(name)))
                    if hasattr(data, "__dict__"):
                        r["cfg"] = {
                            k: getattr(data, k)
                            for k in vars(data)
                            if not k.startswith("_")
                        }
                except Exception:
                    pass
        for r in runs.values():
            r["zip"] = zpath.name
            rows.append(r)
    return rows


def main():
    all_rows = []
    for z in sorted(ZDIR.glob("*.zip")):
        try:
            all_rows.extend(scan_zip(z))
        except zipfile.BadZipFile:
            print("BAD", z.name)

    by_folder: dict[str, list] = defaultdict(list)
    for r in all_rows:
        by_folder[r["run_folder"]].append(r)

    print(f"Zips: {len(list(ZDIR.glob('*.zip')))}")
    print(f"Unique run folder names (across zips): {len(by_folder)}")
    with_train = [k for k, v in by_folder.items() if any(x.get("has_train") for x in v)]
    print(f"With train.csv: {len(with_train)}")
    print()

    by_base: dict[str, list[str]] = defaultdict(list)
    for rf in by_folder:
        base = rf.rsplit(" ", 1)[0] if " " in rf and re.search(r"\d{4}\.", rf) else rf
        by_base[base].append(rf)

    print("=== Same experiment_name, multiple timestamps (possible distinct retries) ===")
    for base, folders in sorted(by_base.items()):
        if len(folders) < 2:
            continue
        print(f"\n{base} ({len(folders)} runs):")
        for rf in sorted(folders):
            rec = by_folder[rf][0]
            ht = "train" if rec.get("has_train") else "partial"
            cfg = rec.get("cfg") or {}
            brief = ", ".join(
                f"{k}={cfg.get(k)}"
                for k in (
                    "batch_size",
                    "num_experts",
                    "top_k",
                    "balance_loss_weight",
                    "router_jitter_noise",
                    "load_penalty_weight",
                )
                if cfg.get(k) is not None
            )
            print(f"  {rf} [{ht}]")
            if brief:
                print(f"    {brief}")

    installed = list(EXP.rglob("train.csv"))
    print(f"\n=== Installed under experiments/ now: {len(installed)} runs ===")
    for t in sorted(installed)[:30]:
        print(" ", t.parent.relative_to(EXP))


if __name__ == "__main__":
    main()
