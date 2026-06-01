"""
Extract and group MoE runs from results/moe results/*.zip (and any loose folders there).

Run from repo root:
  python scripts/organize_moe_results.py

Output layout:
  results/experiments/{frozen_moe|unfrozen_moe|legacy_early}/coco100k/{4exp_k1|8exp_k2}/batch{N}/<descriptive>/
  results/archive/{duplicates|partial|zips}/
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MOE_INBOX = RESULTS / "moe results"
STAGING = RESULTS / "_staging"
EXPERIMENTS = RESULTS / "experiments"

RUN_DIR_RE = re.compile(
    r"(?:^|/)runs/(moe[^\s/]+ \d{4}\.\d{2}\.\d{2}--\d{2}-\d{2}-\d{2})/"
)
RUN_DIR_FLAT_RE = re.compile(r"^(moe[^\s/]+ \d{4}\.\d{2}\.\d{2}--\d{2}-\d{2}-\d{2})/")


@dataclass
class RunRef:
    zip_path: Path | None
    loose_dir: Path | None
    run_folder: str  # "moe_sym_bal08_v1 2026.06.01--12-11-36"
    prefix_in_zip: str  # path prefix inside zip
    score: int = 0
    has_train: bool = False
    has_val: bool = False
    has_val_noisy: bool = False
    epochs: int = 0

    @property
    def experiment_name(self) -> str:
        return self.run_folder.rsplit(" ", 1)[0]

    @property
    def run_date(self) -> str:
        m = re.search(r"(\d{4})\.(\d{2})\.(\d{2})--", self.run_folder)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return "unknown"


def _count_epochs(train_csv: Path) -> int:
    if not train_csv.is_file():
        return 0
    with train_csv.open(encoding="utf-8", errors="replace") as f:
        return max(0, sum(1 for _ in f) - 1)


def _score_run(has_train: bool, has_val: bool, has_noisy: bool, epochs: int) -> int:
    if not has_train:
        return 0
    s = epochs * 1000
    if has_noisy:
        s += 200
    if has_val:
        s += 100
    return s


def _inspect_zip(zip_path: Path) -> list[RunRef]:
    refs: dict[str, RunRef] = {}
    flat_runs: dict[str, str] = {}

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            m = RUN_DIR_RE.search("/" + name.replace("\\", "/"))
            if m:
                run_folder = m.group(1)
                prefix = name[: name.index(run_folder) + len(run_folder) + 1]
                ref = refs.setdefault(
                    run_folder,
                    RunRef(zip_path, None, run_folder, prefix),
                )
                if name.endswith("train.csv"):
                    ref.has_train = True
                elif name.endswith("validation.csv"):
                    ref.has_val = True
                elif name.endswith("validation_noisy.csv"):
                    ref.has_val_noisy = True
                continue

            m2 = RUN_DIR_FLAT_RE.match(name.replace("\\", "/"))
            if m2:
                run_folder = m2.group(1)
                prefix = run_folder + "/"
                ref = refs.setdefault(
                    run_folder,
                    RunRef(zip_path, None, run_folder, prefix),
                )
                if name.endswith("train.csv"):
                    ref.has_train = True
                elif name.endswith("validation.csv"):
                    ref.has_val = True
                elif name.endswith("validation_noisy.csv"):
                    ref.has_val_noisy = True
                continue

            # moe_run/ flat export
            if name.startswith("moe_run/") and name.endswith("train.csv"):
                flat_runs["moe_run"] = "moe_run/"
            # root-level flat (moe_hidden.zip)
            if "/" not in name.rstrip("/") and name.endswith("train.csv"):
                flat_runs["__root__"] = ""

    for key, prefix in flat_runs.items():
        if key == "__root__":
            log_name = zip_path.stem.split()[0]
            run_folder = f"{log_name} {zip_path.stem.split()[-1]}" if " " in zip_path.stem else zip_path.stem
            if " " not in run_folder and re.search(r"\d{4}\.\d{2}\.\d{2}", zip_path.stem):
                run_folder = zip_path.stem
            else:
                run_folder = zip_path.stem  # e.g. moe_hidden 2026.05.18--18-07-20
        else:
            # infer experiment from log inside zip
            log_name = "moe_run_export"
            with zipfile.ZipFile(zip_path) as zf:
                for n in zf.namelist():
                    if n.startswith("moe_run/") and n.endswith(".log"):
                        log_name = Path(n).stem
                        break
            run_folder = f"{log_name} flat_export"
        refs[run_folder] = RunRef(
            zip_path, None, run_folder, prefix, has_train=True, has_val=True
        )

    for ref in refs.values():
        ref.score = _score_run(
            ref.has_train, ref.has_val, ref.has_val_noisy, ref.epochs
        )
    return list(refs.values())


def _inspect_loose_dir(run_dir: Path) -> RunRef:
    has_train = (run_dir / "train.csv").is_file()
    has_val = (run_dir / "validation.csv").is_file()
    has_noisy = (run_dir / "validation_noisy.csv").is_file()
    epochs = _count_epochs(run_dir / "train.csv") if has_train else 0
    return RunRef(
        None,
        run_dir,
        run_dir.name,
        "",
        _score_run(has_train, has_val, has_noisy, epochs),
        has_train,
        has_val,
        has_noisy,
        epochs,
    )


def _find_loose_runs() -> list[RunRef]:
    refs = []
    if not MOE_INBOX.is_dir():
        return refs
    for train in MOE_INBOX.rglob("train.csv"):
        run_dir = train.parent
        if run_dir.name.startswith(".") or "_staging" in str(run_dir):
            continue
        refs.append(_inspect_loose_dir(run_dir))
    # dedupe by absolute path
    seen = set()
    out = []
    for r in refs:
        key = r.loose_dir.resolve() if r.loose_dir else r.run_folder
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _read_config(run_dir: Path) -> dict:
    cfg: dict = {}
    logs = list(run_dir.glob("*.log"))
    if logs:
        text = logs[0].read_text(encoding="utf-8", errors="replace")
        for key in (
            "num_experts",
            "top_k",
            "router_jitter_noise",
            "balance_loss_weight",
            "balance_loss_start_weight",
            "balance_loss_warmup_epochs",
            "load_penalty_weight",
            "adversarial_loss",
            "freeze_hidden_backbone",
            "batch_size",
            "experiment_name",
            "train_folder",
        ):
            m = re.search(r"'" + key + r"': ([^\n,]+)", text)
            if m:
                cfg[key] = m.group(1).strip().strip("'")
    pkl = run_dir / "options-and-config.pickle"
    if pkl.is_file():
        try:
            with pkl.open("rb") as f:
                data = pickle.load(f)
            if hasattr(data, "__dict__"):
                for k, v in vars(data).items():
                    if k not in cfg and v is not None:
                        cfg[k] = v
        except Exception:
            pass
    train_csv = run_dir / "train.csv"
    if train_csv.is_file():
        header = train_csv.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        n_load = len(re.findall(r"expert_load_\d+", header))
        if n_load and "num_experts" not in cfg:
            cfg["num_experts"] = str(n_load)
    tf = str(cfg.get("train_folder", ""))
    if "coco20k" in tf:
        cfg["dataset"] = "coco20k"
    elif "coco100k" in tf:
        cfg["dataset"] = "coco100k"
    else:
        cfg.setdefault("dataset", "coco100k")
    return cfg


def _branch(cfg: dict, experiment_name: str) -> str:
    if "unfrozen" in experiment_name:
        return "unfrozen_moe"
    if cfg.get("freeze_hidden_backbone") in ("False", "false", False):
        return "unfrozen_moe"
    legacy_markers = (
        "stabilized",
        "balance_v",
        "ber_route",
        "uniform_first",
        "moe_hidden",
        "anticollapse",
        "router_diag",
        "top2_ber",
        "hidden178_frozen",
    )
    if any(m in experiment_name for m in legacy_markers):
        return "legacy_early"
    return "frozen_moe"


def _expert_group(cfg: dict) -> str:
    try:
        n = int(cfg.get("num_experts", 4))
        k = int(cfg.get("top_k", 1 if n <= 4 else 2))
    except (TypeError, ValueError):
        n, k = 4, 1
    return f"{n}exp_k{k}"


def _descriptive_name(cfg: dict, experiment_name: str, epochs: int, run_date: str) -> str:
    aliases = {
        "moe_sym_bal08_v1": "sym_bal08_jitter0_bal08warm5_temp14to10",
        "moe_sym_t14_t09_v1": "sym_t14_temp09_bal004warm10",
        "moe_sym_t14_v1": "sym_t14_bal004warm10",
        "moe_routing_isolation_v1": "routing_isolation_jitter005_adv0",
        "moe_collapse_diag_8exp_v1": "collapse_diag_jitter02_loadpen01_bal06",
        "moe_unfrozen_sym_t14_v1": "sym_t14_unfrozen_bal004warm10",
        "moe8_router_diag_v2": "router_diag_v2_8exp",
        "moe4_router_diag_v2": "router_diag_v2_4exp",
    }
    base = aliases.get(experiment_name, experiment_name.replace("moe_", "")[:48])
    ep = f"ep{epochs}" if epochs else "partial"
    return f"{base}_{ep}_{run_date}"


def _dest_rel(cfg: dict, experiment_name: str, epochs: int, run_date: str) -> str:
    branch = _branch(cfg, experiment_name)
    dataset = cfg.get("dataset") or "coco100k"
    experts = _expert_group(cfg)
    try:
        batch = int(cfg.get("batch_size", 32))
    except (TypeError, ValueError):
        batch = 32
    name = _descriptive_name(cfg, experiment_name, epochs, run_date)
    return f"experiments/{branch}/{dataset}/{experts}/batch{batch}/{name}"


def _extract_zip_run(ref: RunRef, dest: Path) -> None:
    assert ref.zip_path is not None
    dest.mkdir(parents=True, exist_ok=True)
    prefix = ref.prefix_in_zip.replace("\\", "/")
    with zipfile.ZipFile(ref.zip_path) as zf:
        for member in zf.namelist():
            norm = member.replace("\\", "/")
            if prefix and not norm.startswith(prefix):
                if prefix == "" and "/" in norm:
                    # root flat: allow only known files
                    if not norm.split("/")[0] in (
                        "checkpoints",
                        "images",
                        "train.csv",
                        "validation.csv",
                        "validation_noisy.csv",
                    ) and "/" in norm:
                        if not norm.startswith(("checkpoints/", "images/")):
                            continue
                else:
                    continue
            rel = norm[len(prefix) :] if prefix else norm
            if not rel or rel.endswith("/"):
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _install_run(ref: RunRef, dest_rel: str, force: bool = False) -> str:
    dest = RESULTS / dest_rel
    if dest.exists() and not force:
        existing_epochs = _count_epochs(dest / "train.csv")
        if existing_epochs >= ref.epochs and ref.epochs > 0:
            return "skip_exists_better"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if ref.loose_dir:
        shutil.copytree(ref.loose_dir, dest)
        return "copied_loose"
    if ref.zip_path:
        _extract_zip_run(ref, dest)
        ref.epochs = _count_epochs(dest / "train.csv")
        return "extracted"
    return "noop"


def _cfg_from_zip_ref(ref: RunRef) -> dict:
    if ref.loose_dir:
        return _read_config(ref.loose_dir)
    if not ref.zip_path:
        return {}
    with zipfile.ZipFile(ref.zip_path) as zf:
        for name in zf.namelist():
            if ref.run_folder in name and name.endswith(".log"):
                text = zf.read(name).decode("utf-8", errors="replace")
                cfg = {}
                for key in (
                    "num_experts", "top_k", "batch_size",
                    "balance_loss_weight", "router_jitter_noise",
                    "load_penalty_weight", "freeze_hidden_backbone",
                    "router_temperature_start", "router_temperature_end",
                    "adversarial_loss",
                ):
                    m = re.search(r"'" + key + r"': ([^\n,]+)", text)
                    if m:
                        cfg[key] = m.group(1).strip().strip("'")
                return cfg
    return {}


def _fingerprint_from_cfg(cfg: dict) -> str:
    parts = []
    for k in sorted(cfg):
        if k in ("experiment_name", "train_folder"):
            continue
        parts.append(f"{k}={cfg.get(k, '')}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _pick_winners(refs: list[RunRef]) -> tuple[list[RunRef], list[RunRef]]:
    """One winner per config fingerprint (not per experiment_name)."""
    by_fp: dict[str, list[RunRef]] = {}
    for r in refs:
        if r.score <= 0 and not r.has_train:
            continue
        cfg = _cfg_from_zip_ref(r)
        fp = _fingerprint_from_cfg(cfg) if cfg else f"name:{r.experiment_name}"
        by_fp.setdefault(fp, []).append(r)

    winners: list[RunRef] = []
    losers: list[RunRef] = []
    for _fp, group in by_fp.items():
        group_sorted = sorted(group, key=lambda x: (x.score, x.run_folder), reverse=True)
        winners.append(group_sorted[0])
        losers.extend(group_sorted[1:])
    # partial / no train
    for r in refs:
        if r.score <= 0 and not r.has_train:
            losers.append(r)
    return winners, losers


def _inventory() -> list[RunRef]:
    refs: list[RunRef] = []
    if MOE_INBOX.is_dir():
        for z in sorted(MOE_INBOX.glob("*.zip")):
            try:
                refs.extend(_inspect_zip(z))
            except zipfile.BadZipFile:
                print("BAD ZIP:", z.name)
        refs.extend(_find_loose_runs())
    # update epochs from zip listing can't read csv; set after grouping
    return refs


def _write_index():
    manifest = []
    for run_dir in sorted(EXPERIMENTS.rglob("train.csv")):
        run_dir = run_dir.parent
        if "archive" in run_dir.parts:
            continue
        cfg = _read_config(run_dir)
        cfg["path"] = str(run_dir.relative_to(RESULTS)).replace("\\", "/")
        cfg["epochs_completed"] = _count_epochs(run_dir / "train.csv")
        cfg.setdefault("experiment_name", run_dir.name.rsplit(" ", 1)[0])
        manifest.append(cfg)
    idx = EXPERIMENTS / "INDEX.json"
    idx.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    if not MOE_INBOX.is_dir():
        raise SystemExit(f"Missing inbox: {MOE_INBOX}")

    refs = _inventory()
    print(f"Found {len(refs)} run references in inbox")

    winners, losers = _pick_winners(refs)
    print(f"Installing {len(winners)} canonical runs, archiving {len(losers)} duplicates/partials")

    log = []
    for ref in winners:
        if not ref.has_train:
            dest_rel = f"archive/partial/{ref.experiment_name}_{ref.run_date}"
            status = _install_run(ref, dest_rel)
        else:
            staging = STAGING / hashlib.md5(ref.run_folder.encode()).hexdigest()[:10]
            if ref.zip_path:
                _extract_zip_run(ref, staging)
                ref.epochs = _count_epochs(staging / "train.csv")
                cfg = _read_config(staging)
            elif ref.loose_dir:
                cfg = _read_config(ref.loose_dir)
                ref.epochs = _count_epochs(ref.loose_dir / "train.csv")
            else:
                continue
            dest_rel = _dest_rel(cfg, ref.experiment_name, ref.epochs, ref.run_date)
            if ref.loose_dir:
                status = _install_run(ref, dest_rel)
            else:
                dest = RESULTS / dest_rel
                if dest.exists():
                    existing_epochs = _count_epochs(dest / "train.csv")
                    if existing_epochs >= ref.epochs:
                        status = "skip_exists_better"
                        shutil.rmtree(staging, ignore_errors=True)
                    else:
                        shutil.rmtree(dest, ignore_errors=True)
                        shutil.move(str(staging), str(dest))
                        status = "moved_staging"
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(staging), str(dest))
                    status = "moved_staging"
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        log.append({"run": ref.run_folder, "dest": dest_rel, "status": status})
        print(status, ref.run_folder, "->", dest_rel)

    dup_dir = RESULTS / "archive" / "duplicates"
    dup_dir.mkdir(parents=True, exist_ok=True)
    for ref in losers:
        if ref.has_train and ref.score > 0:
            note = dup_dir / f"{ref.experiment_name}_{ref.run_folder.replace(' ', '_')}.source.txt"
            src = str(ref.zip_path or ref.loose_dir)
            note.write_text(f"duplicate of winner\nsource: {src}\n", encoding="utf-8")

    zip_archive = RESULTS / "archive" / "zips"
    zip_archive.mkdir(parents=True, exist_ok=True)
    for z in MOE_INBOX.glob("*.zip"):
        dest_zip = zip_archive / z.name
        if dest_zip.exists():
            dest_zip.unlink()
        shutil.move(str(z), str(dest_zip))
        print("archived zip:", z.name)

    if STAGING.exists():
        shutil.rmtree(STAGING, ignore_errors=True)

    _write_index()
    _update_readme(len(winners), len(losers))
    manifest_path = RESULTS / "experiments" / "MANIFEST_ORGANIZE.json"
    manifest_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("Done. See experiments/INDEX.json and MANIFEST_ORGANIZE.json")


def _update_readme(n_win: int, n_dup: int):
    readme = RESULTS / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    block = f"""
## Inbox processed

- Source: `moe results/` (zips moved to `archive/zips/`)
- Installed: {n_win} canonical runs under `experiments/`
- Skipped duplicates: {n_dup} (see `archive/duplicates/*.source.txt`)
- Layout: `experiments/{{frozen_moe|unfrozen_moe|legacy_early}}/coco100k/{{4exp_k1|8exp_k2}}/batch{{N}}/`

Reorganized: {datetime.now().strftime("%Y-%m-%d %H:%M")}
"""
    if "## Inbox processed" in text:
        text = re.sub(r"\n## Inbox processed.*", block, text, flags=re.S)
    else:
        text += block
    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
