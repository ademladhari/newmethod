#!/usr/bin/env python3
"""Build per-variant WAVES-14 comparison tables (SSL / HiDDeN / each MoE)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SSL = Path(r"D:\waves\ssl\outputs_benchmark\results.csv")
ATTACK_ORDER = (
    "identity",
    "rotation",
    "resized_crop",
    "erasing",
    "brightness",
    "contrast",
    "blur",
    "resize_90",
    "jpeg_q50",
    "crop",
    "gaussian",
    "combo_geometric",
    "combo_photometric",
    "combined",
)


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bit_index(rows: list[dict]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for r in rows:
        method = r.get("method") or "ssl"
        if method == "ssl":
            method = "SSL"
        atk = r["attack"]
        val = r.get("bit_accuracy") or r.get("PSNR")
        if "bit_accuracy" in r:
            out[(method, atk)] = float(r["bit_accuracy"])
    return out


def _pick_best(candidates: dict[str, float], tie_pp: float = 0.005) -> str:
    if not candidates:
        return "n/a"
    best_val = max(candidates.values())
    winners = [k for k, v in candidates.items() if abs(v - best_val) <= tie_pp]
    if len(winners) > 1:
        return "tie"
    return winners[0]


def _moe_vs_hidden(moe: float, hidden: float, tie_pp: float = 0.005) -> str:
    d = moe - hidden
    if abs(d) <= tie_pp:
        return "tie"
    return "MoE" if d > 0 else "HiDDeN"


def build_tables(
    combined_csv: Path,
    out_dir: Path,
    ssl_csv: Path | None,
    tie_pp: float = 0.005,
) -> None:
    rows = _read_rows(combined_csv)
    idx = _bit_index(rows)
    ssl_idx: dict[str, float] = {}
    if ssl_csv and ssl_csv.is_file():
        for r in _read_rows(ssl_csv):
            ssl_idx[r["attack"]] = float(r["bit_accuracy"])

    methods = sorted({r["method"] for r in rows})
    hidden = "HiDDeN-ep177"
    moe_methods = [m for m in methods if m.startswith("MoE")]
    if hidden not in methods:
        raise SystemExit(f"Missing {hidden} in {combined_csv}")

    cmp_dir = out_dir / "comparison_tables"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    master_lines = [
        "# WAVES 14-attack comparison (bit accuracy)",
        "",
        f"Source: `{combined_csv}`",
        f"SSL reference: `{ssl_csv}`" if ssl_csv and ssl_csv.is_file() else "SSL: not included",
        "",
    ]
    pivot_rows: list[dict] = []

    for moe in moe_methods:
        lines = [
            f"## {moe} vs HiDDeN (+ SSL reference)",
            "",
            "| Attack | SSL (100) | HiDDeN (1k) | " + moe + " | Best | vs HiDDeN |",
            "|--------|-----------|-------------|" + "-" * (len(moe) + 2) + "|------|-----------|",
        ]
        wins = losses = ties = 0
        for atk in ATTACK_ORDER:
            ssl_v = ssl_idx.get(atk)
            h_v = idx.get((hidden, atk))
            m_v = idx.get((moe, atk))
            if h_v is None or m_v is None:
                continue
            candidates = {hidden: h_v, moe: m_v}
            if ssl_v is not None:
                candidates["SSL"] = ssl_v
            best = _pick_best(candidates, tie_pp)
            vs = _moe_vs_hidden(m_v, h_v, tie_pp)
            if vs == "MoE":
                wins += 1
            elif vs == "HiDDeN":
                losses += 1
            else:
                ties += 1
            ssl_s = f"{ssl_v:.3f}" if ssl_v is not None else "—"
            lines.append(
                f"| {atk} | {ssl_s} | {h_v:.3f} | {m_v:.3f} | {best} | {vs} |"
            )
            pivot_rows.append(
                {
                    "moe_variant": moe,
                    "attack": atk,
                    "ssl_bit_accuracy": f"{ssl_v:.6f}" if ssl_v is not None else "",
                    "hidden_bit_accuracy": f"{h_v:.6f}",
                    "moe_bit_accuracy": f"{m_v:.6f}",
                    "best": best,
                    "vs_hidden": vs,
                }
            )
        lines.extend(["", f"**MoE wins: {wins} | HiDDeN wins: {losses} | ties: {ties}**", ""])
        (cmp_dir / f"{moe.replace('/', '_')}.md").write_text("\n".join(lines), encoding="utf-8")
        master_lines.extend(lines)

    master_path = cmp_dir / "ALL_VARIANTS.md"
    master_path.write_text("\n".join(master_lines), encoding="utf-8")

    pivot_path = cmp_dir / "all_variants_long.csv"
    with pivot_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "moe_variant",
                "attack",
                "ssl_bit_accuracy",
                "hidden_bit_accuracy",
                "moe_bit_accuracy",
                "best",
                "vs_hidden",
            ],
        )
        w.writeheader()
        w.writerows(pivot_rows)

    # Wide pivot: attack rows, columns per method
    wide_path = cmp_dir / "bit_accuracy_wide.csv"
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        cols = ["attack", "SSL_ref_100"] + methods
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for atk in ATTACK_ORDER:
            row = {"attack": atk, "SSL_ref_100": f"{ssl_idx[atk]:.6f}" if atk in ssl_idx else ""}
            for m in methods:
                v = idx.get((m, atk))
                row[m] = f"{v:.6f}" if v is not None else ""
            w.writerow(row)

    print(f"Wrote {master_path}")
    print(f"Wrote {pivot_path}")
    print(f"Wrote {wide_path}")
    print(f"Per-variant markdown in {cmp_dir}/")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--combined",
        type=Path,
        default=ROOT / "results/benchmark_waves14_rerun/results_combined_all.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to parent of --combined",
    )
    p.add_argument("--ssl", type=Path, default=DEFAULT_SSL)
    p.add_argument("--no-ssl", action="store_true")
    p.add_argument("--tie-pp", type=float, default=0.005)
    args = p.parse_args()
    out = args.output or args.combined.parent
    ssl = None if args.no_ssl else args.ssl
    if not args.combined.is_file():
        raise SystemExit(f"Missing {args.combined}")
    build_tables(args.combined, out, ssl, tie_pp=args.tie_pp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
