#!/usr/bin/env python3
"""Replace per-variant attack tables in main.tex with the combined survey table."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = ROOT / "thesis" / "main.tex"
gen = ROOT / "thesis" / "generated" / "attack_comparison_14.tex"

content = tex.read_text(encoding="utf-8")
body = gen.read_text(encoding="utf-8")
if body.startswith("%"):
    body = "\n".join(body.splitlines()[1:]) + "\n"

if "\\label{tab:attack-14-all}" in content:
    marker_start = content.index("\\label{tab:attack-14-all}")
    start = content.rfind("\\begin{table}[H]", 0, marker_start)
    end = content.index("\\end{table}", marker_start) + len("\\end{table}")
elif "\\label{tab:attack-14-r0}" in content:
    marker_start = content.index("\\label{tab:attack-14-r0}")
    start = content.rfind("\\begin{table}[H]", 0, marker_start)
    marker_end = content.index("\\label{tab:attack-14-continue}")
    end = content.index("\\end{table}", marker_end) + len("\\end{table}")
else:
    raise SystemExit("ERROR: no attack-14 table block found in main.tex")

new_content = content[:start] + body.rstrip() + "\n" + content[end:]
old_para = (
    "Table~\\ref{tab:attack-14-r0} extends the nine-attack RQ2 table (Table~\\ref{tab:moe-attacks}) "
    "with five WAVES PIL attacks in the same BER/$\\Delta$ format; "
    "Tables~\\ref{tab:attack-14-top2}--\\ref{tab:attack-14-continue} repeat the layout for each batch-128 variant."
)
new_para = (
    "Table~\\ref{tab:attack-14-all} extends the nine-attack RQ2 table (Table~\\ref{tab:moe-attacks}) "
    "with five WAVES PIL attacks, reporting HiDDeN BER and each MoE variant side by side "
    "(MoE BER and $\\Delta$ bit accuracy per column group)."
)
if old_para in new_content:
    new_content = new_content.replace(old_para, new_para)
tex.write_text(new_content, encoding="utf-8")
print(f"Patched {tex.relative_to(ROOT)} ({end - start} chars replaced)")
