"""Patch main.tex: replace remaining inline tabular result-tables with figure PNGs."""
from pathlib import Path

TEX = Path("thesis/main.tex")
content = TEX.read_text(encoding="utf-8")

# ── helper ────────────────────────────────────────────────────────────────────
def replace_block(content, old, new):
    if old not in content:
        print(f"  WARNING: block not found — skipping:\n    {old[:80]!r}")
        return content
    content = content.replace(old, new, 1)
    print(f"  REPLACED: {old[:60]!r}")
    return content

# ══════════════════════════════════════════════════════════════════════════════
# 1. Experiment 5 — Expert Count N result table
# ══════════════════════════════════════════════════════════════════════════════
OLD_N = r"""\begin{table}[H]
\centering
\caption{Effect of expert count $N$ on BER and routing metrics (unfrozen, $k=1$, batch 128).}
\label{tab:nexp-comparison}
\begin{tabular}{clccc}
\toprule
$N$ & \textbf{Run} & \textbf{BER (\%)} & \texttt{max\_use} & \texttt{load\_l1} \\
\midrule
4 & R0 (ep20) & \textbf{0.32} & 0.31 & 0.08 \\
8 & 8-exp run A (ep20) & 1.70 & 0.30 & 0.53 \\
8 & 8-exp run B (ep20) & 1.36 & 0.51 & 1.11 \\
\bottomrule
\end{tabular}
\end{table}"""

NEW_N = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/05_expert_count_N.png}
\caption{Effect of expert count $N$ on BER and routing health (unfrozen, $k=1$, batch 128, COCO-100k). $N=8$ produces $4$--$5\times$ higher BER and large run-to-run routing variance, indicating insufficient data for 8-way specialisation.}
\label{tab:nexp-comparison}
\end{figure}"""

content = replace_block(content, OLD_N, NEW_N)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Experiment 6 — Training Duration result table
# ══════════════════════════════════════════════════════════════════════════════
OLD_DUR = r"""\begin{table}[H]
\centering
\caption{Validation BER and routing gap at selected checkpoints from the R0 continuation run.}
\label{tab:duration-comparison}
\begin{tabular}{ccccc}
\toprule
\textbf{Epoch} & \textbf{BER (\%)} & \texttt{max\_use} & \texttt{load\_l1} & \textbf{Note} \\
\midrule
20 (R0) & \textbf{0.32} & 0.31 & 0.08 & Reported model \\
22 & 0.14 & 0.33 & 0.12 & Briefly better BER \\
26 & 0.09 & 0.34 & 0.16 & Briefly better BER \\
30 & 0.38 & 0.35 & 0.22 & BER regressing \\
60 & 4.01 & 0.36 & 0.37 & Substantial degradation \\
\bottomrule
\end{tabular}
\end{table}"""

NEW_DUR = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/06_training_duration.png}
\caption{Validation BER and routing metrics at selected checkpoints from the R0 continuation run (ep21--60). Green rows indicate epochs with lower BER than R0 ep20; red rows indicate degraded performance. Ep38 and ep52 achieve lower BER but show increasing routing gap, while ep56 exhibits a transient collapse event (\texttt{max\_use} $= 0.52$, BER $= 12.4\%$).}
\label{tab:duration-comparison}
\end{figure}"""

content = replace_block(content, OLD_DUR, NEW_DUR)

# ══════════════════════════════════════════════════════════════════════════════
# 3. RQ5 routing collapse result table
# ══════════════════════════════════════════════════════════════════════════════
# Find the routing table by label
import re

# Find the table block that contains tab:moe-routing
pattern = re.compile(
    r'\\begin\{table\}\[H\].*?\\label\{tab:moe-routing\}.*?\\end\{table\}',
    re.DOTALL
)
m = pattern.search(content)
if m:
    old_routing = m.group(0)
    new_routing = r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/07_collapse_analysis.png}
\caption{Router collapse analysis: clean BER and attack robustness summary across all evaluated checkpoints. Rows are colour-coded by collapse severity (green = healthy, yellow = continued-training, orange = partial collapse, red = full collapse, purple = attack-trained). ``Attack wins vs HiDDeN'' counts the number of attack conditions (out of 9) on which the MoE model exceeds HiDDeN bit accuracy.}
\label{tab:moe-routing}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/08_full_attack_comparison.png}
\caption{Full attack comparison: bit accuracy (\%) across all evaluated models and nine attack conditions. Green cells indicate $>1.5$~pp above HiDDeN; red cells indicate $>1.5$~pp below; yellow cells are within 1.5~pp.}
\label{fig:full-attack-comparison}
\end{figure}"""
    content = content.replace(old_routing, new_routing, 1)
    print(f"  REPLACED: tab:moe-routing table")
else:
    print("  WARNING: tab:moe-routing table not found")

# ══════════════════════════════════════════════════════════════════════════════
# Write out
# ══════════════════════════════════════════════════════════════════════════════
TEX.write_text(content, encoding="utf-8")
print("\nDone. thesis/main.tex updated.")
