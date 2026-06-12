"""
Patch main.tex: replace inline tabular result-tables with PNG figure environments.
Uses line-number-based extraction to avoid regex disasters.
"""
from pathlib import Path

TEX = Path("thesis/main.tex")
lines = TEX.read_text(encoding="utf-8").splitlines(keepends=True)

def find_table_end(lines, start_idx):
    """Find the line index of \\end{table} starting from start_idx."""
    for i in range(start_idx, len(lines)):
        if r'\end{table}' in lines[i]:
            return i
    return None

def replace_table(lines, label, new_content):
    """
    Find the \\begin{table} that contains `label` and replace the whole
    \\begin{table}...\\end{table} block with new_content.
    """
    # Find the \label line
    label_idx = None
    for i, l in enumerate(lines):
        if label in l:
            label_idx = i
            break
    if label_idx is None:
        print(f"  WARNING: label {label!r} not found — skipping")
        return lines

    # Search backwards for the nearest \begin{table}
    start_idx = None
    for i in range(label_idx, -1, -1):
        if r'\begin{table}' in lines[i]:
            start_idx = i
            break
    if start_idx is None:
        print(f"  WARNING: no \\begin{{table}} before {label!r} — skipping")
        return lines

    # Find matching \end{table}
    end_idx = find_table_end(lines, label_idx)
    if end_idx is None:
        print(f"  WARNING: no \\end{{table}} after {label!r} — skipping")
        return lines

    print(f"  REPLACING lines {start_idx+1}–{end_idx+1}: {label!r}")
    new_lines = new_content.splitlines(keepends=True)
    if not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'
    return lines[:start_idx] + new_lines + lines[end_idx+1:]


# ──────────────────────────────────────────────────────────────────────────────
# tab:moe-clean  — Experiment 1 results
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:moe-clean}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/01_clean_channel_fidelity.png}
\caption{Clean-channel fidelity: HiDDeN ep177 vs.\ MoE R0 on 500 COCO validation images (identity channel). Both models reach $\approx 99.7\%$ bit accuracy and identical encoder MSE.}
\label{tab:moe-clean}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# tab:moe-attacks  — Experiment 2 results
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:moe-attacks}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/02_attack_robustness_r0_vs_hidden.png}
\caption{Attack robustness: HiDDeN ep177 vs.\ MoE R0, 512 images per condition (clean-trained). Green cells indicate MoE gains $>1.5$~pp; red cells indicate losses $>1.5$~pp. MoE wins on Gaussian, crop, and cropout; loses on JPEG and resize.}
\label{tab:moe-attacks}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# tab:ablations-b128  — Architecture Search overview
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:ablations-b128}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/09_batch128_ablation_summary.png}
\caption{Complete batch-128 ablation sweep on COCO-100k. Colour coding: green = R0 (reported); purple = attack-trained variant; orange/yellow = continued training checkpoints; red = failed configurations. \texttt{BB}: backbone; \texttt{Ep}: evaluation epoch.}
\label{tab:ablations-b128}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# tab:frozen-unfrozen-b128  — Experiment 3 batch-128 results
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:frozen-unfrozen-b128}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/03_backbone_comparison_b128.png}
\caption{Backbone freezing comparison: unfrozen R0 vs.\ frozen batch-128 baseline ($N=4$, $k=1$, COCO-100k). Unfrozen training achieves $8.5\times$ lower BER and $7\times$ better routing balance. Batch-32 frozen runs (including the routing isolation run, which reached \texttt{max\_use}~$= 1.00$ after 40 epochs) confirm the pattern.}
\label{tab:frozen-unfrozen-b128}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# tab:frozen-b32  — Experiment 3 batch-32 supporting table — remove as standalone
# (already folded into the figure above)
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:frozen-b32}', "% (Batch-32 frozen data folded into Figure tab:frozen-unfrozen-b128 above)\n")

# ──────────────────────────────────────────────────────────────────────────────
# tab:topk-comparison  — Experiment 4 results
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:topk-comparison}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/04_routing_sparsity_k.png}
\caption{Effect of routing sparsity $k$ (unfrozen, $N=4$, batch 128, COCO-100k). Top-1 routing achieves the lowest BER with near-uniform utilisation; dense routing ($k=4$) forces perfect load balance at the cost of $8\times$ higher BER.}
\label{tab:topk-comparison}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# tab:nexp-comparison  — Experiment 5 results
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:nexp-comparison}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/05_expert_count_N.png}
\caption{Effect of expert count $N$ (unfrozen, $k=1$, batch 128, COCO-100k). $N=8$ produces $4$--$5\times$ higher BER and large run-to-run routing variance, indicating COCO-100k is insufficient for 8-way specialisation.}
\label{tab:nexp-comparison}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# tab:duration-comparison  — Experiment 6 results
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:duration-comparison}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/06_training_duration.png}
\caption{Validation BER and routing metrics at selected checkpoints from the R0 continuation run (ep21--60). Green rows = lower BER than R0 ep20; orange = better BER but widening routing gap; red = degraded or collapse event. Ep38 and ep52 achieve lower clean BER but show a $2$--$5\times$ larger train--validation routing gap, which translates to weaker attack robustness.}
\label{tab:duration-comparison}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# tab:moe-routing  — Experiment 8 / RQ5 results
# ──────────────────────────────────────────────────────────────────────────────
lines = replace_table(lines, r'\label{tab:moe-routing}', r"""\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/07_collapse_analysis.png}
\caption{Router collapse analysis: clean BER and attack robustness summary across all evaluated checkpoints. Colour coding: green = healthy routing; yellow/orange = continued-training drift; red = full collapse; purple = attack-trained. ``Attack wins vs HiDDeN'' counts conditions (out of 9) where MoE exceeds HiDDeN bit accuracy.}
\label{tab:moe-routing}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/tables/08_full_attack_comparison.png}
\caption{Full attack comparison: bit accuracy (\%) across all evaluated models and nine attack conditions. Green $= >1.5$~pp above HiDDeN; red $= >1.5$~pp below; yellow $=$ within 1.5~pp. The ep65 attack-trained model wins on 6/9 conditions; the fully collapsed routing isolation model wins only 1/9.}
\label{fig:full-attack-comparison}
\end{figure}
""")

# ──────────────────────────────────────────────────────────────────────────────
# Write
# ──────────────────────────────────────────────────────────────────────────────
TEX.write_text("".join(lines), encoding="utf-8")
print(f"\nDone. {len(lines)} lines written to {TEX}")
