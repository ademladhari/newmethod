# Research-question figures (Chapter 4)

Grayscale figures supporting RQ1–RQ5. Regenerate:

```bash
py -3 scripts/plot_thesis_rq_figures.py
```

| File | Research question | Use in thesis |
|------|-------------------|---------------|
| `RQ1_clean_quality` | RQ1 | §4.2 clean BER + routing @ ep20 |
| `RQ2_attack_robustness` | RQ2 | §4.3 main attack table figure |
| `RQ3_ablation_conditions` | RQ3 | §4.4 backbone / top-k / experts |
| `RQ3_routing_symmetry` | RQ3 | §4.4 train/val symmetry |
| `RQ4_inference_cost` | RQ4 | §4.6 inference schematic |
| `RQ5_collapse_vs_ber` | RQ5 | §4.5 correlation (r=+0.63, p=0.0030, n=20) |
| `RQ5_failure_mode_signatures` | RQ5 | §4.5 three failure modes |
| `RQ5_training_dynamics` | RQ5 | §4.5 R0 vs collapsed trajectories |

**Note:** RQ2 counts wins/losses from `attack_summary_main_ep20.csv` — verify JPEG/resize wording in text matches the figure.
