# Thesis figures index

Grayscale, print-ready. Regenerate:

```bash
py -3 scripts/build_thesis_grouped_tables.py
py -3 scripts/plot_thesis_grouped_tables.py
py -3 scripts/plot_thesis_experiment_figures.py
```

## Per-experiment (`figures/experiments/`)

Each file: validation BER, noisy BER (if logged), routing stability, expert loads.

- `01_batch128_sym_t14_unfrozen_bal004warm10_ep20_2026-06-01.pdf` — **moe_unfrozen_sym_t14_v1** (SUCCESS)
- `02_runs_moe_sym_bal08_v1_2026_06_04--02-53-13.pdf` — **moe_sym_bal08_v1** (IN PROGRESS)
- `03_runs_moe_unfrozen_sym_t14_attack_v1_2026_06_04--00-28-04.pdf` — **moe_unfrozen_sym_t14_attack_v1** (IN PROGRESS)
- `04_batch16_moe4_balance_v2_100k_ep24_2026-05-30.pdf` — **moe4_balance_v2_100k** (FAILED)
- `05_batch32_moe4_balance_v3_100k_ep26_2026-05-30.pdf` — **moe4_balance_v3_100k** (FAILED)
- `06_batch32_router_diag_v2_4exp_ep17_2026-05-30.pdf` — **moe4_router_diag_v2** (FAILED)
- `07_batch24_moe4_uniform_first_100k_v1_ep19_2026-05-29.pdf` — **moe4_uniform_first_100k_v1** (FAILED)
- `08_batch16_moe8_balance_v2_100k_ep16_2026-05-30.pdf` — **moe8_balance_v2_100k** (FAILED)
- `09_batch32_moe8_balance_v3_100k_ep20_2026-05-30.pdf` — **moe8_balance_v3_100k** (FAILED)
- `10_batch32_router_diag_v2_8exp_ep13_2026-05-30.pdf` — **moe8_router_diag_v2** (FAILED)
- `11_batch24_moe8_uniform_first_100k_v1_ep13_2026-05-29.pdf` — **moe8_uniform_first_100k_v1** (FAILED)
- `12_batch32_collapse_diag_jitter02_loadpen01_bal06_ep34_2026-05-31.pdf` — **moe_collapse_diag_8exp_v1** (FAILED)
- `13_batch32_routing_isolation_jitter005_adv0_ep40_2026-05-31.pdf` — **moe_routing_isolation_v1** (FAILED)
- `14_batch32_sym_t14_temp09_bal004warm10_ep20_2026-06-01.pdf` — **moe_sym_t14_t09_v1** (FAILED)
- `15_batch32_moe4_ber_route_v1_ep43_2026-05-29.pdf` — **moe4_ber_route_v1** (FAILED)
- `16_batch12_moe4_hidden178_frozen_v1_ep88_2026-05-28.pdf` — **moe4_hidden178_frozen_v1** (FAILED)
- `17_batch32_moe8_ber_route_v1_ep33_2026-05-29.pdf` — **moe8_ber_route_v1** (FAILED)
- `18_batch12_moe8_top2_ber_retry_ep76_unknown.pdf` — **moe8_top2_ber_retry** (FAILED)
- `19_batch16_moe_hidden_ep20_2026-05-18.pdf` — **moe_hidden** (FAILED)
- `20_batch24_hidden178_frozen_anticollapse_v2_ep81_2026-05-28.pdf` — **moe_hidden178_frozen_anticollapse_v2** (FAILED)
- `21_batch12_stabilized_4exp_ep54_2026-05-21.pdf` — **moe_stabilized_4exp** (FAILED)
- `22_batch12_stabilized_v2_milder_ep113_unknown.pdf` — **moe_stabilized_v2_milder** (FAILED)
- `23_attack_eval_runs_moe_unfrozen_4exp_sparse_b128_2026_06_02--21-14-20.pdf` — **moe_unfrozen_4exp_sparse_b128** (FAILED)
- `24_runs_moe_unfrozen_4exp_sparse_top2_b128_2026_06_03--08-01-13.pdf` — **moe_unfrozen_4exp_sparse_top2_b128** (FAILED)
- `25_attack_eval_runs_moe_unfrozen_8exp_sparse_b128_2026_06_02--21-36-20.pdf` — **moe_unfrozen_8exp_sparse_b128** (FAILED)
- `26_attack_eval_runs_moe_unfrozen_8exp_sparse_b128_2026_06_03--09-06-45.pdf` — **moe_unfrozen_8exp_sparse_b128** (FAILED)
- `27_exp4continue_extracted_moe_unfrozen_sym_t14_v1_continue.pdf` — **moe_unfrozen_sym_t14_v1** (FAILED)
- `28_batch128_sym_bal08_jitter0_bal08warm5_temp14to10_ep20_2026-06-01.pdf` — **moe_sym_bal08_v1** (FAILED (frozen baseline))

## Grouped tables (`figures/grouped/`)

See `figures/grouped/README.md`.

## Attack eval (`figures/attacks/`)

- `attack_summary.pdf` / `.png`
- `atk_8exp_300k_ep15.pdf` / `.png`
- `atk_8exp_k1_ep20.pdf` / `.png`
- `atk_dense_4exp_k4_ep20.pdf` / `.png`
- `atk_main_ep20.pdf` / `.png`
- `atk_top2_ep20.pdf` / `.png`

## Summary (`figures/summary/`)

- `summary_frozen_vs_unfrozen.pdf`

## Research questions (`figures/rq/`) — **use in Chapter 4**

Regenerate: `py -3 scripts/plot_thesis_rq_figures.py`

| Figure | RQ | Section |
|--------|-----|---------|
| `RQ1_clean_quality` | RQ1 | §4.2 clean BER vs HiDDeN |
| `RQ2_attack_robustness` | RQ2 | §4.3 attack robustness |
| `RQ3_ablation_conditions` | RQ3 | §4.4 backbone / top-k / experts |
| `RQ3_routing_symmetry` | RQ3 | §4.4 train–val symmetry |
| `RQ4_inference_cost` | RQ4 | §4.6 inference cost |
| `RQ5_collapse_vs_ber` | RQ5 | §4.5 collapse correlation |
| `RQ5_failure_mode_signatures` | RQ5 | §4.5 three failure modes |
| `RQ5_training_dynamics` | RQ5 | §4.5 R0 vs collapsed |
