# Training curves

Regenerate: `py -3.10 scripts/plot_training_curves.py`

## Combined

- `combined_thesis_experiments_ber_loadl1.png` — **BER + load_l1** (all reported experiments)
- `thesis_experiments/r0/combined_r0_continue_ber_loadl1.png` — R0 primary + continuation
- `thesis_experiments/4exp/combined_4exp_ber_loadl1.png` — other N=4 ablations
- `thesis_experiments/8exp/combined_8exp_ber_loadl1.png` — N=8 expert group
- `combined_thesis_experiments_ber.png` — reported thesis experiments only (Table 9 + 200k)
- `combined_thesis_experiments_max_use.png` — same runs, routing metric
- `combined_thesis_ber.png` — all thesis-labelled runs incl. diagnostics (13)
- `combined_all_ber.png` — all runs with validation logs (29)
- `combined_thesis_max_use.png` — routing overlay (13 thesis-labelled)

## Thesis experiments (`thesis_experiments/`)

Individual curves for the 10 reported runs only.


## Individual (`individual/`)

- `r0_primary_sym_t14_unfrozen_bal004w.png` — **R0 PRIMARY** (ep 1–20)
- `r0_continuation_sym_t14_unfrozen_continu.png` — **R0 continuation** (ep 1–60)
- `attack-trained_sym_t14_unfrozen_attack.png` — **Attack-trained** (ep 1–68)
- `200k_data_scale_sym_t14_unfrozen_200k_v1.png` — **200k data scale** (ep 1–16)
- `k_2_ablation_unfrozen_4exp_top2_b128.png` — **k=2 ablation** (ep 1–30)
- `k_4_dense_unfrozen_4exp_dense_k4_b.png` — **k=4 dense** (ep 1–20)
- `8exp_run_a_unfrozen_8exp_sparse_k1.png` — **8exp run A** (ep 1–20)
- `8exp_run_b_unfrozen_8exp_sparse_k1.png` — **8exp run B** (ep 1–20)
- `frozen_bal08_ep20_sym_bal08_jitter0_bal08w.png` — **Frozen bal08 ep20** (ep 1–20)
- `frozen_bal08_ep30_sym_bal08_jitter0_bal08w.png` — **Frozen bal08 ep30** (ep 1–30)
- `routing_isolation_routing_isolation_jitter.png` — **Routing isolation** (ep 1–40)
- `8exp_collapse_diag_collapse_diag_jitter02_l.png` — **8exp collapse diag** (ep 1–34)
- `frozen_t14_temp0_9_sym_t14_temp09_bal004war.png` — **Frozen t14 temp0.9** (ep 1–20)
- `legacy_moe4_balance_v2_100k_ep2.png` — **Legacy** (ep 1–24)
- `legacy_moe8_balance_v2_100k_ep1.png` — **Legacy** (ep 1–16)
- `legacy_moe4_ber_route_v1_ep43_2.png` — **Legacy** (ep 1–43)
- `legacy_moe8_uniform_first_100k.png` — **Legacy** (ep 1–13)
- `legacy_hidden178_frozen_anticol.png` — **Legacy** (ep 1–81)
- `legacy_moe8_ber_route_v1_ep33_2.png` — **Legacy** (ep 1–33)
- `legacy_moe4_hidden178_frozen_v1.png` — **Legacy** (ep 1–88)
- `legacy_moe4_uniform_first_100k.png` — **Legacy** (ep 1–19)
- `legacy_moe4_balance_v3_100k_ep2.png` — **Legacy** (ep 1–26)
- `legacy_router_diag_v2_4exp_ep17.png` — **Legacy** (ep 1–17)
- `legacy_moe8_balance_v3_100k_ep2.png` — **Legacy** (ep 1–20)
- `legacy_router_diag_v2_8exp_ep13.png` — **Legacy** (ep 1–13)
- `legacy_moe8_top2_ber_retry_ep76.png` — **Legacy** (ep 1–76)
- `legacy_stabilized_v2_milder_ep1.png` — **Legacy** (ep 1–113)
- `legacy_stabilized_4exp_ep54_202.png` — **Legacy** (ep 1–54)
- `legacy_moe_hidden_ep20_2026-05-.png` — **Legacy** (ep 1–20)