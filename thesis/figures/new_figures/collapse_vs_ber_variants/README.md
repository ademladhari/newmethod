# Collapse vs BER figure variants (COCO-100k, n=20)

| File | Encoding | Best for |
|------|----------|----------|
| `collapse_vs_ber_all_runs.png` | **Default** — colour=backbone, marker=N | Thesis main figure |
| `v1_backbone_marker_n.png` | Same as default | Clean legend |
| `v2_color_by_n.png` | Colour=N, rim style=backbone | Emphasise expert count |
| `v3_facets_n4_n8.png` | Two panels N=4 / N=8 | Compare scales separately |
| `v4_n_text_labels.png` | Backbone + marker + `N4`/`N8` text | No legend needed |
| `v5_hollow_n_edge.png` | Fill=backbone, thick edge=N colour | Print / colour-blind friendly |

Regenerate: `py -3 scripts/plot_collapse_vs_ber.py`
