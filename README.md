# Re-implementing PatchTST and Applying It to Retail Demand Forecasting

**CS 5782 Introduction to Deep Learning — Final Project**  
Cornell University, Spring 2026

## Introduction

This repository re-implements the supervised version of **PatchTST** from:

Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J.  
*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.*  
ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

In addition to the benchmark replication, the project extends PatchTST to **retail demand forecasting** on the Favorita grocery sales dataset.

## Project status

This repo currently includes:

- a from-scratch PatchTST implementation
- benchmark runs on `ETTh1`, `ETTm1`, and `Weather`
- an ablation study on patching and channel-independence
- a 3-seed stability check on `ETTh1` and `ETTm1`
- a retail extension on `Favorita` with `7d / 14d / 30d` forecasting
- poster assets and the current poster PDF

Large raw data files and large training artifacts are **not** tracked on GitHub. The repo keeps code, figures, JSON summaries, poster files, and lightweight results.

## Main takeaway

The project achieves a **partial replication** of PatchTST:

- `Weather` closely matches the paper and PatchTST wins at all four horizons.
- `ETTm1` reproduces the same ranking trend: PatchTST beats DLinear and Transformer across all four horizons, but absolute MSE remains above the paper.
- `ETTh1` is only partially reproduced: PatchTST beats DLinear at `96`, but trails DLinear at `192 / 336 / 720`.
- The retail extension shows that PatchTST transfers well beyond the benchmark datasets and achieves the best MSE on all three retail horizons.

## Chosen Result

We focused on the paper's most central supervised forecasting results:

- **Table 3**: PatchTST vs. DLinear vs. vanilla Transformer on `ETTh1`, `ETTm1`, and `Weather`
- **Table 7**: ablation on patching and channel-independence

These two results capture the paper's main claims: that PatchTST improves forecasting performance and that both architectural ideas matter.

## GitHub Contents

```text
Final project/
├── code/
│   ├── patchtst.py
│   ├── baselines.py
│   ├── data_loader.py
│   ├── train.py
│   ├── visualize.py
│   ├── PatchTST_Colab.ipynb
│   └── requirements.txt
├── poster/
│   ├── poster.pdf
│   ├── assets/
│   │   └── github_repo_qr.png
│   └── figures/
├── report/
│   ├── group113_PatchTST_2page_report.tex
│   └── group113_PatchTST_2page_report.pdf
├── results/
│   ├── benchmark/
│   ├── ablation/
│   ├── favorita/
│   └── seed_sweep/
├── data/                      # local only, ignored on GitHub
├── .gitignore
└── README.md
```

## Re-implementation Details

### 1. Benchmark replication

We re-implemented supervised PatchTST/42 with:

- `seq_len = 336`
- `patch_len = 16`
- `stride = 8`
- RevIN
- channel-independence
- a shared Transformer encoder

We compared:

- `PatchTST`
- `DLinear`
- `vanilla Transformer`

on:

- `ETTh1`
- `ETTm1`
- `Weather`

with forecast horizons:

- `96`
- `192`
- `336`
- `720`

### 2. Ablation study

We ran the four standard variants:

- `P+CI` — full PatchTST
- `CI Only`
- `P Only`
- `Original` — vanilla Transformer

### 3. Seed stability check

We additionally ran PatchTST across `3 seeds` on:

- `ETTh1`
- `ETTm1`

to test whether the gap to the paper is random noise or systematic.

### 4. Retail extension

We applied PatchTST to the **Favorita grocery sales** dataset:

- one Quito store
- `33` product families as channels
- daily sales forecasting
- `7d / 14d / 30d` horizons

## Results / Insights

### Benchmark summary

| Dataset | Result summary |
|---|---|
| `ETTh1` | Partial replication only. PatchTST is best at `96`, but DLinear is better at `192 / 336 / 720`. |
| `ETTm1` | PatchTST is best at all four horizons, but absolute MSE is still above the paper. |
| `Weather` | PatchTST is best at all four horizons and is closest to the paper's reported values. |

### Retail extension

| Horizon | PatchTST MSE | DLinear MSE | Transformer MSE |
|---|---:|---:|---:|
| `7d`  | **0.940** | 1.302 | 2.736 |
| `14d` | **0.995** | 1.366 | 2.846 |
| `30d` | **0.930** | 1.424 | 2.527 |

Takeaway:

- PatchTST beats DLinear by about `27%–35%` in MSE.
- PatchTST beats the vanilla Transformer by a large margin at every retail horizon.
- Retail horizon difficulty is **not strictly monotonic** here; `30d` is slightly better than `14d`, likely due to dataset-specific seasonality and window-level volatility rather than a replication bug.

### Seed sweep

The seed sweep shows that the ETT gap is **systematic**, not just seed noise.

- `ETTh1` mean MSE ranges from `0.444` to `0.663` with std `0.001–0.004`
- `ETTm1` mean MSE ranges from `0.349` to `0.499` with std `0.000–0.002`

This means the remaining gap to the paper is much larger than seed variance.

## Selected figures

Poster-ready figures generated in this repo include:

- [results/benchmark/benchmark_trends.png](results/benchmark/benchmark_trends.png)
- [results/benchmark/gap_to_paper_heatmap.png](results/benchmark/gap_to_paper_heatmap.png)
- [results/ablation/ablation_Weather_336_single.png](results/ablation/ablation_Weather_336_single.png)
- [results/favorita/favorita_trends.png](results/favorita/favorita_trends.png)
- [results/favorita/favorita_forecast_beverages_pred30.png](results/favorita/favorita_forecast_beverages_pred30.png)
- [poster/poster.pdf](poster/poster.pdf)

## Reproduction Steps

### Local setup

```bash
pip install -r code/requirements.txt
```

### Data

`Favorita` data is not included in this repo because of GitHub size limits.

To reproduce the retail extension, manually download:

- `train.csv`
- `stores.csv`

from:

[Kaggle Store Sales: Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

and place them in `data/` as:

- `favorita_train.csv`
- `favorita_stores.csv`

### Training commands

Run everything:

```bash
python code/train.py --all --data_path ./data --save_dir ./results
```

Run each stage separately:

```bash
python code/train.py --benchmark --data_path ./data --save_dir ./results
python code/train.py --ablation --data_path ./data --save_dir ./results
python code/train.py --marketing --data_path ./data --save_dir ./results
```

Run the 3-seed ETT stability check:

```bash
python code/train.py --seed_sweep --data_path ./data --save_dir ./results
```

Summarize existing seed-sweep results only:

```bash
python code/train.py --seed_summary --save_dir ./results
```

### Colab notebook

The notebook version is:

- [code/PatchTST_Colab.ipynb](code/PatchTST_Colab.ipynb)

## Notes on GitHub contents

To keep the repository lightweight, `.gitignore` excludes:

- `data/`
- model checkpoints such as `.pt`
- cached prediction arrays such as `.npz` and `.npy`

The GitHub repo is intended to store:

- source code
- experiment summaries
- generated figures
- poster assets
- report sources

## Poster and report

- Poster PDF: [poster/poster.pdf](poster/poster.pdf)
- Poster QR code asset: [poster/assets/github_repo_qr.png](poster/assets/github_repo_qr.png)
- Report PDF: [report/group113_PatchTST_2page_report.pdf](report/group113_PatchTST_2page_report.pdf)
- Report source: [report/group113_PatchTST_2page_report.tex](report/group113_PatchTST_2page_report.tex)

## References

Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023.

## Acknowledgements

This project was completed for **CS 5782 Introduction to Deep Learning** at Cornell University in Spring 2026.
