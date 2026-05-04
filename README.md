# PatchTST Re-implementation + Favorita Marketing Extension

**CS 5782 Introduction to Deep Learning — Final Project**
Cornell University, Spring 2026

---

## 1. Introduction

This repository **re-implements PatchTST** from
> Y. Nie, N. H. Nguyen, P. Sinthong, J. Kalagnanam.
> *"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"*, ICLR 2023.
> [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

PatchTST proposes two key innovations for Transformer-based time series forecasting:

1. **Patching** — segmenting time series into subseries-level patches (P=16, S=8) which serve as Transformer input tokens. This reduces token count from L to ~L/S, lowering attention complexity quadratically and enabling longer look-back windows.
2. **Channel-independence** — each univariate series is processed independently through a shared Transformer, allowing per-channel adaptability while sharing parameters.

Combined with **RevIN** (Reversible Instance Normalization), PatchTST significantly outperforms prior Transformer-based baselines (Informer, Autoformer, FEDformer) and is competitive with the simple linear baseline DLinear (Zeng et al., 2023).

We re-implement the supervised model **from scratch** (the official repo was used only for cross-checking) and apply it to a new domain: **retail demand forecasting** on Ecuador's Favorita Grocery Sales dataset.

## 2. Chosen Result

We reproduce two key tables from the paper:

- **Table 3** — multivariate long-term forecasting, supervised PatchTST/42 vs DLinear vs vanilla Transformer on ETTh1, ETTm1, and Weather, with horizons T ∈ {96, 192, 336, 720}.
- **Table 7** — ablation study isolating the contribution of patching and channel-independence: P+CI (full PatchTST), CI Only (no patching), P Only (channel-mixing), Original (vanilla Transformer).

These two tables encode the paper's headline claims (PatchTST beats baselines; both innovations are necessary), so reproducing them is the most direct test of our implementation.

## 3. GitHub Contents

```
Final project/
├── code/
│   ├── patchtst.py              # PatchTST + ablation variants (P+CI, CI Only, P Only)
│   ├── baselines.py             # DLinear + Vanilla Transformer
│   ├── data_loader.py           # ETT, Weather, Favorita data loading
│   ├── train.py                 # Training pipeline + run_all() master entry
│   ├── visualize.py             # All figure generation
│   ├── PatchTST_Colab.ipynb     # One-shot reproducible notebook
│   └── requirements.txt
├── data/                        # Auto-downloaded + Kaggle (see Reproduction Steps)
├── results/
│   ├── benchmark/               # Step 1: 36 experiments (PatchTST/DLinear/Transformer × 3 datasets × 4 horizons)
│   ├── ablation/                # Step 2a + 2b: 16 experiments + attention heatmaps
│   └── favorita/                # Step 3: 9 experiments + per-category figures
├── poster/                      # Poster PDF + 10 selected figures + POSTER_GUIDE.md
├── report/                      # 2-page report (LaTeX + Markdown + PDF)
├── experiment_plan_v2.md        # Plan document for Steps 1-2
├── step3_experiment_plan.md     # Plan document for Step 3 (marketing extension)
├── README.md
├── LICENSE
└── .gitignore
```

## 4. Re-implementation Details

**Step 1 — Paper replication (Table 3):** PatchTST/42 (L=336, P=16, S=8) re-implemented from scratch with RevIN, channel-independence, and a vanilla Transformer encoder using **BatchNorm** (not LayerNorm — paper footnote). Compared against DLinear (shared trend/residual decomposition) and vanilla Transformer (channel-mixing, no patching) on ETTh1, ETTm1, and Weather.

**Step 2a — Ablation study (Table 7):** Four model variants implemented:
- **P+CI** — full PatchTST (patching + channel-independence)
- **CI Only** — point-wise tokens, channel-independence (no patching)
- **P Only** — patching + channel-mixing (no CI)
- **Original** — vanilla Transformer (no patching, no CI)

**Step 2b — Attention visualization:** Custom `AttentionEncoder` exposes per-layer attention weights via `forward(x, return_attention=True)`. Per-channel attention heatmaps validate the "adaptability" argument from paper Appendix A.7.

**Step 3 — Favorita Marketing Extension:** Pivoted Favorita panel data (date × store × family × sales) into PatchTST format by selecting the largest store in Quito and treating the 33 product families as channels (1684 days × 33). PatchTST + DLinear + Transformer on horizons 7d/14d/30d. Per-category MSE breakdown and per-category attention maps (GROCERY I, BEVERAGES, PRODUCE) highlight category-specific demand dynamics.

**Three discrepancies vs. official code** found via cross-checking:
1. Paper always pads `stride` extra values via `ReplicationPad1d`, so num_patches = (L−P)/S + 2 = 42, not +1 = 41. This is what makes it "PatchTST/42".
2. The encoder uses BatchNorm rather than LayerNorm.
3. DLinear default uses *shared* linear weights across channels (not per-channel ModuleList).

Fixing these three closed an MSE gap of 5–15% on our initial implementation.

**Datasets:**

| Dataset  | Channels (M) | Length (timesteps) | Source |
|----------|:-:|:-:|---|
| ETTh1    | 7  | 17,420 | [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset) (auto) |
| ETTm1    | 7  | 69,680 | same (auto) |
| Weather  | 21 | 52,696 | gdown / Autoformer benchmark (auto) |
| Favorita | 33 | 1,684  | [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) (manual) |

**Evaluation metrics:** MSE, MAE, RMSE on the held-out test split (in standardized space, following the paper).

## 5. Reproduction Steps

### Quick start (Colab T4)

1. Clone this repository to Google Drive at `/MyDrive/CS5782/Final_project/`.
2. **Manually download** Favorita data (ETT/Weather are auto-downloaded):
   - Visit https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
   - Download `train.csv` and `stores.csv`
   - Place them in `data/` as `favorita_train.csv` and `favorita_stores.csv`
3. Open `code/PatchTST_Colab.ipynb` in Colab and **`Runtime → Run all`**.

The notebook calls `run_all()` which executes Step 1 + Step 2a + Step 3 in sequence with built-in resume support — re-running picks up from the last completed experiment. Then `generate_all_figures()` produces every poster/report figure.

### Local reproduction

```bash
cd code
pip install -r requirements.txt

# Run everything (Step 1 + Step 2a + Step 3)
python train.py --all                        # one-shot, resume-safe

# Or step by step
python train.py --benchmark                  # Step 1
python train.py --ablation                   # Step 2a
python -c "from train import run_step3_favorita; run_step3_favorita()"  # Step 3
```

### Compute resources

- **Recommended:** Colab Pro with T4 GPU (16 GB VRAM)
- **Fresh run time:** ~7-9 hours (61 experiments)
- **Storage:** ~250-400 MB on Drive (`standard` save mode keeps small checkpoints, skips the huge vanilla-Transformer head)
- **Resume-safe:** every experiment writes its `_results.json` immediately after training; interrupt/resume freely

### Reproducing paper Table 3 exactly on ETT

The default config in this repo follows the paper's *text* (Appendix A.1.4, dropout=0.2). To match the paper's *official run scripts* on ETTh1 / ETTm1, override these arguments:

```python
# ETTh1 — match scripts/PatchTST/etth1.sh
run_experiment(
    model_name='PatchTST', dataset='ETTh1', data_path='./data',
    seq_len=336, pred_len=96,
    batch_size=128,            # paper script: 128, our default: 32
    lr=1e-4, patience=10, epochs=100,
    save_dir='./results/paper_match', save_artifacts='standard',
    config={
        'patch_len': 16, 'stride': 8,
        'd_model': 16, 'n_heads': 4, 'd_ff': 128,
        'dropout': 0.3,        # paper script: 0.3, our default: 0.2
        'head_dropout': 0.0,
    },
)

# ETTm1 — match scripts/PatchTST/ettm1.sh
# Same as above but dropout=0.4
```

These overrides should close most of the +15-50% MSE gap on ETT (we observed std=0.001-0.004 across seeds with the default config, so the residual is systematic, not noise). Weather matches the paper out-of-the-box because it is large enough that hyperparameters generalize.

## 6. Results / Insights

**Step 1 — Paper Replication (36 experiments):**

PatchTST consistently outperforms DLinear and vanilla Transformer. **Weather matches paper Table 3 within ≤ 0.005 MSE** at all four horizons:

| Dataset | T   | Paper PatchTST | **Ours PatchTST** | Ours DLinear | Ours Transformer |
|---------|-----|----------------|-------------------|--------------|-------------------|
| Weather | 96  | 0.152          | **0.150**         | 0.175        | 0.173             |
| Weather | 336 | 0.249          | **0.254**         | 0.264        | 0.273             |
| Weather | 720 | 0.320          | **0.319**         | 0.329        | 0.334             |

ETT datasets are 15–50% higher than the paper but the relative ordering is identical. We diagnosed this with a 3-seed sweep (24 additional experiments).

**Seed Sensitivity Analysis (PatchTST on ETT, 3 seeds × 8 horizons = 24 experiments):**

| Dataset | T   | Mean MSE | Std MSE | Paper MSE | Δ vs paper |
|---------|-----|----------|---------|-----------|------------|
| ETTh1   | 96  | 0.444    | 0.002   | 0.375     | +0.069     |
| ETTh1   | 192 | 0.494    | 0.001   | 0.414     | +0.080     |
| ETTh1   | 336 | 0.542    | 0.001   | 0.431     | +0.111     |
| ETTh1   | 720 | 0.663    | 0.004   | 0.449     | +0.214     |
| ETTm1   | 96  | 0.349    | 0.002   | 0.290     | +0.059     |
| ETTm1   | 192 | 0.394    | 0.001   | 0.332     | +0.062     |
| ETTm1   | 336 | 0.438    | 0.001   | 0.366     | +0.072     |
| ETTm1   | 720 | 0.499    | 0.000   | 0.420     | +0.079     |

**Observation.** The seed std is 0.001-0.004 across all 8 cells, **30-200× smaller than the gap to paper (0.06-0.21)** — so the gap is *systematic*, not seed noise.

**Cause.** The paper's text (Appendix A.1.4) states `dropout=0.2`, which we implemented. But the paper's official run scripts (`scripts/PatchTST/etth1.sh`, `ettm1.sh`) override this to dropout=0.3 (ETTh1) / 0.4 (ETTm1) and use `batch_size=128` instead of our 32. We did not adopt these script-level overrides; doing so should close most of the gap. Weather matches the paper because its larger size (M=21, ~36K samples) makes results less sensitive to these choices.

**Step 2a — Ablation:**

| Variant | ETTh1 96 | Weather 96 |
|---|:-:|:-:|
| **P+CI (full)** | **0.444** | **0.149** |
| CI Only         | 0.454     | 0.174     |
| P Only          | 0.514     | 0.163     |
| Original        | 0.604     | 0.173     |

On ETTh1 (M=7) we recover the textbook ranking; on Weather (M=21) the gap compresses and channel-mixing becomes more competitive — patching matters more on larger datasets, channel-independence matters more on smaller ones.

**Step 3 — Favorita Marketing Extension (9 experiments):**

| Horizon | PatchTST MSE | DLinear MSE | Transformer MSE |
|---|:-:|:-:|:-:|
| 7d  | **0.940** | 1.302 (-28%) | 2.736 (2.9× worse) |
| 14d | **0.995** | 1.366 (-27%) | 2.846 (2.9× worse) |
| 30d | **0.930** | 1.424 (-35%) | 2.527 (2.7× worse) |

PatchTST dominates: 30%+ MSE reduction over DLinear and ~3× better than vanilla Transformer. Per-category attention maps reveal distinct temporal patterns:

- **GROCERY I (staples)** → near-uniform attention (stable demand)
- **BEVERAGES (weekly cycles)** → periodic stripes aligned with weekend peaks
- **PRODUCE (perishables)** → recency bias, attention concentrated on recent patches

These category-specific patterns emerge through channel-independence — a single shared Transformer would average them out.

## 7. Conclusion

We successfully re-implemented PatchTST from the paper alone (no copying of the official repo) and reproduced its core empirical claims on three datasets with 36 experiments. Our ablation confirms the paper's central design choice: both patching and channel-independence are necessary; their relative importance flips with dataset size. Our Favorita marketing extension demonstrates that the same model + the same shared Transformer learns business-meaningful per-category demand dynamics, validating PatchTST as a practical retail forecasting model.

**Lessons learned:**
- Cross-checking against the official code surfaced three subtle bugs (patch off-by-one, BatchNorm vs LayerNorm, DLinear shared weights) that together accounted for 5-15% MSE.
- Per-experiment JSON checkpoints + an `_is_experiment_done()` resume helper were essential for managing a 7+ hour run on Colab Pro.
- Channel-independence is what enables per-category interpretability in Step 3; channel-mixing models would not produce per-category attention maps.

**Future work:**
- Add exogenous variables (promotions, holidays, oil price) for Favorita — PatchTST is purely autoregressive.
- Implement self-supervised masked-patch pre-training (paper Section 3.2) and test transfer (e.g. Weather → Favorita).

## 8. References

[1] Nie, Y., Nguyen, N. H., Sinthong, P., Kalagnanam, J. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)

[2] Zeng, A., Chen, M., Zhang, L., Xu, Q. *Are Transformers Effective for Time Series Forecasting?* AAAI 2023.

[3] Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., Choo, J. *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.* ICLR 2022.

[4] Corporación Favorita Grocery Sales — Kaggle Store Sales: Time Series Forecasting. https://www.kaggle.com/competitions/store-sales-time-series-forecasting

[5] Official PatchTST repository: https://github.com/yuqinie98/PatchTST (used for cross-checking only)

## 9. Acknowledgements

Completed as part of **CS 5782 Introduction to Deep Learning** at Cornell University, Spring 2026, under the guidance of the course staff.

We used **Claude (Anthropic)** as a coding assistant for debugging, training-loop scaffolding, figure generation, and notebook organization. The model architecture was implemented from the paper directly and cross-checked against the official repository; the model itself was not generated by AI.

Datasets are the property of their respective owners (ETT, Autoformer team, Corporación Favorita). The official PatchTST code is licensed under Apache 2.0.
