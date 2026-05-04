"""
Visualization utilities for PatchTST project.
Generates figures for the poster, report, and analysis.

Poster-ready figures:
  - Architecture diagram (patching + channel-independence)
  - Patching illustration
  - Benchmark trend lines across prediction horizons
  - Gap-to-paper heatmap
  - Ablation design matrix
  - Marketing lollipop comparison
  - Forecast with lookback context
  - Per-channel MSE breakdown
  - Store correlation heatmap
  - Weekly seasonality
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'sans-serif'


def plot_predictions(preds, targets, model_name, dataset, pred_len,
                     save_dir='./results', n_samples=3, channel=0):
    """
    Plot forecasting predictions vs ground truth.
    Shows n_samples random examples for a single channel.
    """
    fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 3.5))
    if n_samples == 1:
        axes = [axes]

    np.random.seed(42)
    indices = np.random.choice(len(preds), n_samples, replace=False)

    for ax, idx in zip(axes, indices):
        ax.plot(targets[idx, :, channel], label='Ground Truth', color='blue', linewidth=1.5)
        ax.plot(preds[idx, :, channel], label='Prediction', color='red',
                linewidth=1.5, linestyle='--')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=9)
        ax.set_title(f'Sample {idx}')

    plt.suptitle(f'{model_name} on {dataset} (pred_len={pred_len}, channel={channel})',
                 fontsize=14)
    plt.tight_layout()

    path = os.path.join(save_dir, f'{model_name}_{dataset}_{pred_len}_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_comparison_table(results_dir='./results'):
    """
    Create a bar chart comparing MSE across models for each dataset/pred_len.
    """
    results_file = os.path.join(results_dir, 'benchmark_results.json')
    if not os.path.exists(results_file):
        print(f"No benchmark results found at {results_file}")
        return

    with open(results_file) as f:
        results = json.load(f)

    # Filter out errors
    results = [r for r in results if 'error' not in r]
    if not results:
        print("No valid results to plot")
        return

    # Group by dataset — preserve PatchTST first ordering
    datasets = sorted(set(r['dataset'] for r in results))
    pred_lens = sorted(set(r['pred_len'] for r in results))

    # Color and ordering: PatchTST highlighted as the focal model
    color_map = {'PatchTST': '#1565C0', 'DLinear': '#4CAF50', 'Transformer': '#FF9800'}
    preferred_order = ['PatchTST', 'DLinear', 'Transformer']

    for dataset in datasets:
        # Only include models that actually have data on this dataset
        models_present = [m for m in preferred_order
                          if any(r['dataset'] == dataset and r['model'] == m for r in results)]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for metric_idx, metric in enumerate(['test_mse', 'test_mae']):
            ax = axes[metric_idx]
            x = np.arange(len(pred_lens))
            n = len(models_present)
            width = 0.8 / n

            for i, model in enumerate(models_present):
                values = []
                for pl in pred_lens:
                    match = [r for r in results
                             if r['dataset'] == dataset and r['model'] == model
                             and r['pred_len'] == pl]
                    values.append(match[0][metric] if match else 0)

                offset = (i - (n - 1) / 2) * width
                bars = ax.bar(x + offset, values, width, label=model,
                              color=color_map.get(model, '#999'),
                              edgecolor='white', linewidth=1)
                for b, v in zip(bars, values):
                    if v > 0:
                        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.01,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

            ax.set_xlabel('Prediction Length', fontsize=11)
            ax.set_ylabel(metric.split('_')[1].upper(), fontsize=11)
            ax.set_title(f'{dataset} — {metric.split("_")[1].upper()}', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(pred_lens)
            ax.legend(fontsize=10, loc='upper left')
            ax.grid(axis='y', alpha=0.3)
            cur_top = ax.get_ylim()[1]
            ax.set_ylim(0, cur_top * 1.12)

        plt.suptitle(f'Model Comparison on {dataset}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(results_dir, f'comparison_{dataset}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


def plot_training_curves(history, model_name, dataset, pred_len, save_dir='./results'):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()

    ax2.plot(epochs, history['val_mse'], label='Val MSE', color='blue')
    ax2.plot(epochs, history['val_mae'], label='Val MAE', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric')
    ax2.set_title('Validation Metrics')
    ax2.legend()

    plt.suptitle(f'{model_name} on {dataset} (pred_len={pred_len})', fontsize=14)
    plt.tight_layout()

    path = os.path.join(save_dir, f'{model_name}_{dataset}_{pred_len}_training.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_attention_heatmaps(model, sample_input, channel_indices=None,
                             channel_names=None, layer=-1, save_dir='./results',
                             name='attention'):
    """
    Step 2b: Visualize attention maps from PatchTST encoder.

    PatchTST processes each channel independently through the same Transformer.
    Different channels can learn different attention patterns — this is the
    "adaptability" argument from Appendix A.7. We visualize this by extracting
    attention from the final encoder layer for selected channels.

    Args:
        model: trained PatchTST model
        sample_input: (1, seq_len, n_vars) input tensor
        channel_indices: list of channel indices to visualize (default: [0, 1, 2])
        channel_names: optional list of names for those channels
        layer: which encoder layer's attention to plot (-1 = last)
        save_dir: where to save the figure
        name: prefix for the saved file

    Returns:
        attention_maps: tensor of shape (n_vars, n_heads, N, N)
    """
    import torch

    if channel_indices is None:
        channel_indices = [0, 1, 2]

    model.eval()
    with torch.no_grad():
        _, all_attn = model(sample_input, return_attention=True)

    # all_attn: list (per layer) of (batch*n_vars, n_heads, N, N) tensors
    # batch=1, so first dim is just n_vars (channels)
    attn_layer = all_attn[layer].detach().cpu()  # (n_vars, n_heads, N, N)

    # Average over heads for visualization (or could show heads separately)
    attn_avg = attn_layer.mean(dim=1)  # (n_vars, N, N)
    n_channels = len(channel_indices)

    # ---- Plot 1: averaged-over-heads attention per channel ----
    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 4.5))
    if n_channels == 1:
        axes = [axes]

    for ax, ch_idx in zip(axes, channel_indices):
        attn = attn_avg[ch_idx].numpy()
        im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0,
                       vmax=attn_avg[channel_indices].max().item())
        label = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f'Channel {ch_idx}'
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Patch Position')
        ax.set_ylabel('Query Patch Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Channel-Independent Attention Maps — Different channels, different patterns',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{name}_heatmaps.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ---- Plot 2: per-head breakdown for the first selected channel ----
    n_heads = attn_layer.shape[1]
    n_show = min(4, n_heads)
    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    ch_idx = channel_indices[0]
    label = channel_names[ch_idx] if channel_names and ch_idx < len(channel_names) else f'Channel {ch_idx}'

    head_max = attn_layer[ch_idx, :n_show].max().item()
    for h, ax in enumerate(axes):
        attn = attn_layer[ch_idx, h].numpy()
        im = ax.imshow(attn, cmap='magma', aspect='auto', vmin=0, vmax=head_max)
        ax.set_title(f'Head {h+1}', fontsize=11)
        ax.set_xlabel('Key Patch')
        if h == 0:
            ax.set_ylabel('Query Patch')

    plt.suptitle(f'Per-Head Attention for {label}', fontsize=12, y=1.02)
    plt.tight_layout()
    path2 = os.path.join(save_dir, f'{name}_heads_{label.replace(" ", "_")}.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")

    return attn_layer


def visualize_attention_for_step2b(checkpoint_path, dataset_name='ETTh1',
                                     data_path='./data', save_dir='./results/ablation',
                                     pred_len=96, results_json=None):
    """
    Step 2b runner: load a trained PatchTST model and produce attention heatmaps.

    Selects 2-3 channels with visibly different behavior (smooth vs spiky) and
    plots their attention maps side-by-side.

    Args:
        checkpoint_path: path to the .pt file
        results_json: optional path to the matching results.json — if provided,
                      reads exact model config from there (more robust)
    """
    import torch
    import numpy as np
    import json as json_lib
    from patchtst import PatchTST
    from data_loader import (load_ett_data, load_weather_data,
                              load_store_demand_data, create_dataloaders)

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    if dataset_name.startswith('ETT'):
        train_data, val_data, test_data = load_ett_data(data_path, dataset_name)
        n_vars = 7
    elif dataset_name == 'Weather':
        train_data, val_data, test_data = load_weather_data(data_path)
        n_vars = train_data.shape[1]
    else:
        raise ValueError(f"Unsupported dataset for attention viz: {dataset_name}")

    # Read config from results.json if provided
    cfg = {}
    if results_json is None:
        # Auto-derive sibling json path
        guessed = checkpoint_path.replace('_best.pt', '_results.json')
        if os.path.exists(guessed):
            results_json = guessed
    if results_json and os.path.exists(results_json):
        with open(results_json) as f:
            info = json_lib.load(f)
        cfg = info.get('config', {})

    # Build model — use config from results.json if available,
    # else fall back to paper defaults (D=16,H=4 for ETT; D=128,H=16 for others)
    if dataset_name in ['ETTh1', 'ETTh2']:
        default_dm, default_nh, default_df = 16, 4, 128
    else:
        default_dm, default_nh, default_df = 128, 16, 256

    model = PatchTST(
        enc_in=n_vars, seq_len=336, pred_len=pred_len,
        patch_len=cfg.get('patch_len', 16),
        stride=cfg.get('stride', 8),
        d_model=cfg.get('d_model', default_dm),
        n_heads=cfg.get('n_heads', default_nh),
        e_layers=cfg.get('e_layers', 3),
        d_ff=cfg.get('d_ff', default_df),
        dropout=cfg.get('dropout', 0.2),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)

    # Create a sample test batch
    _, _, test_loader, _ = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=336, pred_len=pred_len, batch_size=1,
    )
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)  # (1, 336, n_vars)

    # Pick channels with different variability (smooth vs spiky)
    var_per_channel = sample_x[0].std(dim=0).cpu().numpy()  # (n_vars,)
    sorted_idx = np.argsort(var_per_channel)
    channel_indices = [int(sorted_idx[0]), int(sorted_idx[len(sorted_idx)//2]), int(sorted_idx[-1])]

    if dataset_name.startswith('ETT'):
        channel_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    else:
        channel_names = [f'Var {i}' for i in range(n_vars)]

    print(f"\nVisualizing attention on {dataset_name}")
    print(f"Selected channels (low/mid/high variance): {channel_indices}")
    print(f"  Names: {[channel_names[i] for i in channel_indices]}")

    attn_maps = plot_attention_heatmaps(
        model, sample_x,
        channel_indices=channel_indices,
        channel_names=channel_names,
        layer=-1,
        save_dir=save_dir,
        name=f'attention_{dataset_name}',
    )

    return attn_maps


def create_results_table(results_dir='./results'):
    """Print a formatted results table (for the report)."""
    results_file = os.path.join(results_dir, 'benchmark_results.json')
    if not os.path.exists(results_file):
        print("No benchmark results found")
        return

    with open(results_file) as f:
        results = json.load(f)

    results = [r for r in results if 'error' not in r]

    # Print LaTeX-style table
    print("\n% LaTeX table")
    print("\\begin{tabular}{l|l|cc|cc|cc}")
    print("\\hline")
    print("Dataset & Pred Len & \\multicolumn{2}{c|}{PatchTST} & "
          "\\multicolumn{2}{c|}{DLinear} & \\multicolumn{2}{c}{Transformer} \\\\")
    print("& & MSE & MAE & MSE & MAE & MSE & MAE \\\\")
    print("\\hline")

    datasets = sorted(set(r['dataset'] for r in results))
    pred_lens = sorted(set(r['pred_len'] for r in results))

    for dataset in datasets:
        for pl in pred_lens:
            row = f"{dataset} & {pl}"
            for model in ['PatchTST', 'DLinear', 'Transformer']:
                match = [r for r in results
                         if r['dataset'] == dataset and r['model'] == model
                         and r['pred_len'] == pl]
                if match:
                    row += f" & {match[0]['test_mse']:.4f} & {match[0]['test_mae']:.4f}"
                else:
                    row += " & - & -"
            row += " \\\\"
            print(row)
    print("\\hline")
    print("\\end{tabular}")


# ============================================================
# Marketing Extension Visualizations
# ============================================================

def plot_store_demand_overview(data, channel_names, save_dir='./results'):
    """Plot overview of the Store Item Demand data showing seasonality patterns."""
    n_channels = min(len(channel_names), 5)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3 * n_channels), sharex=True)
    if n_channels == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, channel_names[:n_channels])):
        ax.plot(data[:, i], linewidth=0.7, color=f'C{i}')
        ax.set_ylabel('Sales')
        ax.set_title(name, fontsize=11)

    axes[-1].set_xlabel('Day')
    plt.suptitle('Store Item Demand: Daily Sales per Channel', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'store_demand_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_weekly_seasonality(data, channel_names, save_dir='./results'):
    """
    Show average sales by day-of-week to demonstrate weekly seasonality.
    This is the key pattern that patching with P=7 should capture.
    """
    n_channels = min(len(channel_names), 5)
    n_weeks = len(data) // 7

    # Reshape into weeks and compute average per day-of-week
    trimmed = data[:n_weeks * 7]
    weekly = trimmed.reshape(n_weeks, 7, -1)  # (n_weeks, 7, n_channels)
    avg_weekly = weekly.mean(axis=0)  # (7, n_channels)

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_channels):
        ax.plot(days, avg_weekly[:, i], marker='o', linewidth=2,
                label=channel_names[i])

    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Average Sales')
    ax.set_title('Weekly Seasonality Pattern in Store Demand')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'weekly_seasonality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_marketing_model_comparison(results_dir='./results/marketing'):
    """
    Create comparison charts for the marketing extension experiments.
    Reads marketing_results.json.
    """
    results_file = os.path.join(results_dir, 'marketing_results.json')
    if not os.path.exists(results_file):
        print(f"No marketing results found at {results_file}")
        return

    with open(results_file) as f:
        all_results = json.load(f)

    results = [r for r in all_results if 'error' not in r]
    if not results:
        print("No valid results")
        return

    # ---- Experiment 1: Model comparison bar chart ----
    exp1 = [r for r in results if r.get('experiment') == 'model_comparison']
    if exp1:
        models = sorted(set(r['model'] for r in exp1))
        pred_lens = sorted(set(r['pred_len'] for r in exp1))
        colors = {'PatchTST': '#2196F3', 'DLinear': '#4CAF50', 'Transformer': '#FF9800'}

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for metric_idx, metric in enumerate(['test_mse', 'test_mae']):
            ax = axes[metric_idx]
            x = np.arange(len(pred_lens))
            width = 0.25
            for i, model in enumerate(models):
                vals = []
                for pl in pred_lens:
                    match = [r for r in exp1 if r['model'] == model and r['pred_len'] == pl]
                    vals.append(match[0][metric] if match else 0)
                ax.bar(x + i * width, vals, width, label=model,
                       color=colors.get(model, '#999'))
            ax.set_xlabel('Forecast Horizon (days)')
            ax.set_ylabel(metric.split('_')[1].upper())
            ax.set_title(f'Store Demand - {metric.split("_")[1].upper()}')
            ax.set_xticks(x + width)
            ax.set_xticklabels([f'{pl}d' for pl in pred_lens])
            ax.legend()

        plt.suptitle('Model Comparison: Retail Demand Forecasting', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(results_dir, 'marketing_model_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ---- Experiment 2: Patch length ablation ----
    exp2 = [r for r in results if r.get('experiment') == 'patch_ablation']
    if exp2:
        labels = sorted(set(r.get('patch_label', '?') for r in exp2))
        pred_lens = sorted(set(r['pred_len'] for r in exp2))

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(pred_lens))
        width = 0.2
        for i, label in enumerate(labels):
            vals = []
            for pl in pred_lens:
                match = [r for r in exp2 if r.get('patch_label') == label and r['pred_len'] == pl]
                vals.append(match[0]['test_mse'] if match else 0)
            ax.bar(x + i * width, vals, width, label=label)

        ax.set_xlabel('Forecast Horizon (days)')
        ax.set_ylabel('MSE')
        ax.set_title('Patch Length Ablation: Does Weekly Alignment Help?')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{pl}d' for pl in pred_lens])
        ax.legend()
        plt.tight_layout()
        path = os.path.join(results_dir, 'patch_length_ablation.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")

    # ---- Experiment 3: Channel structure comparison ----
    exp3 = [r for r in results if r.get('experiment') == 'channel_structure']
    if exp3:
        modes = sorted(set(r.get('channel_mode', '?') for r in exp3))
        pred_lens = sorted(set(r['pred_len'] for r in exp3))

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(pred_lens))
        width = 0.25
        for i, mode in enumerate(modes):
            vals = []
            for pl in pred_lens:
                match = [r for r in exp3 if r.get('channel_mode') == mode and r['pred_len'] == pl]
                vals.append(match[0]['test_mse'] if match else 0)
            ax.bar(x + i * width, vals, width, label=mode)

        ax.set_xlabel('Forecast Horizon (days)')
        ax.set_ylabel('MSE')
        ax.set_title('Channel-Independence: Store-level vs Item-level Channels')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{pl}d' for pl in pred_lens])
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = os.path.join(results_dir, 'channel_structure_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


def plot_seed_sweep(summary_path='./results/seed_sweep/seed_sweep_summary.json',
                    save_dir='./results/seed_sweep'):
    """
    Visualize seed-sweep results: mean +- std per (dataset, horizon), with
    paper Table 3 numbers as horizontal markers for direct comparison.

    Reads `seed_sweep_summary.json` produced by summarize_seed_sweep().
    Saves two PNGs:
      - seed_sweep_comparison.png  : grouped bar chart with error bars
      - seed_sweep_table.png       : rendered table (Mean MSE +- std vs paper)
    """
    if not os.path.exists(summary_path):
        print(f"No seed sweep summary found at {summary_path}")
        print("Run summarize_seed_sweep() first.")
        return

    os.makedirs(save_dir, exist_ok=True)
    with open(summary_path) as f:
        summary = json.load(f)

    if not summary:
        print("Empty seed sweep summary.")
        return

    datasets = sorted(set(r['dataset'] for r in summary))
    pred_lens = sorted(set(r['pred_len'] for r in summary))

    # ---- Figure 1: bar chart with error bars + paper reference ----
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5),
                              squeeze=False)
    bar_color = '#1565C0'
    paper_color = '#E53935'

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[0, ax_idx]
        rows = sorted([r for r in summary if r['dataset'] == dataset],
                      key=lambda r: r['pred_len'])

        x = np.arange(len(rows))
        means = np.array([r['mean_test_mse'] for r in rows])
        stds = np.array([r['std_test_mse'] for r in rows])
        papers = np.array([r['paper_mse'] if r['paper_mse'] is not None else np.nan
                           for r in rows])
        labels = [str(r['pred_len']) for r in rows]

        # Bars with error bars
        bars = ax.bar(x - 0.18, means, width=0.36, yerr=stds, capsize=5,
                      color=bar_color, edgecolor='white', linewidth=1,
                      label='Ours (mean ± std)', error_kw={'linewidth': 1.5})
        # Paper reference bars
        ax.bar(x + 0.18, papers, width=0.36, color=paper_color,
               edgecolor='white', linewidth=1, label='Paper Table 3')

        # Annotate
        for i, r in enumerate(rows):
            ax.text(x[i] - 0.18, means[i] + stds[i] + 0.01,
                    f"{means[i]:.3f}\n±{stds[i]:.3f}",
                    ha='center', va='bottom', fontsize=8)
            ax.text(x[i] + 0.18, papers[i] + 0.005,
                    f"{papers[i]:.3f}",
                    ha='center', va='bottom', fontsize=8, color=paper_color)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Prediction Horizon (T)', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_title(f'{dataset} — PatchTST seed sensitivity ({rows[0]["n_runs"]} seeds)',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        cur_top = ax.get_ylim()[1]
        ax.set_ylim(0, cur_top * 1.15)

    fig.suptitle(
        'PatchTST Seed Sensitivity: Our 3-seed mean ± std vs Paper Table 3',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'seed_sweep_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    # ---- Figure 2: rendered comparison table ----
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.5 * len(summary)))
    ax.axis('off')

    headers = ['Dataset', 'T', 'Runs', 'Ours mean MSE', 'Std', 'Paper MSE',
               'Δ vs paper', 'Ours mean MAE', 'Paper MAE']
    rows_data = []
    for dataset in datasets:
        for pred_len in pred_lens:
            r = next((x for x in summary
                      if x['dataset'] == dataset and x['pred_len'] == pred_len), None)
            if r is None:
                continue
            paper_mse = r['paper_mse']
            paper_mae = r['paper_mae']
            delta = r['delta_mse_vs_paper']
            rows_data.append([
                dataset, str(pred_len), str(r['n_runs']),
                f"{r['mean_test_mse']:.3f}",
                f"{r['std_test_mse']:.3f}",
                f"{paper_mse:.3f}" if paper_mse is not None else '—',
                f"{delta:+.3f}" if delta is not None else '—',
                f"{r['mean_test_mae']:.3f}",
                f"{paper_mae:.3f}" if paper_mae is not None else '—',
            ])

    table = ax.table(cellText=rows_data, colLabels=headers,
                     loc='center', cellLoc='center',
                     bbox=[0.0, 0.0, 1.0, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows_data) + 1):
        for j in range(len(headers)):
            if j in (3, 4):  # Ours mean/std
                table[i, j].set_facecolor('#FFF9C4')
            elif i % 2 == 0:
                table[i, j].set_facecolor('#F5F5F5')

    fig.suptitle('Seed Sweep Summary — PatchTST mean ± std vs Paper Table 3',
                 fontsize=13, fontweight='bold', y=0.97)
    path2 = os.path.join(save_dir, 'seed_sweep_table.png')
    plt.savefig(path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path2}")


def plot_demand_forecast_examples(preds, targets, channel_names, model_name,
                                  pred_len, save_dir='./results/marketing'):
    """
    Plot demand forecasting examples showing multiple store predictions.
    Good for the poster - shows practical marketing forecasting.
    """
    n_channels = min(len(channel_names), 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    np.random.seed(42)
    idx = np.random.randint(0, len(preds))

    for i in range(n_channels):
        ax = axes[i]
        ax.plot(targets[idx, :, i], label='Actual Sales', color='blue', linewidth=2)
        ax.plot(preds[idx, :, i], label='Forecast', color='red',
                linewidth=2, linestyle='--')
        ax.set_title(channel_names[i], fontsize=12)
        ax.set_xlabel('Day')
        ax.set_ylabel('Sales (normalized)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f'{model_name}: {pred_len}-Day Demand Forecast by Store',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_demand_forecast_{pred_len}d.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Poster-Ready Figures
# ============================================================

def plot_patching_illustration(save_dir='./results'):
    """
    Illustrate how patching works: a time series is segmented into
    overlapping patches that become Transformer tokens.
    Great for the Methodology section of the poster.
    """
    np.random.seed(42)
    # Use shorter L for clearer visualization (fewer patches = readable labels)
    L = 64
    P = 16  # patch length
    S = 8   # stride
    t = np.arange(L)
    signal = np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 7) + np.random.randn(L) * 0.15

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 1.4]})

    # Top: original time series
    axes[0].plot(t, signal, color='#333', linewidth=2)
    axes[0].set_title(f'Input Univariate Time Series (L = {L})',
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].set_xlim(-0.5, L + 0.5)
    axes[0].grid(True, alpha=0.3)

    # Bottom: patched version with colored segments
    patch_starts = list(range(0, L - P + 1, S))
    n_patches = len(patch_starts)
    colors = plt.cm.tab10(np.linspace(0, 1, n_patches))

    # Faint background line
    axes[1].plot(t, signal, color='#ccc', linewidth=1, zorder=1, alpha=0.5)

    # Determine y range
    y_min, y_max = signal.min(), signal.max()
    y_range = y_max - y_min

    for i, start in enumerate(patch_starts):
        end = start + P
        ts = t[start:end]
        ys = signal[start:end]
        # Draw the patch as a colored line + fill
        axes[1].plot(ts, ys, linewidth=2.5, color=colors[i], zorder=3)
        axes[1].fill_between(ts, ys, alpha=0.25, color=colors[i], zorder=2)

        # Place label at consistent position above the patch
        mid = (start + end - 1) / 2
        axes[1].annotate(f'P{i+1}',
                         xy=(mid, y_max + 0.05 * y_range),
                         ha='center', va='bottom', fontsize=11, fontweight='bold',
                         color=colors[i],
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   edgecolor=colors[i], linewidth=1.5))

    axes[1].set_title(f'After Patching (P={P}, S={S}) → {n_patches} patches as Transformer tokens',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_xlim(-0.5, L + 0.5)
    axes[1].set_ylim(y_min - 0.1 * y_range, y_max + 0.4 * y_range)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'patching_illustration.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_architecture_diagram(save_dir='./results'):
    """
    Clean PatchTST architecture diagram showing two parallel processing tracks
    (top and bottom) sharing the same Transformer weights — illustrating
    channel-independence visually.
    """
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    blue   = dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    green  = dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    orange = dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2)
    purple = dict(boxstyle='round,pad=0.4', facecolor='#F3E5F5', edgecolor='#6A1B9A', linewidth=2)

    # Coordinates
    Y_TOP = 4.2     # Channel 1 track
    Y_MID = 2.7     # Center (input/output)
    Y_BOT = 1.2     # Channel M track

    # ---- Input ----
    ax.text(0.9, Y_MID, 'Input\n$x \\in \\mathbb{R}^{M \\times L}$',
            ha='center', va='center', fontsize=12, fontweight='bold', bbox=blue)

    # ---- Channel Split ----
    ax.text(3.1, Y_MID, 'Channel\nSplit\n(M streams)',
            ha='center', va='center', fontsize=11, fontweight='bold', bbox=green)

    # ---- Top track (Channel 1) ----
    ax.text(5.6, Y_TOP, 'Channel 1\nRevIN +\nPatching',
            ha='center', va='center', fontsize=10, fontweight='bold', bbox=orange)
    ax.text(8.4, Y_TOP, 'Transformer\nEncoder',
            ha='center', va='center', fontsize=11, fontweight='bold', bbox=purple)
    ax.text(11.1, Y_TOP, 'Flatten +\nLinear Head',
            ha='center', va='center', fontsize=10, fontweight='bold', bbox=orange)

    # ---- Bottom track (Channel M) ----
    ax.text(5.6, Y_BOT, 'Channel M\nRevIN +\nPatching',
            ha='center', va='center', fontsize=10, fontweight='bold', bbox=orange)
    ax.text(8.4, Y_BOT, 'Transformer\nEncoder',
            ha='center', va='center', fontsize=11, fontweight='bold', bbox=purple)
    ax.text(11.1, Y_BOT, 'Flatten +\nLinear Head',
            ha='center', va='center', fontsize=10, fontweight='bold', bbox=orange)

    # ---- Vertical dots between channels ----
    for x in [5.6, 8.4, 11.1]:
        ax.text(x, (Y_TOP + Y_BOT) / 2, r'$\vdots$',
                ha='center', va='center', fontsize=20, color='gray')

    # ---- Concatenate ----
    ax.text(13.6, Y_MID, 'Concat\n$\\hat{x} \\in \\mathbb{R}^{M \\times T}$',
            ha='center', va='center', fontsize=12, fontweight='bold', bbox=green)

    # ---- Arrows ----
    arrow_props = dict(arrowstyle='->', color='#333', lw=1.8)
    # Input -> Channel Split
    ax.annotate('', xy=(2.4, Y_MID), xytext=(1.8, Y_MID), arrowprops=arrow_props)

    # Channel Split -> top/bottom Patching
    ax.annotate('', xy=(4.7, Y_TOP), xytext=(3.7, Y_MID + 0.2), arrowprops=arrow_props)
    ax.annotate('', xy=(4.7, Y_BOT), xytext=(3.7, Y_MID - 0.2), arrowprops=arrow_props)

    # Patching -> Encoder (top + bottom)
    ax.annotate('', xy=(7.6, Y_TOP), xytext=(6.6, Y_TOP), arrowprops=arrow_props)
    ax.annotate('', xy=(7.6, Y_BOT), xytext=(6.6, Y_BOT), arrowprops=arrow_props)
    # Encoder -> Head (top + bottom)
    ax.annotate('', xy=(10.2, Y_TOP), xytext=(9.2, Y_TOP), arrowprops=arrow_props)
    ax.annotate('', xy=(10.2, Y_BOT), xytext=(9.2, Y_BOT), arrowprops=arrow_props)

    # Heads -> Concat
    ax.annotate('', xy=(13.0, Y_MID + 0.2), xytext=(12.0, Y_TOP), arrowprops=arrow_props)
    ax.annotate('', xy=(13.0, Y_MID - 0.2), xytext=(12.0, Y_BOT), arrowprops=arrow_props)

    # ---- Shared weights connection (bidirectional dashed between encoders) ----
    ax.annotate('', xy=(8.4, Y_BOT + 0.5), xytext=(8.4, Y_TOP - 0.5),
                arrowprops=dict(arrowstyle='<->', color='#6A1B9A', lw=2, linestyle='dashed'))
    ax.text(8.4, (Y_TOP + Y_BOT) / 2 - 0.05, ' Shared\n weights',
            ha='left', va='center', fontsize=9, color='#6A1B9A',
            fontweight='bold', style='italic')

    ax.set_title('PatchTST Architecture: Channel-Independent Patch Time Series Transformer',
                 fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    path = os.path.join(save_dir, 'architecture_diagram.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def _paper_patchtst42_reference():
    """PatchTST/42 supervised reference values from Table 3."""
    return {
        ('ETTh1', 96):  (0.375, 0.399), ('ETTh1', 192): (0.414, 0.421),
        ('ETTh1', 336): (0.431, 0.436), ('ETTh1', 720): (0.449, 0.466),
        ('ETTm1', 96):  (0.290, 0.342), ('ETTm1', 192): (0.332, 0.369),
        ('ETTm1', 336): (0.366, 0.392), ('ETTm1', 720): (0.420, 0.424),
        ('Weather', 96):  (0.152, 0.199), ('Weather', 192): (0.197, 0.243),
        ('Weather', 336): (0.249, 0.283), ('Weather', 720): (0.320, 0.335),
    }


def _load_clean_results(results_file):
    """Load a JSON list of experiment results and drop error entries."""
    if not os.path.exists(results_file):
        return []
    with open(results_file) as f:
        results = json.load(f)
    return [r for r in results if 'error' not in r]


def plot_benchmark_trends(results_dir='./results', save_dir='./results', datasets=None):
    """
    Poster-first benchmark figure: line plot of MSE vs horizon.
    This is more compact and trend-readable than repeated grouped bar charts.
    """
    results_file = os.path.join(results_dir, 'benchmark_results.json')
    results = _load_clean_results(results_file)
    if not results:
        print(f"No benchmark results found at {results_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    dataset_order = ['ETTh1', 'ETTm1', 'Weather']
    if datasets is None:
        datasets = ['ETTm1', 'Weather']
    datasets = [
        ds for ds in dataset_order
        if ds in datasets and any(r['dataset'] == ds for r in results)
    ]
    if not datasets:
        print("No matching datasets found for benchmark trends")
        return

    pred_lens = [96, 192, 336, 720]
    color_map = {'PatchTST': '#1565C0', 'DLinear': '#43A047', 'Transformer': '#EF6C00'}
    marker_map = {'PatchTST': 'o', 'DLinear': 's', 'Transformer': '^'}

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.8), squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        ds_rows = [r for r in results if r['dataset'] == dataset]
        models = [m for m in ['PatchTST', 'DLinear', 'Transformer']
                  if any(r['model'] == m for r in ds_rows)]

        for model in models:
            rows = sorted(
                [r for r in ds_rows if r['model'] == model and r['pred_len'] in pred_lens],
                key=lambda r: r['pred_len'],
            )
            x = [r['pred_len'] for r in rows]
            y = [r['test_mse'] for r in rows]
            ax.plot(
                x, y, label=model, color=color_map[model], marker=marker_map[model],
                linewidth=2.6 if model == 'PatchTST' else 2.0,
                markersize=7 if model == 'PatchTST' else 6,
            )
            for xi, yi in zip(x, y):
                ax.text(xi, yi + max(y) * 0.02, f'{yi:.3f}',
                        ha='center', va='bottom', fontsize=8, color=color_map[model])

        ax.set_title(dataset, fontsize=13, fontweight='bold')
        ax.set_xlabel('Prediction Horizon', fontsize=11)
        ax.set_ylabel('MSE', fontsize=11)
        ax.set_xticks(pred_lens)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0, 0].legend(loc='upper left', fontsize=10, frameon=True)
    fig.suptitle('Benchmark Trends: MSE vs Forecast Horizon', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, 'benchmark_trends.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_gap_to_paper_heatmap(our_results_dir='./results', save_dir='./results'):
    """
    Poster-first replication figure: heatmap of Delta MSE (ours - paper).
    Green means close to or slightly better than paper; red means worse.
    """
    results_file = os.path.join(our_results_dir, 'benchmark_results.json')
    results = _load_clean_results(results_file)
    if not results:
        print(f"No benchmark results found at {results_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    paper = _paper_patchtst42_reference()
    datasets = [ds for ds in ['ETTh1', 'ETTm1', 'Weather']
                if any(r['dataset'] == ds for r in results)]
    pred_lens = [96, 192, 336, 720]
    delta = np.full((len(datasets), len(pred_lens)), np.nan, dtype=float)

    for i, ds in enumerate(datasets):
        for j, pl in enumerate(pred_lens):
            ours = next((r['test_mse'] for r in results
                         if r.get('model') == 'PatchTST'
                         and r['dataset'] == ds and r['pred_len'] == pl), None)
            ref = paper.get((ds, pl), (None, None))[0]
            if ours is not None and ref is not None:
                delta[i, j] = ours - ref

    max_abs = np.nanmax(np.abs(delta))
    vmax = max(0.01, max_abs)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    im = ax.imshow(delta, cmap='RdYlGn_r', aspect='auto',
                   norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))

    ax.set_xticks(range(len(pred_lens)))
    ax.set_xticklabels(pred_lens)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel('Prediction Horizon', fontsize=11)
    ax.set_title('Gap to Paper: PatchTST Delta MSE (Ours - Table 3)', fontsize=14, fontweight='bold')

    for i, ds in enumerate(datasets):
        for j, pl in enumerate(pred_lens):
            val = delta[i, j]
            if np.isnan(val):
                label = '—'
                color = 'black'
            else:
                label = f'{val:+.3f}'
                color = 'white' if abs(val) > vmax * 0.55 else 'black'
            ax.text(j, i, label, ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Delta MSE', fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, 'gap_to_paper_heatmap.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_ablation_matrix(results_dir='./results/ablation', save_dir='./results/ablation',
                         dataset='ETTh1'):
    """
    Poster-first ablation figure: 2x2 design matrix for each available horizon.
    Rows = Channel-Independence (No/Yes), Columns = Patching (No/Yes).
    """
    results_file = os.path.join(results_dir, 'ablation_results.json')
    results = _load_clean_results(results_file)
    if not results:
        print(f"No ablation results found at {results_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    rows = [r for r in results if r['dataset'] == dataset]
    if not rows:
        print(f"No ablation rows found for dataset={dataset}")
        return

    pred_lens = sorted(set(r['pred_len'] for r in rows))
    label_to_pos = {
        'Original': (0, 0),
        'P Only': (0, 1),
        'CI Only': (1, 0),
        'P+CI': (1, 1),
    }
    label_to_text = {
        'Original': 'Original',
        'P Only': 'P Only',
        'CI Only': 'CI Only',
        'P+CI': 'P+CI',
    }

    all_vals = [r['test_mse'] for r in rows if r.get('ablation_label') in label_to_pos]
    vmin, vmax = min(all_vals), max(all_vals)
    teal_dark = '#2F6E73'
    teal_title = '#334144'
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'poster_teal_heat',
        ['#2F6E73', '#5D9A9D', '#9DCCCD', '#D8ECEB', '#F6FBFB'],
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        1,
        len(pred_lens),
        figsize=(5.6 * len(pred_lens), 4.8),
        squeeze=False,
    )
    if len(pred_lens) == 1:
        axes = np.array([[axes[0, 0]]])

    for ax, pred_len in zip(axes[0], pred_lens):
        mat = np.full((2, 2), np.nan, dtype=float)
        labels = {}
        sub = [r for r in rows if r['pred_len'] == pred_len]
        for r in sub:
            label = r.get('ablation_label')
            if label not in label_to_pos:
                continue
            rr, cc = label_to_pos[label]
            mat[rr, cc] = r['test_mse']
            labels[(rr, cc)] = label

        im = ax.imshow(mat, cmap=cmap, norm=norm)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Patching', 'Patching'], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No CI', 'CI'], fontsize=10)
        ax.set_title(f'{dataset} - T={pred_len}', fontsize=12, fontweight='bold', color=teal_title)

        best_val = np.nanmin(mat)
        for rr in range(2):
            for cc in range(2):
                val = mat[rr, cc]
                if np.isnan(val):
                    text = 'N/A'
                    color = 'black'
                else:
                    label = labels[(rr, cc)]
                    best_mark = '\nBEST' if abs(val - best_val) < 1e-9 else ''
                    text = f'{label_to_text[label]}\nMSE={val:.3f}{best_mark}'
                    rgba = cmap(norm(val))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    color = 'white' if luminance < 0.55 else 'black'
                ax.text(cc, rr, text, ha='center', va='center',
                        fontsize=8.5, fontweight='bold', color=color)

    fig.suptitle('Ablation Design Matrix: Effect of Patching and Channel-Independence',
                 fontsize=14, fontweight='bold', color=teal_title)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.8, bottom=0.15, wspace=0.28)
    cbar_ax = fig.add_axes([0.92, 0.22, 0.016, 0.52])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('MSE', fontsize=11)
    cbar.outline.set_edgecolor(teal_dark)
    cbar.ax.yaxis.label.set_color(teal_title)
    path = os.path.join(save_dir, f'ablation_matrix_{dataset}.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_ablation_single_panel(results_dir='./results/ablation',
                               save_dir='./results/ablation',
                               dataset='Weather', pred_len=336):
    """
    Poster-friendly single-panel ablation heatmap for one dataset/horizon.
    """
    results_file = os.path.join(results_dir, 'ablation_results.json')
    results = _load_clean_results(results_file)
    if not results:
        print(f"No ablation results found at {results_file}")
        return

    rows = [
        r for r in results
        if r['dataset'] == dataset and r['pred_len'] == pred_len
    ]
    if not rows:
        print(f"No ablation rows found for dataset={dataset}, pred_len={pred_len}")
        return

    os.makedirs(save_dir, exist_ok=True)
    label_to_pos = {
        'Original': (0, 0),
        'P Only': (0, 1),
        'CI Only': (1, 0),
        'P+CI': (1, 1),
    }
    label_to_text = {
        'Original': 'Original',
        'P Only': 'P Only',
        'CI Only': 'CI Only',
        'P+CI': 'P+CI',
    }

    local_vals = [r['test_mse'] for r in rows if r.get('ablation_label') in label_to_pos]
    vmin, vmax = min(local_vals), max(local_vals)
    teal_dark = '#2F6E73'
    teal_title = '#334144'
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'poster_teal_heat_single',
        ['#2F6E73', '#5D9A9D', '#9DCCCD', '#D8ECEB', '#F6FBFB'],
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    mat = np.full((2, 2), np.nan, dtype=float)
    labels = {}
    for r in rows:
        label = r.get('ablation_label')
        if label not in label_to_pos:
            continue
        rr, cc = label_to_pos[label]
        mat[rr, cc] = r['test_mse']
        labels[(rr, cc)] = label

    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    im = ax.imshow(mat, cmap=cmap, norm=norm)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No Patching', 'Patching'], fontsize=8.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No CI', 'CI'], fontsize=8.5)
    ax.set_title(f'{dataset} - T={pred_len}', fontsize=11.5, fontweight='bold', color=teal_title)

    best_val = np.nanmin(mat)
    for rr in range(2):
        for cc in range(2):
            val = mat[rr, cc]
            label = labels[(rr, cc)]
            best_mark = '\nBEST' if abs(val - best_val) < 1e-9 else ''
            text = f'{label_to_text[label]}\nMSE={val:.3f}{best_mark}'
            rgba = cmap(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = 'white' if luminance < 0.55 else 'black'
            ax.text(cc, rr, text, ha='center', va='center',
                    fontsize=8.2, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.06)
    cbar.set_label('MSE', fontsize=9.5)
    cbar.outline.set_edgecolor(teal_dark)
    cbar.ax.yaxis.label.set_color(teal_title)

    plt.tight_layout()
    path = os.path.join(save_dir, f'ablation_{dataset}_{pred_len}_single.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_results_vs_paper(our_results_dir='./results', save_dir='./results'):
    """
    Side-by-side comparison: our results vs paper Table 3.
    Renders as a figure (not text) for the poster.
    """
    # Paper reference numbers (PatchTST/42) — only for datasets in plan v2
    paper = _paper_patchtst42_reference()

    # Load our results
    results_file = os.path.join(our_results_dir, 'benchmark_results.json')
    ours = {}
    if os.path.exists(results_file):
        with open(results_file) as f:
            all_r = json.load(f)
        for r in all_r:
            if r.get('model') == 'PatchTST' and 'error' not in r:
                ours[(r['dataset'], r['pred_len'])] = (r['test_mse'], r['test_mae'])

    # Create table figure
    datasets = ['ETTh1', 'ETTm1', 'Weather']
    pred_lens = [96, 192, 336, 720]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')

    headers = ['Dataset', 'T', 'Paper MSE', 'Paper MAE', 'Ours MSE', 'Ours MAE']
    rows = []
    for ds in datasets:
        for pl in pred_lens:
            p_mse, p_mae = paper.get((ds, pl), ('—', '—'))
            o_mse, o_mae = ours.get((ds, pl), ('—', '—'))
            if isinstance(o_mse, float):
                o_mse, o_mae = f'{o_mse:.3f}', f'{o_mae:.3f}'
            if isinstance(p_mse, float):
                p_mse, p_mae = f'{p_mse:.3f}', f'{p_mae:.3f}'
            rows.append([ds, str(pl), p_mse, p_mae, o_mse, o_mae])

    table = ax.table(cellText=rows, colLabels=headers,
                     loc='center', cellLoc='center',
                     bbox=[0.0, 0.0, 1.0, 0.88])  # leave 12% top space for title
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Style header (full row)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row coloring + highlight our columns
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if j in (4, 5):  # Ours columns
                table[i, j].set_facecolor('#FFF9C4')
            elif i % 2 == 0:
                table[i, j].set_facecolor('#F5F5F5')

    fig.suptitle('PatchTST/42: Our Results vs Paper Table 3',
                 fontsize=15, fontweight='bold', y=0.96)

    path = os.path.join(save_dir, 'results_vs_paper.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_forecast_with_context(preds, targets, raw_test_data, seq_len, channel_names,
                                model_name='PatchTST', save_dir='./results',
                                n_examples=1, channels=None):
    """
    Plot forecasts WITH the lookback context window visible.
    Layout: rows = different channels (stacked vertically), 1 column.
    This is more readable than a multi-channel grid for poster purposes.

    Shows: [--- lookback (gray) ---][--- actual (blue) ---/--- forecast (red, dashed) ---]
    """
    if channels is None:
        channels = [0]

    n_ch = len(channels)
    pred_len = preds.shape[1]

    # Pick one representative sample with substantial variation
    np.random.seed(0)
    # Try a few candidates and pick the one with the highest variance in actual_future
    candidate_idx = np.random.choice(len(preds), min(20, len(preds)), replace=False)
    best_idx = max(candidate_idx,
                   key=lambda i: targets[i].std())
    idx = int(best_idx)

    # Stack channels vertically — each subplot is full-width and readable
    fig, axes = plt.subplots(n_ch, 1,
                              figsize=(14, 3.0 * n_ch),
                              squeeze=False)

    for row, ch in enumerate(channels):
        ax = axes[row, 0]

        context = raw_test_data[idx:idx + seq_len, ch]
        actual_future = targets[idx, :, ch]
        predicted = preds[idx, :, ch]

        t_context = np.arange(len(context))
        t_future = np.arange(len(context), len(context) + pred_len)

        # Shaded lookback region for visual clarity
        ax.axvspan(0, len(context), color='#f5f5f5', alpha=0.6, zorder=0)

        ax.plot(t_context, context, color='#666', linewidth=1.5,
                alpha=0.85, label='Lookback (input)')
        ax.plot(t_future, actual_future, color='#1565C0', linewidth=2.5,
                label='Actual')
        ax.plot(t_future, predicted, color='#E53935', linewidth=2.5,
                linestyle='--', label='Forecast')
        ax.axvline(x=len(context), color='black', linestyle=':',
                   linewidth=1.2, alpha=0.7)
        ax.grid(True, alpha=0.25)

        ch_name = channel_names[ch] if ch < len(channel_names) else f'Ch{ch}'
        ax.set_title(ch_name, fontsize=13, fontweight='bold', loc='left')
        ax.set_ylabel('Value', fontsize=11)
        if row == n_ch - 1:
            ax.set_xlabel('Time Step', fontsize=12)

        # Only put legend on the top subplot to avoid clutter
        if row == 0:
            ax.legend(fontsize=11, loc='upper left',
                       bbox_to_anchor=(1.005, 1.0), frameon=True)

    fig.suptitle(f'{model_name}: Lookback (L={seq_len}) + Forecast (T={pred_len})',
                 fontsize=15, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, f'{model_name}_forecast_context_{pred_len}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_per_channel_mse(preds, targets, channel_names, model_name='PatchTST',
                          save_dir='./results'):
    """
    Bar chart showing MSE per channel/store.
    Reveals which stores/items are harder to forecast — good marketing insight.
    """
    n_channels = preds.shape[2]
    per_ch_mse = []
    for i in range(n_channels):
        mse = np.mean((preds[:, :, i] - targets[:, :, i]) ** 2)
        per_ch_mse.append(mse)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, n_channels))
    # Sort by MSE for visual clarity
    sorted_idx = np.argsort(per_ch_mse)
    sorted_names = [channel_names[i] if i < len(channel_names) else f'Ch{i}' for i in sorted_idx]
    sorted_mse = [per_ch_mse[i] for i in sorted_idx]

    bars = ax.barh(range(n_channels), sorted_mse,
                   color=[colors[i] for i in range(n_channels)])
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('MSE')
    ax.set_title(f'{model_name}: Per-Channel Forecast Error',
                 fontsize=14, fontweight='bold')

    # Add value labels
    for i, v in enumerate(sorted_mse):
        ax.text(v + max(sorted_mse) * 0.01, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f'{model_name}_per_channel_mse.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_channel_correlation_heatmap(data, channel_names, save_dir='./results'):
    """
    Correlation heatmap between stores/channels.
    If channels are weakly correlated, channel-independence is justified.
    This is a key argument for the poster.
    """
    corr = np.corrcoef(data.T)
    n = len(channel_names)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(channel_names, fontsize=9)

    # Add correlation values
    for i in range(n):
        for j in range(n):
            color = 'white' if abs(corr[i, j]) > 0.7 else 'black'
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='Pearson Correlation')
    ax.set_title('Cross-Channel Correlation\n(Justifies Channel-Independence Design)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'channel_correlation_heatmap.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_lookback_window_effect(results_dir='./results', save_dir='./results'):
    """
    Line chart: MSE vs look-back window length (like paper Figure 2).
    Shows PatchTST benefits from longer history while baselines don't.
    Run experiments with varying seq_len first, then call this.
    """
    results_file = os.path.join(results_dir, 'lookback_results.json')
    if not os.path.exists(results_file):
        print(f"No lookback results at {results_file}")
        print("Run experiments with varying seq_len and save as lookback_results.json")
        return

    with open(results_file) as f:
        results = json.load(f)

    models = sorted(set(r['model'] for r in results))
    seq_lens = sorted(set(r['seq_len'] for r in results))
    colors = {'PatchTST': '#2196F3', 'DLinear': '#4CAF50', 'Transformer': '#FF9800'}

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in models:
        mses = []
        for sl in seq_lens:
            match = [r for r in results if r['model'] == model and r['seq_len'] == sl]
            mses.append(match[0]['test_mse'] if match else None)
        valid = [(s, m) for s, m in zip(seq_lens, mses) if m is not None]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid],
                    marker='o', linewidth=2, label=model,
                    color=colors.get(model, '#999'))

    ax.set_xlabel('Look-back Window (L)')
    ax.set_ylabel('MSE')
    ax.set_title('Effect of Look-back Window Length', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'lookback_window_effect.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_ablation_results(results_dir='./results/ablation', save_dir='./results/ablation'):
    """
    Visualize ablation study: grouped bar chart showing P+CI vs CI vs P vs Original.
    This is the most important figure for proving the paper's core contribution.
    """
    results_file = os.path.join(results_dir, 'ablation_results.json')
    if not os.path.exists(results_file):
        print(f"No ablation results at {results_file}")
        return

    with open(results_file) as f:
        results = json.load(f)

    results = [r for r in results if 'error' not in r]
    if not results:
        return

    datasets = sorted(set(r['dataset'] for r in results))
    labels_all = ['P+CI', 'CI Only', 'P Only', 'Original']
    colors = {'P+CI': '#1565C0', 'CI Only': '#4CAF50', 'P Only': '#FF9800', 'Original': '#E53935'}
    pred_lens = sorted(set(r['pred_len'] for r in results))

    for dataset in datasets:
        # Determine which variants have data anywhere on this dataset
        present_labels = [
            lbl for lbl in labels_all
            if any(r.get('ablation_label') == lbl and r['dataset'] == dataset
                   for r in results)
        ]

        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(pred_lens))
        n = len(present_labels)
        # Bar width: keep consistent regardless of how many variants exist per slot
        width = 0.8 / n

        for i, label in enumerate(present_labels):
            offset = (i - (n - 1) / 2) * width
            for j, pl in enumerate(pred_lens):
                match = [r for r in results
                         if r.get('ablation_label') == label
                         and r['dataset'] == dataset and r['pred_len'] == pl]
                if not match:
                    continue  # skip drawing missing combos
                v = match[0]['test_mse']
                bar_x = x[j] + offset
                bar = ax.bar(bar_x, v, width, label=label if j == 0 or
                              all(not [r for r in results
                                       if r.get('ablation_label') == label
                                       and r['dataset'] == dataset
                                       and r['pred_len'] == pred_lens[k]]
                                  for k in range(j))
                              else "_nolegend_",
                             color=colors[label], edgecolor='white', linewidth=1)
                ax.text(bar_x, v + 0.005, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Prediction Length', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        title = f'Ablation Study on {dataset}: Effect of Patching (P) & Channel-Independence (CI)'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pred_lens)

        # Manual legend (one entry per variant, in standard order)
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=colors[lbl], label=lbl) for lbl in present_labels]
        ax.legend(handles=legend_handles, fontsize=10, loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        cur_top = ax.get_ylim()[1]
        ax.set_ylim(0, cur_top * 1.12)

        plt.tight_layout()
        path = os.path.join(save_dir, f'ablation_{dataset}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {path}")


# ============================================================
# Step 3: Favorita-specific visualizations
# ============================================================

def plot_step3_per_category_mse(checkpoint_path, data_path='./data',
                                 save_dir='./results/favorita',
                                 results_json=None, top_n_label=10):
    """
    Step 3c: Per-category MSE bar chart on Favorita.

    Loads the trained PatchTST checkpoint, runs inference, then computes per-channel
    (per-product-family) MSE and plots a sorted horizontal bar chart.

    Args:
        checkpoint_path: PatchTST checkpoint (e.g., from Favorita pred_len=30)
        results_json: optional path to {model}_Favorita_{pred}_results.json
        top_n_label: only label the N highest+lowest error families to keep the chart clean
    """
    import torch
    import json as json_lib
    from patchtst import PatchTST
    from data_loader import load_favorita_data, create_dataloaders

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if results_json is None:
        results_json = checkpoint_path.replace('_best.pt', '_results.json')
    with open(results_json) as f:
        info = json_lib.load(f)
    seq_len = info['seq_len']
    pred_len = info['pred_len']

    # Load data + family names
    train_data, val_data, test_data, family_names, store_info = load_favorita_data(
        data_path, verbose=False,
    )
    n_families = len(family_names)

    # Build model
    cfg = info['config']
    model = PatchTST(
        enc_in=n_families, seq_len=seq_len, pred_len=pred_len,
        patch_len=cfg.get('patch_len', 16),
        stride=cfg.get('stride', 8),
        d_model=cfg.get('d_model', 16),
        n_heads=cfg.get('n_heads', 4),
        e_layers=cfg.get('e_layers', 3),
        d_ff=cfg.get('d_ff', 128),
        dropout=cfg.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # Inference
    _, _, test_loader, _ = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_len, pred_len=pred_len, batch_size=32,
    )
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            o = model(x)
            preds_list.append(o.cpu().numpy())
            targets_list.append(y.cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # Per-family MSE (averaged across all test windows and prediction steps)
    per_family_mse = ((preds - targets) ** 2).mean(axis=(0, 1))  # (n_families,)
    sorted_idx = np.argsort(per_family_mse)
    sorted_names = [family_names[i] for i in sorted_idx]
    sorted_mse = per_family_mse[sorted_idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, n_families * 0.28)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_families))
    bars = ax.barh(range(n_families), sorted_mse, color=colors)
    ax.set_yticks(range(n_families))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel('MSE (test set)', fontsize=11)
    ax.set_title(
        f'Favorita: Per-Category Forecast Error  '
        f'(store {store_info.get("store_nbr","?")} in {store_info.get("city","?")}, '
        f'pred_len={pred_len})',
        fontsize=12, fontweight='bold')

    # Annotate values
    max_mse = sorted_mse.max()
    for i, v in enumerate(sorted_mse):
        ax.text(v + max_mse * 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    ax.set_xlim(0, max_mse * 1.15)

    plt.tight_layout()
    path = os.path.join(save_dir, f'favorita_per_category_mse_pred{pred_len}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    print(f"  Easiest 3: {sorted_names[:3]}  (MSE: {sorted_mse[:3]})")
    print(f"  Hardest 3: {sorted_names[-3:]}  (MSE: {sorted_mse[-3:]})")
    return per_family_mse, family_names


def visualize_attention_for_step3(checkpoint_path, data_path='./data',
                                    save_dir='./results/favorita',
                                    results_json=None,
                                    target_categories=('GROCERY I', 'BEVERAGES', 'PRODUCE')):
    """
    Step 3c: Attention map visualization for selected product categories.

    Plots N×N attention heatmaps for 3 representative categories with different
    expected demand dynamics:
      - GROCERY I (staples — uniform attention expected)
      - BEVERAGES (weekly periodicity — periodic stripes expected)
      - PRODUCE   (perishables — recency bias expected)

    Falls back to (low/mid/high variance) channels if any of the named categories
    is missing in the data.
    """
    import torch
    import json as json_lib
    from patchtst import PatchTST
    from data_loader import load_favorita_data, create_dataloaders

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if results_json is None:
        results_json = checkpoint_path.replace('_best.pt', '_results.json')
    with open(results_json) as f:
        info = json_lib.load(f)
    seq_len = info['seq_len']
    pred_len = info['pred_len']

    # Load data
    train_data, val_data, test_data, family_names, store_info = load_favorita_data(
        data_path, verbose=False,
    )
    n_families = len(family_names)

    # Build + load model
    cfg = info['config']
    model = PatchTST(
        enc_in=n_families, seq_len=seq_len, pred_len=pred_len,
        patch_len=cfg.get('patch_len', 16),
        stride=cfg.get('stride', 8),
        d_model=cfg.get('d_model', 16),
        n_heads=cfg.get('n_heads', 4),
        e_layers=cfg.get('e_layers', 3),
        d_ff=cfg.get('d_ff', 128),
        dropout=cfg.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # Pick channel indices for target categories
    chosen = []
    for cat in target_categories:
        # Match case-insensitively, allow substring match
        matches = [i for i, n in enumerate(family_names)
                   if cat.upper() in n.upper()]
        if matches:
            chosen.append((matches[0], family_names[matches[0]]))
        else:
            print(f"  Warning: '{cat}' not found in family names")

    if len(chosen) < 3:
        print(f"  Falling back to low/mid/high variance channels")
        # Use one test sample to compute variance per channel
        _, _, test_loader, _ = create_dataloaders(
            train_data, val_data, test_data,
            seq_len=seq_len, pred_len=pred_len, batch_size=1,
        )
        sample, _ = next(iter(test_loader))
        var = sample[0].std(dim=0).cpu().numpy()
        idx_sorted = np.argsort(var)
        for idx in [idx_sorted[0], idx_sorted[len(idx_sorted)//2], idx_sorted[-1]]:
            chosen.append((int(idx), family_names[int(idx)]))
        chosen = chosen[:3]

    channel_indices = [c[0] for c in chosen]
    channel_labels = [c[1] for c in chosen]
    print(f"  Visualizing categories: {channel_labels} (indices {channel_indices})")

    # Get a test sample and run forward with attention
    _, _, test_loader, _ = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_len, pred_len=pred_len, batch_size=1,
    )
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x.to(device)
    with torch.no_grad():
        _, all_attn = model(sample_x, return_attention=True)

    # all_attn[layer]: (n_families, n_heads, N, N)  (batch=1 → first dim is n_families)
    attn_last = all_attn[-1].detach().cpu()
    attn_avg = attn_last.mean(dim=1)  # avg over heads -> (n_families, N, N)
    n_patches = attn_avg.shape[-1]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    vmax = max(attn_avg[ch].max().item() for ch in channel_indices)

    for ax, ch, label in zip(axes, channel_indices, channel_labels):
        attn = attn_avg[ch].numpy()
        im = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Patch Position')
        ax.set_ylabel('Query Patch Position')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Favorita: Per-Category Attention Maps  '
        f'(store {store_info.get("store_nbr","?")} {store_info.get("city","?")}, '
        f'pred_len={pred_len}, last encoder layer)\n'
        f'Different product categories learn different temporal attention patterns',
        fontsize=12, fontweight='bold', y=1.04,
    )
    plt.tight_layout()
    path = os.path.join(save_dir, f'favorita_attention_pred{pred_len}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

    return attn_last, channel_indices, channel_labels


def plot_favorita_category_forecast(checkpoint_path, data_path='./data',
                                    save_dir='./results/favorita',
                                    results_json=None,
                                    category='BEVERAGES',
                                    history_days=60):
    """
    Poster-friendly qualitative retail example for one product family.

    Shows one representative test window with:
      - gray history (lookback input)
      - blue actual future
      - orange PatchTST forecast

    For poster readability, only the last `history_days` of the lookback
    window are shown.
    """
    import torch
    import json as json_lib
    from patchtst import PatchTST
    from data_loader import load_favorita_data, create_dataloaders

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if results_json is None:
        results_json = checkpoint_path.replace('_best.pt', '_results.json')
    with open(results_json) as f:
        info = json_lib.load(f)
    seq_len = info['seq_len']
    pred_len = info['pred_len']

    train_data, val_data, test_data, family_names, store_info = load_favorita_data(
        data_path, verbose=False,
    )
    n_families = len(family_names)

    matches = [i for i, name in enumerate(family_names) if category.upper() in name.upper()]
    if matches:
        ch = matches[0]
        category_name = family_names[ch]
    else:
        ch = int(np.argmax(test_data.std(axis=0)))
        category_name = family_names[ch]
        print(f"  Warning: '{category}' not found. Falling back to '{category_name}'.")

    cfg = info['config']
    model = PatchTST(
        enc_in=n_families, seq_len=seq_len, pred_len=pred_len,
        patch_len=cfg.get('patch_len', 16),
        stride=cfg.get('stride', 8),
        d_model=cfg.get('d_model', 16),
        n_heads=cfg.get('n_heads', 4),
        e_layers=cfg.get('e_layers', 3),
        d_ff=cfg.get('d_ff', 128),
        dropout=cfg.get('dropout', 0.2),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    _, _, test_loader, scaler = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_len, pred_len=pred_len, batch_size=32,
    )

    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            o = model(x)
            preds_list.append(o.cpu().numpy())
            targets_list.append(y.numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    preds_raw = scaler.inverse_transform(preds.reshape(-1, n_families)).reshape(preds.shape)
    targets_raw = scaler.inverse_transform(targets.reshape(-1, n_families)).reshape(targets.shape)

    future_std = targets_raw[:, :, ch].std(axis=1)
    future_mse = ((preds_raw[:, :, ch] - targets_raw[:, :, ch]) ** 2).mean(axis=1)
    top_k = min(10, len(future_std))
    candidate_idx = np.argsort(future_std)[-top_k:]
    ranked = candidate_idx[np.argsort(future_mse[candidate_idx])]
    idx = int(ranked[len(ranked) // 2])

    context_full = test_data[idx:idx + seq_len, ch]
    actual_future = targets_raw[idx, :, ch]
    predicted = preds_raw[idx, :, ch]

    shown_history = min(history_days, seq_len)
    context = context_full[-shown_history:]

    t_context = np.arange(shown_history)
    t_future = np.arange(shown_history, shown_history + pred_len)
    cornell_red = '#B31B1B'
    orange = '#F28E2B'
    blue = '#1565C0'

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axvspan(0, shown_history, color='#F5F1EC', alpha=0.9, zorder=0)
    ax.plot(t_context, context, color='#666666', linewidth=2.0, label='History')
    ax.plot(t_future, actual_future, color=blue, linewidth=2.8, label='Actual')
    ax.plot(t_future, predicted, color=orange, linewidth=2.8, linestyle='--',
            label='PatchTST forecast')
    ax.axvline(x=shown_history, color='black', linestyle=':', linewidth=1.2, alpha=0.8)
    ax.text(shown_history + 1, max(actual_future.max(), predicted.max()) * 1.02,
            'Forecast start', fontsize=10, color='black')

    ax.set_title(f'Favorita Forecast Example - {category_name}',
                 fontsize=15, fontweight='bold', color=cornell_red)
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Daily Sales', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=10, frameon=True)

    plt.tight_layout()
    safe_name = category_name.lower().replace(' ', '_').replace('/', '_')
    path = os.path.join(save_dir, f'favorita_forecast_{safe_name}_pred{pred_len}.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    print(f"  Category: {category_name} | sample_idx={idx} | window_mse={future_mse[idx]:.3f}")
    return path, category_name, idx


def plot_step3_comparison(results_dir='./results/favorita',
                          save_dir='./results/favorita'):
    """Step 3b: PatchTST vs DLinear vs Transformer bar chart on Favorita (3 horizons)."""
    results_file = os.path.join(results_dir, 'favorita_results.json')
    if not os.path.exists(results_file):
        print(f"No results at {results_file}")
        return

    with open(results_file) as f:
        results = [r for r in json.load(f) if 'error' not in r]

    pred_lens = [7, 14, 30]
    all_models = ['PatchTST', 'DLinear', 'Transformer']
    models = [m for m in all_models
              if any(r.get('model') == m for r in results)]
    colors = {'PatchTST': '#1565C0', 'DLinear': '#4CAF50', 'Transformer': '#E65100'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for metric_idx, metric in enumerate(['test_mse', 'test_mae']):
        ax = axes[metric_idx]
        x = np.arange(len(pred_lens))
        n = len(models)
        width = 0.8 / n
        for i, m in enumerate(models):
            vals = []
            for pl in pred_lens:
                match = [r for r in results
                         if r.get('model') == m and r.get('pred_len') == pl]
                vals.append(match[0][metric] if match else 0)
            offset = (i - (n - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, label=m,
                          color=colors[m], edgecolor='white', linewidth=1)
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Forecast Horizon (days)', fontsize=11)
        ax.set_ylabel(metric.split('_')[1].upper(), fontsize=11)
        ax.set_title(f'Favorita — {metric.split("_")[1].upper()}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{pl}d' for pl in pred_lens])
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        cur_top = ax.get_ylim()[1]
        ax.set_ylim(0, cur_top * 1.25)

    fig.suptitle('Favorita: Retail Demand Forecasting (largest Quito store, 33 product families)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'favorita_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_favorita_bar_poster(results_dir='./results/favorita',
                             save_dir='./results/favorita',
                             metric='test_mse'):
    """
    Poster-friendly compact bar chart for Favorita extension.
    Uses Cornell-red-centered palette and only one metric panel.
    """
    results_file = os.path.join(results_dir, 'favorita_results.json')
    results = _load_clean_results(results_file)
    if not results:
        print(f"No Favorita results found at {results_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    pred_lens = [7, 14, 30]
    models = ['PatchTST', 'DLinear', 'Transformer']
    colors = {
        'PatchTST': '#6B8FB3',
        'DLinear': '#7E9E72',
        'Transformer': '#C7A86D',
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(pred_lens))
    width = 0.22

    for i, model in enumerate(models):
        vals = []
        for pred_len in pred_lens:
            row = next(
                r for r in results
                if r['model'] == model and r['pred_len'] == pred_len
            )
            vals.append(row[metric])

        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            color=colors[model],
            edgecolor='white',
            linewidth=0.8,
            label=model,
            zorder=3,
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.03,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
                color=colors[model],
                fontweight='bold',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f'{pl}d' for pl in pred_lens], fontsize=10)
    ax.set_xlabel('Forecast Horizon', fontsize=10)
    ax.set_ylabel('MSE', fontsize=10)
    ax.set_title('Favorita MSE', fontsize=12.5, fontweight='bold', color='#B31B1B')
    ax.grid(axis='y', alpha=0.22, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#B0A9A3')
    ax.spines['bottom'].set_color('#B0A9A3')
    ax.legend(loc='upper left', fontsize=8.5, frameon=False)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    path = os.path.join(save_dir, 'favorita_bar_mse_poster.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_favorita_trends(results_dir='./results/favorita',
                         save_dir='./results/favorita',
                         metric='test_mse'):
    """
    Poster-friendly line chart for Favorita with the same visual language
    as the benchmark trend plots.
    """
    results_file = os.path.join(results_dir, 'favorita_results.json')
    results = _load_clean_results(results_file)
    if not results:
        print(f"No Favorita results found at {results_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    pred_lens = [7, 14, 30]
    models = ['PatchTST', 'DLinear', 'Transformer']
    color_map = {'PatchTST': '#1565C0', 'DLinear': '#43A047', 'Transformer': '#EF6C00'}
    marker_map = {'PatchTST': 'o', 'DLinear': 's', 'Transformer': '^'}

    fig, ax = plt.subplots(figsize=(6.5, 4.1))

    for model in models:
        rows = sorted(
            [r for r in results if r['model'] == model and r['pred_len'] in pred_lens],
            key=lambda r: r['pred_len'],
        )
        x = [r['pred_len'] for r in rows]
        y = [r[metric] for r in rows]
        ax.plot(
            x, y, label=model, color=color_map[model], marker=marker_map[model],
            linewidth=2.6 if model == 'PatchTST' else 2.0,
            markersize=7 if model == 'PatchTST' else 6,
        )
        offset = max(y) * 0.03
        for idx_point, (xi, yi) in enumerate(zip(x, y)):
            x_text = xi + 0.06 if model == 'Transformer' and idx_point == 0 else xi
            ha = 'left' if model == 'Transformer' and idx_point == 0 else 'center'
            ax.text(
                x_text, yi + offset, f'{yi:.3f}',
                ha=ha, va='bottom', fontsize=8,
                color=color_map[model], fontweight='bold',
            )

    all_y = [r[metric] for r in results if r['model'] in models and r['pred_len'] in pred_lens]
    y_min = min(all_y)
    y_max = max(all_y)
    y_range = y_max - y_min

    ax.set_title('Favorita', fontsize=12.5, fontweight='bold')
    ax.set_xlabel('Prediction Horizon (days)', fontsize=10)
    ax.set_ylabel('MSE', fontsize=10)
    ax.set_xticks(pred_lens)
    ax.set_xticklabels([f'{pl}d' for pl in pred_lens], fontsize=10)
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.14 * y_range)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', bbox_to_anchor=(0.93, 0.975),
              fontsize=7.6, frameon=True, fancybox=True,
              framealpha=0.78, borderpad=0.35,
              handletextpad=0.5, labelspacing=0.35)

    plt.tight_layout()
    path = os.path.join(save_dir, 'favorita_trends.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_favorita_lollipop(results_dir='./results/favorita',
                           save_dir='./results/favorita',
                           metric='test_mse'):
    """
    Poster-first marketing figure: one clean lollipop/dumbbell chart using MSE.
    """
    results_file = os.path.join(results_dir, 'favorita_results.json')
    results = _load_clean_results(results_file)
    if not results:
        print(f"No Favorita results found at {results_file}")
        return

    os.makedirs(save_dir, exist_ok=True)
    pred_lens = [7, 14, 30]
    models = ['PatchTST', 'DLinear', 'Transformer']
    cornell_red = '#B31B1B'
    colors = {
        'PatchTST': cornell_red,
        'DLinear': '#7D8F69',
        'Transformer': '#C68642',
    }
    y_positions = np.arange(len(pred_lens))[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))

    for y, pred_len in zip(y_positions, pred_lens):
        rows = [r for r in results if r['pred_len'] == pred_len and r['model'] in models]
        rows = sorted(rows, key=lambda r: models.index(r['model']))
        values = [r[metric] for r in rows]
        ax.hlines(y, min(values), max(values), color='#D8D1CB', linewidth=2, zorder=1)

        for r in rows:
            x = r[metric]
            model = r['model']
            ax.scatter(x, y, s=105 if model == 'PatchTST' else 84,
                       color=colors[model], edgecolor='white', linewidth=1.5, zorder=3)
            ax.text(x + (max(values) - min(values)) * 0.03, y, f'{x:.3f}',
                    va='center', ha='left', fontsize=8.5, color=colors[model], fontweight='bold')

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=model,
               markerfacecolor=colors[model], markeredgecolor='white',
               markersize=9 if model == 'PatchTST' else 8)
        for model in models
    ]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=3, fontsize=8.5, frameon=False, handletextpad=0.6, columnspacing=1.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'{pl}d' for pl in pred_lens], fontsize=10)
    ax.set_xlabel('MSE', fontsize=10)
    ax.set_ylabel('')
    ax.set_title('Favorita Retail Forecasting MSE', fontsize=12.5, fontweight='bold',
                 color=cornell_red)
    ax.grid(axis='x', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#B0A9A3')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(save_dir, 'favorita_lollipop_mse.png')
    plt.savefig(path, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def generate_all_figures(results_dir='./results', data_path='./data'):
    """
    Master figure generator — one call produces every figure used in the
    poster + report from saved metrics + checkpoints.

    Looks under:
      {results_dir}/benchmark/   — Step 1 (run_full_benchmark output)
      {results_dir}/ablation/    — Step 2a + 2b (run_ablation_study output)
      {results_dir}/favorita/    — Step 3 (run_step3_favorita output)

    Skips any figure whose required input is missing (e.g., no favorita
    checkpoint -> skip Favorita attention map).

    Use after run_all() completes. Resume-safe — already-generated PNGs are
    overwritten.
    """
    import os
    benchmark_dir = os.path.join(results_dir, 'benchmark')
    ablation_dir  = os.path.join(results_dir, 'ablation')
    favorita_dir  = os.path.join(results_dir, 'favorita')

    print("=" * 70)
    print("  GENERATE ALL FIGURES")
    print("=" * 70)

    # 1. Methodology figures (no data needed)
    print("\n[1/5] Methodology diagrams")
    plot_patching_illustration(benchmark_dir if os.path.isdir(benchmark_dir) else results_dir)
    plot_architecture_diagram(benchmark_dir if os.path.isdir(benchmark_dir) else results_dir)

    # 2. Step 1: Main benchmark figures
    print("\n[2/5] Step 1 benchmark figures")
    if os.path.isdir(benchmark_dir):
        plot_benchmark_trends(benchmark_dir, benchmark_dir, datasets=['ETTm1', 'Weather'])
        plot_gap_to_paper_heatmap(benchmark_dir, benchmark_dir)
        # Keep legacy poster/report figures too for optional use.
        plot_results_vs_paper(benchmark_dir, benchmark_dir)
    else:
        print("  Skipping: no benchmark/ folder")

    # 3. Step 2a: Ablation figure
    print("\n[3/5] Step 2a ablation figure")
    if os.path.isdir(ablation_dir):
        plot_ablation_matrix(ablation_dir, ablation_dir, dataset='ETTh1')
        # Keep the original grouped bar chart as a fallback figure.
        plot_ablation_results(ablation_dir, ablation_dir)
    else:
        print("  Skipping: no ablation/ folder")

    # 4. Step 2b: Attention heatmaps
    print("\n[4/5] Step 2b attention heatmaps")
    for ds in ['ETTh1', 'Weather']:
        ckpt = os.path.join(ablation_dir, f'PatchTST_{ds}_96_best.pt')
        if os.path.exists(ckpt):
            try:
                visualize_attention_for_step2b(
                    checkpoint_path=ckpt, dataset_name=ds,
                    data_path=data_path, save_dir=ablation_dir, pred_len=96,
                )
            except Exception as e:
                print(f"  Skipping {ds} attention: {e}")
        else:
            print(f"  Skipping {ds} attention: no checkpoint at {ckpt}")

    # 5. Step 3: Favorita figures
    print("\n[5/5] Step 3 Favorita figures")
    if os.path.isdir(favorita_dir):
        plot_favorita_lollipop(favorita_dir, favorita_dir, metric='test_mse')
        plot_step3_comparison(favorita_dir, favorita_dir)

        fav_ckpt = os.path.join(favorita_dir, 'PatchTST_Favorita_30_best.pt')
        if os.path.exists(fav_ckpt):
            try:
                plot_favorita_category_forecast(
                    fav_ckpt, data_path, favorita_dir, category='BEVERAGES'
                )
                plot_step3_per_category_mse(fav_ckpt, data_path, favorita_dir)
                visualize_attention_for_step3(
                    fav_ckpt, data_path, favorita_dir,
                    target_categories=('GROCERY I', 'BEVERAGES', 'PRODUCE'),
                )
            except Exception as e:
                print(f"  Skipping Favorita per-category/attention: {e}")
        else:
            print(f"  Skipping per-category/attention: no checkpoint at {fav_ckpt}")
    else:
        print("  Skipping: no favorita/ folder")

    # Summary listing
    import glob
    print("\n" + "=" * 70)
    print("  ALL GENERATED FIGURES")
    print("=" * 70)
    for folder in [benchmark_dir, ablation_dir, favorita_dir]:
        if not os.path.isdir(folder):
            continue
        pngs = sorted(glob.glob(os.path.join(folder, '*.png')))
        if pngs:
            print(f"\n{folder}/ ({len(pngs)} figures):")
            for p in pngs:
                print(f"  {os.path.basename(p)}")


def generate_all_poster_figures(results_dir='./results', marketing_dir='./results/marketing',
                                 data_path='./data'):
    """
    One-stop function to generate ALL poster figures after training is complete.
    Call this at the end of the notebook.
    """
    print("=" * 60)
    print("  GENERATING ALL POSTER FIGURES")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(marketing_dir, exist_ok=True)

    # 1. Architecture + methodology diagrams
    print("\n--- Methodology Figures ---")
    plot_architecture_diagram(results_dir)
    plot_patching_illustration(results_dir)

    # 2. ETT results
    print("\n--- ETT Results Figures ---")
    plot_results_vs_paper(results_dir, results_dir)
    plot_comparison_table(results_dir)

    # 3. Marketing figures
    print("\n--- Marketing Extension Figures ---")
    try:
        from data_loader import load_store_demand_data
        train, val, test, names = load_store_demand_data(data_path, mode='by_store')
        full = np.concatenate([train, val, test], axis=0)
        plot_store_demand_overview(full, names, marketing_dir)
        plot_weekly_seasonality(full, names, marketing_dir)
        plot_channel_correlation_heatmap(full, names, marketing_dir)
    except Exception as e:
        print(f"  Skipping store demand data plots: {e}")

    plot_marketing_model_comparison(marketing_dir)

    # 4. Per-channel analysis + forecast examples from saved predictions
    print("\n--- Forecast Visualization Figures ---")
    import glob
    pred_files = glob.glob(os.path.join(marketing_dir, '*_preds.npy'))
    for pf in pred_files:
        tf = pf.replace('_preds.npy', '_targets.npy')
        if os.path.exists(tf):
            preds = np.load(pf)
            targets = np.load(tf)
            # Extract model name from filename
            base = os.path.basename(pf).replace('_preds.npy', '')
            parts = base.split('_')
            model_name = parts[0]
            try:
                plot_per_channel_mse(preds, targets, names, model_name, marketing_dir)
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("  ALL FIGURES GENERATED")
    print(f"  Check: {results_dir}/ and {marketing_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    plot_comparison_table()
    create_results_table()
    plot_marketing_model_comparison()
