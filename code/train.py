"""
Training and evaluation pipeline for PatchTST and baselines.
Designed to run on Google Colab Pro with GPU.
"""

import os
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

from patchtst import PatchTST, PatchTST_CI_Only, PatchTST_P_Only
from baselines import DLinear, VanillaTransformer
from data_loader import (
    load_ett_data, load_weather_data, load_custom_csv,
    load_store_demand_data, load_favorita_data,
    create_dataloaders, download_ett_data
)


def compute_metrics(preds, targets):
    """Compute MSE, MAE, RMSE."""
    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)
    return {'mse': mse, 'mae': mae, 'rmse': rmse}


PAPER_PATCHTST42 = {
    ('ETTh1', 96):  {'mse': 0.375, 'mae': 0.399},
    ('ETTh1', 192): {'mse': 0.414, 'mae': 0.421},
    ('ETTh1', 336): {'mse': 0.431, 'mae': 0.436},
    ('ETTh1', 720): {'mse': 0.449, 'mae': 0.466},
    ('ETTm1', 96):  {'mse': 0.290, 'mae': 0.342},
    ('ETTm1', 192): {'mse': 0.332, 'mae': 0.369},
    ('ETTm1', 336): {'mse': 0.366, 'mae': 0.392},
    ('ETTm1', 720): {'mse': 0.420, 'mae': 0.424},
}


def set_seed(seed, deterministic=True):
    """Set RNG seeds for Python, NumPy, and PyTorch."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def _make_base_name(model_name, dataset, pred_len, experiment_tag=None):
    """Create a shared artifact basename for one experiment."""
    base = f'{model_name}_{dataset}_{pred_len}'
    if experiment_tag:
        base = f'{base}_{experiment_tag}'
    return base


def _upsert_result_record(all_results, result):
    """Insert or replace one result record in a consolidated JSON list."""
    key = (
        result.get('model'),
        result.get('dataset'),
        result.get('pred_len'),
        result.get('seed'),
        result.get('experiment_tag'),
    )
    for idx, existing in enumerate(all_results):
        existing_key = (
            existing.get('model'),
            existing.get('dataset'),
            existing.get('pred_len'),
            existing.get('seed'),
            existing.get('experiment_tag'),
        )
        if existing_key == key:
            all_results[idx] = result
            return
    all_results.append(result)


def _parse_int_list(raw_text):
    """Parse a comma-separated string of integers."""
    return [int(x.strip()) for x in raw_text.split(',') if x.strip()]


def _parse_str_list(raw_text):
    """Parse a comma-separated string of tokens."""
    return [x.strip() for x in raw_text.split(',') if x.strip()]


def _resolve_seed_sweep_dir(save_dir):
    """Keep seed-sweep outputs separate from the main benchmark results."""
    norm = os.path.normpath(save_dir)
    return save_dir if os.path.basename(norm) == 'seed_sweep' else os.path.join(save_dir, 'seed_sweep')


def _ensure_dir(path):
    """
    Create an output directory and surface a clearer message when Colab Drive
    has been disconnected.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Cannot create output directory: {path}. "
            "If you are running in Colab, Google Drive is likely disconnected. "
            "Remount Drive or switch save_dir to a local path such as /content/results/seed_sweep."
        ) from e


def summarize_seed_sweep(results, save_dir=None):
    """
    Aggregate seed-sweep metrics and compare their mean against paper Table 3.
    """
    summary = []
    filtered = [
        r for r in results
        if r.get('model') == 'PatchTST'
        and r.get('dataset') in ('ETTh1', 'ETTm1')
        and r.get('seed') is not None
        and 'error' not in r
    ]

    print("\n" + "=" * 92)
    print("Seed Sweep Summary (PatchTST on ETTh1/ETTm1)")
    print("=" * 92)
    print(f"{'Dataset':<8} {'T':<5} {'Runs':<6} {'Mean MSE':<10} {'Std MSE':<10} "
          f"{'Paper MSE':<10} {'Delta':<10} {'Mean MAE':<10} {'Paper MAE':<10}")
    print("-" * 92)

    for dataset in ['ETTh1', 'ETTm1']:
        for pred_len in [96, 192, 336, 720]:
            group = [r for r in filtered if r['dataset'] == dataset and r['pred_len'] == pred_len]
            if not group:
                continue

            mse_values = np.array([r['test_mse'] for r in group], dtype=np.float64)
            mae_values = np.array([r['test_mae'] for r in group], dtype=np.float64)
            paper_ref = PAPER_PATCHTST42.get((dataset, pred_len))
            paper_mse = paper_ref['mse'] if paper_ref else None
            paper_mae = paper_ref['mae'] if paper_ref else None

            record = {
                'dataset': dataset,
                'pred_len': pred_len,
                'n_runs': len(group),
                'seeds': sorted(int(r['seed']) for r in group),
                'mean_test_mse': float(mse_values.mean()),
                'std_test_mse': float(mse_values.std(ddof=0)),
                'min_test_mse': float(mse_values.min()),
                'max_test_mse': float(mse_values.max()),
                'mean_test_mae': float(mae_values.mean()),
                'std_test_mae': float(mae_values.std(ddof=0)),
                'min_test_mae': float(mae_values.min()),
                'max_test_mae': float(mae_values.max()),
                'paper_mse': float(paper_mse) if paper_mse is not None else None,
                'paper_mae': float(paper_mae) if paper_mae is not None else None,
                'delta_mse_vs_paper': float(mse_values.mean() - paper_mse) if paper_mse is not None else None,
                'delta_mae_vs_paper': float(mae_values.mean() - paper_mae) if paper_mae is not None else None,
            }
            summary.append(record)

            paper_mse_str = f"{paper_mse:.3f}" if paper_mse is not None else '—'
            paper_mae_str = f"{paper_mae:.3f}" if paper_mae is not None else '—'
            delta_str = f"{record['delta_mse_vs_paper']:+.3f}" if paper_mse is not None else '—'
            print(f"{dataset:<8} {pred_len:<5} {len(group):<6} {record['mean_test_mse']:<10.3f} "
                  f"{record['std_test_mse']:<10.3f} {paper_mse_str:<10} {delta_str:<10} "
                  f"{record['mean_test_mae']:<10.3f} {paper_mae_str:<10}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        summary_path = os.path.join(save_dir, 'seed_sweep_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {summary_path}")

    return summary


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, save_path):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(model, train_loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    n_batches = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)

            total_loss += loss.item()
            n_batches += 1

            preds_list.append(pred.cpu().numpy())
            targets_list.append(batch_y.cpu().numpy())

    avg_loss = total_loss / n_batches
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    metrics = compute_metrics(preds, targets)

    return avg_loss, metrics, preds, targets


def build_model(model_name, enc_in, seq_len, pred_len, config):
    """Build model by name with given configuration."""
    if model_name == 'PatchTST':
        model = PatchTST(
            enc_in=enc_in,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_len=config.get('patch_len', 16),
            stride=config.get('stride', 8),
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 16),
            e_layers=config.get('e_layers', 3),
            d_ff=config.get('d_ff', 256),
            dropout=config.get('dropout', 0.2),
            head_dropout=config.get('head_dropout', 0.0),
            use_revin=config.get('use_revin', True),
        )
    elif model_name == 'PatchTST_CI_Only':
        # Ablation: channel-independence only, no patching (point-wise tokens)
        model = PatchTST_CI_Only(
            enc_in=enc_in, seq_len=seq_len, pred_len=pred_len,
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 16),
            e_layers=config.get('e_layers', 3),
            d_ff=config.get('d_ff', 256),
            dropout=config.get('dropout', 0.2),
            use_revin=config.get('use_revin', True),
        )
    elif model_name == 'PatchTST_P_Only':
        # Ablation: patching only, no channel-independence (channel-mixing)
        model = PatchTST_P_Only(
            enc_in=enc_in, seq_len=seq_len, pred_len=pred_len,
            patch_len=config.get('patch_len', 16),
            stride=config.get('stride', 8),
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 16),
            e_layers=config.get('e_layers', 3),
            d_ff=config.get('d_ff', 256),
            dropout=config.get('dropout', 0.2),
            use_revin=config.get('use_revin', True),
        )
    elif model_name == 'DLinear':
        model = DLinear(
            enc_in=enc_in,
            seq_len=seq_len,
            pred_len=pred_len,
            kernel_size=config.get('kernel_size', 25),
        )
    elif model_name == 'Transformer':
        model = VanillaTransformer(
            enc_in=enc_in,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 8),
            e_layers=config.get('e_layers', 2),
            d_ff=config.get('d_ff', 256),
            dropout=config.get('dropout', 0.2),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def run_experiment(
    model_name='PatchTST',
    dataset='ETTh1',
    data_path='./data',
    seq_len=336,
    pred_len=96,
    batch_size=32,
    epochs=100,
    lr=1e-4,
    patience=10,
    save_dir='./results',
    config=None,
    save_artifacts='minimal',
    seed=None,
    experiment_tag=None,
    deterministic=True,
):
    """
    Run a full training + evaluation experiment.

    Args:
        save_artifacts: how much to save to disk
            'minimal' — JSON results + history + scaler only (~3 KB total)
                        Predictions can be regenerated later via regenerate_predictions()
            'standard' — minimal + best checkpoint (.pt)
            'full' — standard + compressed test predictions (.npz)
                     Use only for experiments you'll visualize directly
            'none' — nothing saved (only returns results dict in memory)

    Returns:
        dict with test metrics and training history
    """
    if config is None:
        config = {}

    if seed is not None:
        set_seed(seed, deterministic=deterministic)
        print(f"Random seed fixed to {seed} (deterministic={deterministic})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- Load Data ----
    print(f"\nLoading {dataset}...")
    channel_names = None
    if dataset.startswith('ETT'):
        train_data, val_data, test_data = load_ett_data(data_path, dataset)
    elif dataset == 'Weather':
        train_data, val_data, test_data = load_weather_data(data_path)
    elif dataset.startswith('StoreDemand'):
        # StoreDemand_bystore, StoreDemand_byitem, StoreDemand_storeitems
        parts = dataset.split('_')
        mode = parts[1] if len(parts) > 1 else 'by_store'
        mode_map = {'bystore': 'by_store', 'byitem': 'by_item', 'storeitems': 'store_items'}
        mode = mode_map.get(mode, mode)
        train_data, val_data, test_data, channel_names = load_store_demand_data(
            data_path, mode=mode,
            store_id=config.get('store_id'),
            n_stores=config.get('n_stores', 10),
            n_items=config.get('n_items', 10),
        )
    elif dataset == 'Favorita':
        # Favorita Grocery Sales: largest Quito store, 33 product families
        train_data, val_data, test_data, channel_names, _ = load_favorita_data(
            data_path, city=config.get('city', 'Quito'), verbose=True,
        )
    else:
        train_data, val_data, test_data = load_custom_csv(
            data_path, dataset,
            date_col=config.get('date_col'),
            target_cols=config.get('target_cols'),
        )

    enc_in = train_data.shape[1]
    print(f"Features: {enc_in}, Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_len, pred_len=pred_len,
        batch_size=batch_size,
    )

    # ---- Build Model ----
    # Use smaller params for small datasets (ETTh1, ETTh2, Favorita) as in paper
    if dataset in ['ETTh1', 'ETTh2', 'Favorita'] and model_name == 'PatchTST':
        config.setdefault('d_model', 16)
        config.setdefault('n_heads', 4)
        config.setdefault('d_ff', 128)

    model = build_model(model_name, enc_in, seq_len, pred_len, config)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model_name}, Parameters: {n_params:,}")

    # ---- Training Setup ----
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # OneCycleLR scheduler (as used in the paper)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.3,
    )

    early_stopping = EarlyStopping(patience=patience)
    os.makedirs(save_dir, exist_ok=True)

    # File naming convention — all artifacts share the same base name
    base_name = _make_base_name(model_name, dataset, pred_len, experiment_tag=experiment_tag)
    ckpt_path = os.path.join(save_dir, f'{base_name}_best.pt')

    # ---- Training Loop ----
    history = {
        'train_loss': [], 'val_loss': [], 'val_mse': [], 'val_mae': [],
        'epoch_time': [], 'lr': [],
    }
    print(f"\nTraining for up to {epochs} epochs (patience={patience})...")
    print("-" * 60)

    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_mse'].append(float(val_metrics['mse']))
        history['val_mae'].append(float(val_metrics['mae']))
        history['epoch_time'].append(float(elapsed))
        history['lr'].append(float(current_lr))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val MSE: {val_metrics['mse']:.4f} | "
                  f"Val MAE: {val_metrics['mae']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        early_stopping(val_loss, model, ckpt_path)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    total_train_time = time.time() - train_start_time

    # ---- Final Evaluation on best checkpoint ----
    print("\n" + "=" * 60)
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # Always evaluate on test
    test_loss, test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device
    )
    # Quick val MSE/MAE — without storing predictions
    val_loss_final, val_metrics_final, _, _ = evaluate(
        model, val_loader, criterion, device
    )

    print(f"Val  MSE: {val_metrics_final['mse']:.4f}, MAE: {val_metrics_final['mae']:.4f}")
    print(f"Test MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}")

    best_epoch = epoch - early_stopping.counter

    # ---- Build results dict ----
    results = {
        'model': model_name,
        'dataset': dataset,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'enc_in': enc_in,
        # Test metrics (final reporting)
        'test_mse': float(test_metrics['mse']),
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        # Validation metrics at best checkpoint
        'val_mse': float(val_metrics_final['mse']),
        'val_mae': float(val_metrics_final['mae']),
        'val_rmse': float(val_metrics_final['rmse']),
        # Training info
        'n_params': int(n_params),
        'best_epoch': int(best_epoch),
        'total_epochs_run': int(epoch),
        'total_train_time_sec': float(total_train_time),
        'avg_epoch_time_sec': float(np.mean(history['epoch_time'])),
        # Hyperparameters
        'config': dict(config),
        'batch_size': batch_size,
        'lr': lr,
        'patience': patience,
        'seed': int(seed) if seed is not None else None,
        'experiment_tag': experiment_tag,
        'optimizer': 'Adam',
        'scheduler': 'OneCycleLR',
        'loss': 'MSE',
        # Data split info
        'train_size': int(len(train_data)),
        'val_size': int(len(val_data)),
        'test_size': int(len(test_data)),
        # Channel names if available
        'channel_names': channel_names,
        # Save level used
        'save_artifacts': save_artifacts,
    }

    if save_artifacts == 'none':
        return results, history

    # ---- Save tiered artifacts ----
    # Tier 1 (always): metrics + history + scaler  [~3 KB total]
    results_path = os.path.join(save_dir, f'{base_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    history_path = os.path.join(save_dir, f'{base_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Scaler is small (~600 bytes) and needed to denormalize when regenerating preds
    np.savez(
        os.path.join(save_dir, f'{base_name}_scaler.npz'),
        mean=scaler.mean, std=scaler.std,
    )

    print(f"\nMetrics saved to {results_path}")

    # Tier 2 (standard, full): keep checkpoint
    if save_artifacts in ('standard', 'full'):
        # Checkpoint already saved by EarlyStopping during training
        ckpt_size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"Checkpoint saved ({ckpt_size_mb:.1f} MB)")
    else:
        # Remove checkpoint to save space
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    # Tier 3 (full only): compressed test predictions
    if save_artifacts == 'full':
        preds_path = os.path.join(save_dir, f'{base_name}_preds.npz')
        np.savez_compressed(
            preds_path,
            preds=test_preds.astype(np.float32),
            targets=test_targets.astype(np.float32),
        )
        preds_size_mb = os.path.getsize(preds_path) / 1e6
        print(f"Predictions saved ({preds_size_mb:.1f} MB compressed)")

    return results, history


def run_full_benchmark(data_path='./data', save_dir='./results',
                        save_artifacts='standard',
                        transformer_save='minimal'):
    """
    Step 1: Main Benchmark (Table 3 Replication) — Plan v2

    Goal: Reproduce PatchTST's supervised forecasting results and verify
    its advantage over DLinear and vanilla Transformer.

    Configuration:
    - ETTh1  (M=7, hourly):  PatchTST, DLinear, Transformer x 4 horizons = 12
    - ETTm1  (M=7, 15-min):  PatchTST, DLinear, Transformer x 4 horizons = 12
    - Weather (M=21, 10-min): PatchTST, DLinear, Transformer x 4 horizons = 12
                              (Transformer uses seq_len=96 to keep its huge
                               flatten-head fitting on T4.)

    Total: 36 experiments, ~3-4 hours on T4

    PatchTST/42: L=336, P=16, S=8, N=42
    - For ETT (small): D=16, H=4, d_ff=128 (paper Appendix A.1.4)
    - For Weather:     D=128, H=16, d_ff=256

    Storage modes:
    - save_artifacts='standard' (default): saves checkpoints for PatchTST/DLinear
        so you can regenerate predictions later for any plot.
        Total Step 1 storage: ~200 MB (mostly PatchTST Weather)
    - transformer_save='minimal' (default): Transformer checkpoints are huge
        (~250 MB at pred_len=720 because of the flatten head) so we skip them.
        Transformer is just a baseline for the comparison table — its metrics
        are saved but checkpoint is not.
    - Set save_artifacts='full' to additionally save test predictions for ALL
        experiments (eats ~10 GB, not recommended for 15 GB Drive).
    - Set save_artifacts='minimal' to save NO checkpoints (smallest footprint
        ~100 KB total, but no way to regenerate predictions later).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plan v2: skip ETTh2/ETTm2 (marginal value, same structure as h1/m1)
    # All 3 datasets get all 3 models (Transformer fits on Weather via shorter L=96)
    experiment_grid = []
    for dataset in ['ETTh1', 'ETTm1', 'Weather']:
        for pred_len in [96, 192, 336, 720]:
            for model_name in ['PatchTST', 'DLinear', 'Transformer']:
                experiment_grid.append((dataset, pred_len, model_name))

    # Resume: load existing results if any
    consolidated_path = os.path.join(save_dir, 'benchmark_results.json')
    all_results = []
    if os.path.exists(consolidated_path):
        try:
            with open(consolidated_path) as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    total = len(experiment_grid)
    n_skipped = 0

    for done, (dataset, pred_len, model_name) in enumerate(experiment_grid, 1):
        # Skip if already completed (resume mode)
        if _is_experiment_done(model_name, dataset, pred_len, save_dir):
            n_skipped += 1
            print(f"\n[{done}/{total}] {model_name} | {dataset} | pred_len={pred_len} -- SKIPPED (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"  [{done}/{total}] {model_name} | {dataset} | pred_len={pred_len}")
        if n_skipped > 0:
            print(f"  (resumed; skipped {n_skipped} completed experiments)")
        print(f"{'='*60}")

        # Seq_len: 336 for PatchTST/DLinear, 96 for Transformer (memory)
        seq_len = 336 if model_name != 'Transformer' else 96

        config = {}
        if model_name == 'PatchTST':
            config = {'patch_len': 16, 'stride': 8}

        # Per-model storage override: Transformer checkpoints are huge
        # (huge flatten head). Use transformer_save for it.
        per_model_save = transformer_save if model_name == 'Transformer' else save_artifacts

        try:
            results, _ = run_experiment(
                model_name=model_name,
                dataset=dataset,
                data_path=data_path,
                seq_len=seq_len,
                pred_len=pred_len,
                batch_size=32,
                epochs=100,
                lr=1e-4,
                patience=10,
                save_dir=save_dir,
                config=config,
                save_artifacts=per_model_save,
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'model': model_name, 'dataset': dataset,
                'pred_len': pred_len, 'error': str(e)
            })

        # Save after each experiment (resume-friendly)
        with open(os.path.join(save_dir, 'benchmark_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n\n" + "=" * 80)
    print("STEP 1: MAIN BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<15} {'Dataset':<10} {'Pred Len':<10} {'MSE':<10} {'MAE':<10}")
    print("-" * 55)
    for r in all_results:
        if 'error' not in r:
            print(f"{r['model']:<15} {r['dataset']:<10} {r['pred_len']:<10} "
                  f"{r['test_mse']:<10.4f} {r['test_mae']:<10.4f}")

    return all_results


def run_ablation_study(data_path='./data', save_dir='./results/ablation',
                        save_artifacts='standard'):
    """
    Step 2a: Patching x Channel-Independence Ablation (Table 7 style) — Plan v2

    Goal: Isolate the contribution of patching and channel-independence
    by testing all four combinations.

    Variants:
    - P+CI (full PatchTST):    patching + channel-independence
    - CI Only:                  point-wise tokens (no patching) + channel-independence
    - P Only:                   patching + channel-mixing (no CI)
    - Original:                 no patching + channel-mixing = vanilla Transformer

    Configuration per plan v2:
    - ETTh1  (M=7):  all 4 variants x 2 horizons (96, 336)               = 8 experiments
    - Weather (M=21): only P+CI, CI Only x 2 horizons (96, 336)          = 4 experiments
                      (P Only / Original use channel-mixing on M=21,
                       which OOMs as the paper itself reports)

    Total: 12 experiments, ~45 min on T4

    The key comparison on Weather (does patching help on top of CI?) is
    still testable with just P+CI vs CI Only.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Variant -> (model_name, seq_len, base_config)
    variant_configs = {
        'P+CI':     ('PatchTST',         336, {'patch_len': 16, 'stride': 8}),
        'CI Only':  ('PatchTST_CI_Only',  96, {}),
        'P Only':   ('PatchTST_P_Only',  336, {'patch_len': 16, 'stride': 8}),
        'Original': ('Transformer',       96, {}),
    }

    # Plan v2 + Option C/D addendum:
    # Full paper Table 7 replication on the 2 datasets we test:
    # - ETTh1 (M=7):    all 4 variants × 2 horizons = 8 experiments
    # - Weather (M=21): all 4 variants × 2 horizons = 8 experiments
    # Total: 16 experiments
    experiment_grid = []
    for pred_len in [96, 336]:
        for label in ['P+CI', 'CI Only', 'P Only', 'Original']:
            experiment_grid.append(('ETTh1', pred_len, label))
    for pred_len in [96, 336]:
        for label in ['P+CI', 'CI Only', 'P Only', 'Original']:
            experiment_grid.append(('Weather', pred_len, label))

    # Resume: load existing results if any
    consolidated_path = os.path.join(save_dir, 'ablation_results.json')
    all_results = []
    if os.path.exists(consolidated_path):
        try:
            with open(consolidated_path) as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    total = len(experiment_grid)
    n_skipped = 0

    for done, (dataset, pred_len, label) in enumerate(experiment_grid, 1):
        model_name, seq_len, base_cfg = variant_configs[label]

        # Skip if already completed (resume mode)
        if _is_experiment_done(model_name, dataset, pred_len, save_dir):
            n_skipped += 1
            print(f"\n[{done}/{total}] ABLATION: {label} | {dataset} | pred_len={pred_len} -- SKIPPED")
            continue

        print(f"\n{'='*60}")
        print(f"  [{done}/{total}] ABLATION: {label} | {dataset} | pred_len={pred_len}")
        if n_skipped > 0:
            print(f"  (resumed; skipped {n_skipped} completed experiments)")
        print(f"{'='*60}")

        # Use smaller params for ETTh1 as in paper Appendix A.1.4
        cfg = dict(base_cfg)
        if dataset == 'ETTh1' and model_name.startswith('PatchTST'):
            cfg.setdefault('d_model', 16)
            cfg.setdefault('n_heads', 4)
            cfg.setdefault('d_ff', 128)

        try:
            results, _ = run_experiment(
                model_name=model_name,
                dataset=dataset,
                data_path=data_path,
                seq_len=seq_len,
                pred_len=pred_len,
                batch_size=32,
                epochs=50,
                lr=1e-4,
                patience=10,
                save_dir=save_dir,
                config=cfg,
                save_artifacts=save_artifacts,
            )
            results['ablation_label'] = label
            all_results.append(results)
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'model': model_name, 'dataset': dataset,
                'pred_len': pred_len, 'ablation_label': label,
                'error': str(e)
            })

        # Save after each experiment (resume-friendly)
        with open(os.path.join(save_dir, 'ablation_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    # Print summary table (like paper Table 7)
    print("\n\n" + "=" * 75)
    print("STEP 2a: ABLATION STUDY — Patching (P) and Channel-Independence (CI)")
    print("=" * 75)
    print(f"{'Dataset':<10} {'Pred':<6} {'P+CI (Full)':<14} {'CI Only':<14} {'P Only':<14} {'Original':<14}")
    print("-" * 72)

    for dataset in ['ETTh1', 'Weather']:
        for pred_len in [96, 336]:
            row = f"{dataset:<10} {pred_len:<6}"
            for label in ['P+CI', 'CI Only', 'P Only', 'Original']:
                match = [r for r in all_results
                         if r.get('ablation_label') == label
                         and r.get('dataset') == dataset
                         and r.get('pred_len') == pred_len
                         and 'error' not in r]
                if match:
                    row += f" {match[0]['test_mse']:.4f}/{match[0]['test_mae']:.4f}"
                else:
                    row += " missing       "
            print(row)

    return all_results


def run_step3_favorita(data_path='./data', save_dir='./results/favorita',
                        save_artifacts='standard'):
    """
    Step 3: Marketing Extension — Favorita Grocery Sales.

    Applies PatchTST and DLinear to retail demand forecasting on the Favorita
    Kaggle dataset. Single-store deep dive on the largest Quito store with all
    33 product families as channels.

    Configuration (deviates from step3_experiment_plan.md to fit the data):
      - seq_len = 200 (the plan said 336, but Favorita has only ~1684 days;
                       with 60/20/20 split, val = ~337 days, so seq_len must
                       be ≤ ~306 to fit any val sliding window)
      - 60/20/20 split (instead of 70/10/20 — short dataset needs bigger val/test)

    Step 3b: Model comparison
      - PatchTST (small config: D=16, H=4, 3 layers, P=16, S=8) on horizons 7, 14, 30 = 3 exp
      - DLinear (shared)                                          on horizons 7, 14, 30 = 3 exp
      - Transformer (vanilla, channel-mixing)                     on horizons 7, 14, 30 = 3 exp
      Total: 9 experiments, ~50-60 min on T4

    After this completes, run plot_step3_per_category_mse() and
    visualize_attention_for_step3() for the deep analysis figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = []

    # Plan: 3 models x 3 horizons = 9 experiments
    experiment_grid = []
    for pred_len in [7, 14, 30]:
        for model_name in ['PatchTST', 'DLinear', 'Transformer']:
            experiment_grid.append((pred_len, model_name))

    # Resume support
    consolidated_path = os.path.join(save_dir, 'favorita_results.json')
    if os.path.exists(consolidated_path):
        try:
            with open(consolidated_path) as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    total = len(experiment_grid)
    n_skipped = 0

    for done, (pred_len, model_name) in enumerate(experiment_grid, 1):
        # Skip if already completed
        if _is_experiment_done(model_name, 'Favorita', pred_len, save_dir):
            n_skipped += 1
            print(f"\n[{done}/{total}] {model_name} | Favorita | pred_len={pred_len} -- SKIPPED")
            continue

        print(f"\n{'='*60}")
        print(f"  [{done}/{total}] {model_name} | Favorita | pred_len={pred_len}")
        if n_skipped > 0:
            print(f"  (resumed; skipped {n_skipped} completed experiments)")
        print(f"{'='*60}")

        # PatchTST/DLinear: L=200 → num_patches=25 for PatchTST
        # Transformer: L=96 (shorter — vanilla Transformer head explodes with long L)
        # (seq_len constrained by Favorita's short dataset — see docstring)
        seq_len = 96 if model_name == 'Transformer' else 200
        config = {}
        if model_name == 'PatchTST':
            config = {'patch_len': 16, 'stride': 8}

        try:
            results, _ = run_experiment(
                model_name=model_name,
                dataset='Favorita',
                data_path=data_path,
                seq_len=seq_len,
                pred_len=pred_len,
                batch_size=32,
                epochs=50,
                lr=1e-4,
                patience=10,
                save_dir=save_dir,
                config=config,
                save_artifacts=save_artifacts,
            )
            all_results.append(results)
        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                'model': model_name, 'dataset': 'Favorita',
                'pred_len': pred_len, 'error': str(e)
            })

        # Save consolidated after each experiment
        with open(consolidated_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # Summary table
    print("\n\n" + "=" * 90)
    print("STEP 3: FAVORITA RESULTS SUMMARY (Quito largest store, 33 families)")
    print("=" * 90)
    print(f"{'Pred (days)':<10} "
          f"{'PatchTST MSE':<13} {'PatchTST MAE':<13} "
          f"{'DLinear MSE':<13} {'DLinear MAE':<13} "
          f"{'Transf MSE':<13} {'Transf MAE':<13}")
    print("-" * 90)
    for pred_len in [7, 14, 30]:
        row = f"{pred_len:<10}"
        for m in ['PatchTST', 'DLinear', 'Transformer']:
            match = [r for r in all_results
                     if r.get('model') == m and r.get('pred_len') == pred_len
                     and 'error' not in r]
            if match:
                row += f" {match[0]['test_mse']:<13.4f} {match[0]['test_mae']:<13.4f}"
            else:
                row += f" {'—':<13} {'—':<13}"
        print(row)

    return all_results


def run_marketing_benchmark(data_path='./data', save_dir='./results/marketing'):
    """
    [Deprecated placeholder] Old Step 3 stub using Kaggle Store Item Demand.
    Use run_step3_favorita() instead per step3_experiment_plan.md.

    Kept here only for backward compatibility with old notebook cells.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = []

    # ----------------------------------------------------------------
    # Experiment 1: Model comparison on StoreDemand (by_store, 10 channels)
    # Prediction horizons: 7 (1 week), 14 (2 weeks), 30 (1 month), 90 (1 quarter)
    # These horizons are meaningful for marketing planning
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Model Comparison on Store-Level Demand")
    print("=" * 70)

    models = ['PatchTST', 'DLinear', 'Transformer']
    pred_lens = [7, 14, 30, 90]

    for pred_len in pred_lens:
        for model_name in models:
            print(f"\n--- {model_name} | StoreDemand_bystore | pred_len={pred_len} ---")
            # seq_len=180 (~6 months look-back, fits daily retail data well)
            seq_len = 180 if model_name != 'Transformer' else 90
            config = {'patch_len': 16, 'stride': 8} if model_name == 'PatchTST' else {}
            try:
                results, _ = run_experiment(
                    model_name=model_name,
                    dataset='StoreDemand_bystore',
                    data_path=data_path,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    batch_size=32,
                    epochs=50,
                    lr=1e-3,
                    patience=10,
                    save_dir=save_dir,
                    config=config,
                )
                results['experiment'] = 'model_comparison'
                all_results.append(results)
            except Exception as e:
                print(f"ERROR: {e}")

    # ----------------------------------------------------------------
    # Experiment 2: Patch length ablation on StoreDemand
    # P=7 aligns with weekly cycle, P=14 biweekly, P=16 paper default
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Patch Length Ablation (Weekly Alignment)")
    print("=" * 70)

    patch_configs = [
        {'patch_len': 7,  'stride': 7,  'label': 'P7_weekly'},
        {'patch_len': 14, 'stride': 7,  'label': 'P14_biweekly'},
        {'patch_len': 16, 'stride': 8,  'label': 'P16_default'},
    ]

    for pc in patch_configs:
        for pred_len in [14, 30]:
            print(f"\n--- PatchTST ({pc['label']}) | pred_len={pred_len} ---")
            config = {'patch_len': pc['patch_len'], 'stride': pc['stride']}
            try:
                results, _ = run_experiment(
                    model_name='PatchTST',
                    dataset='StoreDemand_bystore',
                    data_path=data_path,
                    seq_len=180,
                    pred_len=pred_len,
                    batch_size=32,
                    epochs=50,
                    lr=1e-3,
                    patience=10,
                    save_dir=save_dir,
                    config=config,
                )
                results['experiment'] = 'patch_ablation'
                results['patch_label'] = pc['label']
                all_results.append(results)
            except Exception as e:
                print(f"ERROR: {e}")

    # ----------------------------------------------------------------
    # Experiment 3: Channel structure comparison
    # by_store (10 channels) vs store_items (10 items from 1 store)
    # Tests whether channel-independence helps with heterogeneous vs homogeneous channels
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Channel Structure (Store-level vs Item-level)")
    print("=" * 70)

    channel_modes = [
        ('StoreDemand_bystore', 'by_store (10 stores)'),
        ('StoreDemand_byitem',  'by_item (top 10 items)'),
        ('StoreDemand_storeitems', 'store_items (store 1, 10 items)'),
    ]

    for dataset_name, label in channel_modes:
        for pred_len in [14, 30]:
            print(f"\n--- PatchTST | {label} | pred_len={pred_len} ---")
            config = {'patch_len': 16, 'stride': 8, 'store_id': 1, 'n_items': 10}
            try:
                results, _ = run_experiment(
                    model_name='PatchTST',
                    dataset=dataset_name,
                    data_path=data_path,
                    seq_len=180,
                    pred_len=pred_len,
                    batch_size=32,
                    epochs=50,
                    lr=1e-3,
                    patience=10,
                    save_dir=save_dir,
                    config=config,
                )
                results['experiment'] = 'channel_structure'
                results['channel_mode'] = label
                all_results.append(results)
            except Exception as e:
                print(f"ERROR: {e}")

    # Save all marketing results
    with open(os.path.join(save_dir, 'marketing_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n\n" + "=" * 80)
    print("MARKETING EXTENSION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<20} {'Model':<12} {'Dataset':<25} {'PredLen':<8} {'MSE':<10} {'MAE':<10}")
    print("-" * 85)
    for r in all_results:
        if 'error' not in r:
            exp = r.get('experiment', '?')
            print(f"{exp:<20} {r['model']:<12} {r['dataset']:<25} "
                  f"{r['pred_len']:<8} {r['test_mse']:<10.4f} {r['test_mae']:<10.4f}")

    return all_results


def _is_experiment_done(model_name, dataset, pred_len, save_dir, experiment_tag=None):
    """Check if an experiment's results.json already exists (used for resume)."""
    base = _make_base_name(model_name, dataset, pred_len, experiment_tag=experiment_tag)
    results_path = os.path.join(save_dir, f'{base}_results.json')
    if not os.path.exists(results_path):
        return False
    # Sanity check: file is non-empty and parses as JSON with test_mse
    try:
        with open(results_path) as f:
            data = json.load(f)
        return 'test_mse' in data
    except Exception:
        return False


def load_experiment(model_name, dataset, pred_len, save_dir='./results', experiment_tag=None):
    """
    Load saved artifacts from a completed experiment. Whichever files exist
    will be loaded; missing ones (e.g. predictions in 'minimal' mode) are skipped.

    Returns a dict with optional keys:
        - results: metrics + config from results.json
        - history: per-epoch training history
        - preds, targets: test predictions and ground truth (if save_artifacts='full')
        - scaler_mean, scaler_std: normalization statistics
        - checkpoint_path: path to model weights (if save_artifacts in ['standard','full'])

    To regenerate predictions for an experiment saved in minimal mode, call
    regenerate_predictions(...) after re-training the model briefly.
    """
    base = _make_base_name(model_name, dataset, pred_len, experiment_tag=experiment_tag)
    paths = {
        'results':    os.path.join(save_dir, f'{base}_results.json'),
        'history':    os.path.join(save_dir, f'{base}_history.json'),
        'preds_npz':  os.path.join(save_dir, f'{base}_preds.npz'),
        # Legacy uncompressed format
        'preds_old':    os.path.join(save_dir, f'{base}_preds.npy'),
        'targets_old':  os.path.join(save_dir, f'{base}_targets.npy'),
        'scaler':     os.path.join(save_dir, f'{base}_scaler.npz'),
        'checkpoint': os.path.join(save_dir, f'{base}_best.pt'),
    }

    out = {'checkpoint_path': paths['checkpoint']}

    if os.path.exists(paths['results']):
        with open(paths['results']) as f:
            out['results'] = json.load(f)
    if os.path.exists(paths['history']):
        with open(paths['history']) as f:
            out['history'] = json.load(f)

    # New compressed format
    if os.path.exists(paths['preds_npz']):
        npz = np.load(paths['preds_npz'])
        out['preds'] = npz['preds']
        out['targets'] = npz['targets']
    # Old uncompressed format (back-compat)
    elif os.path.exists(paths['preds_old']):
        out['preds'] = np.load(paths['preds_old'])
        if os.path.exists(paths['targets_old']):
            out['targets'] = np.load(paths['targets_old'])

    if os.path.exists(paths['scaler']):
        sc = np.load(paths['scaler'])
        out['scaler_mean'] = sc['mean']
        out['scaler_std']  = sc['std']

    return out


def regenerate_predictions(model_name, dataset, pred_len, data_path='./data',
                            save_dir='./results', batch_size=32, experiment_tag=None):
    """
    Re-load a saved checkpoint and produce test predictions, then save as compressed npz.

    Use this when you ran an experiment in 'minimal' or 'standard' mode and now
    need predictions for visualization. Requires checkpoint to exist.
    """
    base = _make_base_name(model_name, dataset, pred_len, experiment_tag=experiment_tag)
    ckpt_path = os.path.join(save_dir, f'{base}_best.pt')
    results_path = os.path.join(save_dir, f'{base}_results.json')

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Re-train with save_artifacts='standard' or 'full'.")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"No results JSON at {results_path}.")

    with open(results_path) as f:
        info = json.load(f)

    seq_len = info['seq_len']
    enc_in = info['enc_in']
    config = info['config']

    # Load data
    if dataset.startswith('ETT'):
        train_data, val_data, test_data = load_ett_data(data_path, dataset)
    elif dataset == 'Weather':
        train_data, val_data, test_data = load_weather_data(data_path)
    elif dataset.startswith('StoreDemand'):
        parts = dataset.split('_')
        mode = parts[1] if len(parts) > 1 else 'by_store'
        mode_map = {'bystore': 'by_store', 'byitem': 'by_item', 'storeitems': 'store_items'}
        mode = mode_map.get(mode, mode)
        train_data, val_data, test_data, _ = load_store_demand_data(
            data_path, mode=mode,
            store_id=config.get('store_id'),
            n_stores=config.get('n_stores', 10),
            n_items=config.get('n_items', 10),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    _, _, test_loader, _ = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=seq_len, pred_len=pred_len, batch_size=batch_size,
    )

    # Build model
    model = build_model(model_name, enc_in, seq_len, pred_len, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

    # Run inference
    _, test_metrics, test_preds, test_targets = evaluate(
        model, test_loader, nn.MSELoss(), device,
    )

    # Save compressed
    out_path = os.path.join(save_dir, f'{base}_preds.npz')
    np.savez_compressed(
        out_path,
        preds=test_preds.astype(np.float32),
        targets=test_targets.astype(np.float32),
    )
    print(f"Saved predictions to {out_path}")
    print(f"  Test MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")
    return test_preds, test_targets


def run_steps_1_2(data_path='./data', save_dir='./results',
                   step1_save='standard', step2_save='standard'):
    """
    Phase 1 (per plan v2): Steps 1 + 2a only.

    Storage modes (default chosen for 15 GB Drive limit):
    - step1_save='standard' — saves metrics + history + scaler + checkpoints
                              for PatchTST and DLinear (small ckpts). Transformer
                              checkpoints are skipped automatically (huge head).
                              Total Step 1 storage: ~200 MB.
                              You can regenerate predictions for ANY of these later.
    - step2_save='standard' — Step 2a keeps checkpoints (needed for attention viz)
                              No predictions saved. Total Step 2 storage: ~50 MB.
    - Use 'full' to also save predictions (heavy — only for the few experiments
      you want to plot directly without regenerating).
    - Use 'minimal' if Drive is very tight (predictions cannot be regenerated
      later without re-training).

    After this completes, review results and decide on Step 3 (Marketing
    Extension, TBD) and Step 2b (Attention Visualization, run separately
    via visualize_attention_for_step2b).

    Step 1: Main Benchmark (Table 3)                    ~2-3 hours
    Step 2a: Ablation Study (Table 7 style)             ~45 min
    Phase 1 grand total: 44 experiments, ~3-4 hours on Colab T4
    """
    import datetime
    start = datetime.datetime.now()
    print("=" * 70)
    print(f"  STARTING PHASE 1 (Steps 1+2) at {start.strftime('%H:%M:%S')}")
    print("=" * 70)

    # --- Step 1: Main Benchmark ---
    print("\n\n" + "#" * 70)
    print(f"#  STEP 1/2: Main Benchmark (save_artifacts={step1_save})")
    print("#  ETTh1+ETTm1+Weather: 32 experiments")
    print("#" * 70)
    t1 = time.time()
    run_full_benchmark(data_path, os.path.join(save_dir, 'benchmark'),
                       save_artifacts=step1_save)
    print(f"\n  Step 1 done in {(time.time()-t1)/60:.1f} min")

    # --- Step 2: Ablation Study ---
    print("\n\n" + "#" * 70)
    print(f"#  STEP 2/2: Ablation Study (save_artifacts={step2_save})")
    print("#  ETTh1+Weather: 12 experiments")
    print("#" * 70)
    t2 = time.time()
    run_ablation_study(data_path, os.path.join(save_dir, 'ablation'),
                       save_artifacts=step2_save)
    print(f"\n  Step 2 done in {(time.time()-t2)/60:.1f} min")

    # --- Summary ---
    end = datetime.datetime.now()
    elapsed = (end - start).total_seconds() / 60
    print("\n\n" + "=" * 70)
    print(f"  PHASE 1 DONE! Total time: {elapsed:.1f} min")
    print(f"  Started: {start.strftime('%H:%M:%S')}, Ended: {end.strftime('%H:%M:%S')}")
    print(f"  Results in: {save_dir}/benchmark/ and {save_dir}/ablation/")
    print("=" * 70)
    print("\nReview results, then run Step 3 with:")
    print("  run_marketing_benchmark(data_path='../data', save_dir='../results/marketing')")
    print("Or with custom config (see run_marketing_benchmark docstring)")


def run_all(data_path='./data', save_dir='./results'):
    """
    Run the complete project pipeline end-to-end. RESUME-SAFE: re-running this
    function will skip experiments whose results JSON already exists, so it can
    be called from scratch on a fresh machine, or partway through to fill gaps.

    Pipeline:
      Step 1:  Main Benchmark (Table 3 replication)   ~3-5 hours
               PatchTST, DLinear, Transformer x ETTh1, ETTm1, Weather
               (Transformer skipped on Weather: M=21 OOM risk)
      Step 2a: Ablation (Table 7 replication)         ~3 hours
               P+CI, CI Only, P Only, Original x ETTh1, Weather x 2 horizons
      Step 3:  Favorita Marketing Extension           ~45 min
               PatchTST + DLinear x 3 horizons on largest Quito store

    Total fresh run: ~7-9 hours on Colab T4.

    Required data files in `data_path`:
      - ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv (auto-download via download_ett_data)
      - weather.csv (auto-download via download_weather_data + gdown fallback)
      - favorita_train.csv + favorita_stores.csv (manual or via Kaggle CLI)
    """
    import datetime
    start = datetime.datetime.now()
    print("=" * 70)
    print(f"  STARTING FULL PIPELINE at {start.strftime('%H:%M:%S')}")
    print("=" * 70)

    # Steps 1 + 2a
    run_steps_1_2(data_path, save_dir)

    # --- Step 3: Favorita Marketing Extension ---
    print("\n\n" + "#" * 70)
    print("#  STEP 3/3: Favorita Marketing Extension")
    print("#  PatchTST + DLinear x 3 horizons (7d, 14d, 30d) = 6 experiments")
    print("#" * 70)
    t3 = time.time()
    try:
        run_step3_favorita(data_path, os.path.join(save_dir, 'favorita'))
    except Exception as e:
        print(f"\n  Step 3 error: {e}")
        print("  (Make sure favorita_train.csv and favorita_stores.csv are in data/)")
        print("  Download from: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data")
    print(f"\n  Step 3 done in {(time.time()-t3)/60:.1f} min")

    # --- Summary ---
    end = datetime.datetime.now()
    elapsed = (end - start).total_seconds() / 60
    print("\n\n" + "=" * 70)
    print(f"  ALL DONE! Total time: {elapsed:.1f} min")
    print(f"  Started: {start.strftime('%H:%M:%S')}, Ended: {end.strftime('%H:%M:%S')}")
    print(f"  Results saved in: {save_dir}/")
    print("=" * 70)
    print("\nNext step: generate all poster figures by running cells 10/15/24/25")
    print("or via:  generate_all_poster_figures(...)")


def run_patchtst_seed_sweep(data_path='./data', save_dir='./results/seed_sweep',
                            seeds=(2021, 2022, 2023),
                            datasets=('ETTh1', 'ETTm1'),
                            pred_lens=(96, 192, 336, 720),
                            save_artifacts='standard'):
    """
    Resume-safe PatchTST seed sweep for ETTh1/ETTm1.

    Results are stored separately from the main benchmark so existing files are
    preserved and already-completed seed runs can be skipped.
    """
    os.makedirs(save_dir, exist_ok=True)
    consolidated_path = os.path.join(save_dir, 'seed_sweep_results.json')
    all_results = []
    if os.path.exists(consolidated_path):
        try:
            with open(consolidated_path) as f:
                all_results = json.load(f)
        except Exception:
            all_results = []

    experiment_grid = [
        ('PatchTST', dataset, pred_len, seed)
        for seed in seeds
        for dataset in datasets
        for pred_len in pred_lens
    ]

    total = len(experiment_grid)
    n_skipped = 0

    print("\n" + "=" * 76)
    print("PatchTST Seed Sweep")
    print(f"Datasets: {list(datasets)} | Horizons: {list(pred_lens)} | Seeds: {list(seeds)}")
    print(f"Save dir: {save_dir}")
    print("=" * 76)

    for done, (model_name, dataset, pred_len, seed) in enumerate(experiment_grid, 1):
        experiment_tag = f'seed{seed}'

        if _is_experiment_done(model_name, dataset, pred_len, save_dir, experiment_tag=experiment_tag):
            n_skipped += 1
            print(f"\n[{done}/{total}] {model_name} | {dataset} | T={pred_len} | {experiment_tag} -- SKIPPED")
            continue

        print(f"\n{'='*76}")
        print(f"  [{done}/{total}] {model_name} | {dataset} | pred_len={pred_len} | {experiment_tag}")
        if n_skipped > 0:
            print(f"  (resumed; skipped {n_skipped} completed runs)")
        print(f"{'='*76}")

        try:
            results, _ = run_experiment(
                model_name=model_name,
                dataset=dataset,
                data_path=data_path,
                seq_len=336,
                pred_len=pred_len,
                batch_size=32,
                epochs=100,
                lr=1e-4,
                patience=10,
                save_dir=save_dir,
                config={'patch_len': 16, 'stride': 8},
                save_artifacts=save_artifacts,
                seed=seed,
                experiment_tag=experiment_tag,
                deterministic=True,
            )
            results['experiment'] = f'{dataset}_T{pred_len}_{experiment_tag}'
            _upsert_result_record(all_results, results)
        except Exception as e:
            print(f"ERROR: {e}")
            _upsert_result_record(all_results, {
                'model': model_name,
                'dataset': dataset,
                'pred_len': pred_len,
                'seed': seed,
                'experiment_tag': experiment_tag,
                'error': str(e),
            })

        all_results = sorted(
            all_results,
            key=lambda r: (
                str(r.get('model')),
                str(r.get('dataset')),
                int(r.get('pred_len', -1)),
                int(r.get('seed', -1)) if r.get('seed') is not None else -1,
                str(r.get('experiment_tag')),
            ),
        )
        with open(consolidated_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    summary = summarize_seed_sweep(all_results, save_dir=save_dir)
    return all_results, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PatchTST Training')
    parser.add_argument('--model', type=str, default='PatchTST',
                        choices=['PatchTST', 'DLinear', 'Transformer'])
    parser.add_argument('--dataset', type=str, default='ETTh1')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed for a single experiment')
    parser.add_argument('--experiment_tag', type=str, default=None,
                        help='Optional suffix appended to saved artifact filenames')
    parser.add_argument('--all', action='store_true',
                        help='Run everything: benchmark + ablation + marketing')
    parser.add_argument('--phase1', action='store_true',
                        help='Run only Step 1 (benchmark) + Step 2 (ablation)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark on ETTh1, ETTm1, Weather, ETTh2, ETTm2')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study (P+CI vs CI vs P vs Original)')
    parser.add_argument('--marketing', action='store_true',
                        help='Run marketing extension on Store Demand data')
    parser.add_argument('--seed_sweep', action='store_true',
                        help='Run PatchTST on ETTh1/ETTm1 across multiple seeds with resume-safe skipping')
    parser.add_argument('--seed_summary', action='store_true',
                        help='Summarize existing seed-sweep results without training')
    parser.add_argument('--seeds', type=str, default='2021,2022,2023',
                        help='Comma-separated seeds for --seed_sweep')
    parser.add_argument('--sweep_datasets', type=str, default='ETTh1,ETTm1',
                        help='Comma-separated datasets for --seed_sweep')
    parser.add_argument('--sweep_pred_lens', type=str, default='96,192,336,720',
                        help='Comma-separated prediction lengths for --seed_sweep')
    parser.add_argument('--save_artifacts', type=str, default=None,
                        choices=['none', 'minimal', 'standard', 'full'],
                        help='Artifact save level for single runs or seed sweep')

    args = parser.parse_args()

    if args.all:
        run_all(args.data_path, args.save_dir)
    elif args.phase1:
        run_steps_1_2(args.data_path, args.save_dir)
    elif args.benchmark:
        run_full_benchmark(args.data_path, args.save_dir)
    elif args.ablation:
        run_ablation_study(args.data_path, args.save_dir)
    elif args.marketing:
        run_marketing_benchmark(args.data_path, args.save_dir)
    elif args.seed_sweep:
        run_patchtst_seed_sweep(
            data_path=args.data_path,
            save_dir=_resolve_seed_sweep_dir(args.save_dir),
            seeds=_parse_int_list(args.seeds),
            datasets=_parse_str_list(args.sweep_datasets),
            pred_lens=_parse_int_list(args.sweep_pred_lens),
            save_artifacts=args.save_artifacts or 'standard',
        )
    elif args.seed_summary:
        summary_dir = _resolve_seed_sweep_dir(args.save_dir)
        consolidated_path = os.path.join(summary_dir, 'seed_sweep_results.json')
        if not os.path.exists(consolidated_path):
            raise FileNotFoundError(f"No seed sweep results found at {consolidated_path}")
        with open(consolidated_path) as f:
            summarize_seed_sweep(json.load(f), save_dir=summary_dir)
    else:
        run_experiment(
            model_name=args.model,
            dataset=args.dataset,
            data_path=args.data_path,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            save_dir=args.save_dir,
            save_artifacts=args.save_artifacts or 'minimal',
            seed=args.seed,
            experiment_tag=args.experiment_tag,
        )
