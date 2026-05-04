"""
Data loading utilities for time series forecasting datasets.

Supports:
- ETTh1, ETTh2 (hourly Electricity Transformer Temperature)
- ETTm1, ETTm2 (15-minute ETT)
- Weather
- StoreDemand (Kaggle Store Item Demand Forecasting - marketing extension)
- Custom CSV datasets
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid division by zero

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


class TimeSeriesDataset(Dataset):
    """
    General time series dataset for forecasting.

    Creates sliding windows of (input, target) pairs:
    - input:  x[t : t + seq_len]
    - target: x[t + seq_len : t + seq_len + pred_len]
    """

    def __init__(self, data, seq_len, pred_len):
        """
        Args:
            data: numpy array of shape (timesteps, features)
            seq_len: look-back window length
            pred_len: prediction horizon
        """
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return x, y


def load_ett_data(data_path, dataset_name='ETTh1'):
    """
    Load ETT dataset from CSV.

    ETT datasets have columns: date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    - 7 features total
    - Split: 12/4/4 months for ETTh, 12/4/4 months for ETTm

    Returns:
        train_data, val_data, test_data: numpy arrays
    """
    df = pd.read_csv(os.path.join(data_path, f'{dataset_name}.csv'))

    # Remove date column
    data = df.iloc[:, 1:].values.astype(np.float32)

    # Standard splits for ETT datasets
    if 'ETTh' in dataset_name:
        # Hourly: 8640/2880/2880 (12/4/4 months)
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
    elif 'ETTm' in dataset_name:
        # 15-min: 34560/11520/11520 (12/4/4 months)
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
    else:
        # Default 70/10/20 split
        n = len(data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def load_weather_data(data_path):
    """
    Load Weather dataset from CSV.
    21 meteorological indicators, 52696 timesteps.
    Split: 70/10/20
    """
    df = pd.read_csv(os.path.join(data_path, 'weather.csv'))
    data = df.iloc[:, 1:].values.astype(np.float32)

    n = len(data)
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)

    return data[:train_end], data[train_end:val_end], data[val_end:]


# ============================================================
# Store Item Demand Forecasting (Marketing Extension)
# ============================================================

def download_store_demand_data(save_dir='./data'):
    """
    Download Store Item Demand Forecasting dataset from Kaggle.

    Requires: kaggle CLI configured (kaggle.json in ~/.kaggle/)
    Alternative: download manually from
    https://www.kaggle.com/c/demand-forecasting-kernels-only/data

    The dataset contains:
    - train.csv: date, store (1-10), item (1-50), sales
    - 5 years of daily sales (2013-01-01 to 2017-12-31)
    - 913,000 rows total
    """
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, 'store_demand_train.csv')

    if os.path.exists(train_path):
        print(f"  Store demand data already exists at {train_path}")
        return train_path

    print("Downloading Store Item Demand dataset from Kaggle...")
    print("If this fails, download manually from:")
    print("  https://www.kaggle.com/c/demand-forecasting-kernels-only/data")
    print(f"  and save train.csv as {train_path}")

    try:
        import subprocess
        subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', 'demand-forecasting-kernels-only',
            '-f', 'train.csv',
            '-p', save_dir
        ], check=True)
        # Rename to our convention
        downloaded = os.path.join(save_dir, 'train.csv')
        if os.path.exists(downloaded):
            os.rename(downloaded, train_path)
        print(f"  Saved to {train_path}")
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        print("  Please download manually and place as store_demand_train.csv in data/")

    return train_path


def load_store_demand_data(data_path, mode='by_store', store_id=None, item_id=None,
                           n_stores=10, n_items=10):
    """
    Load and pivot Store Item Demand data for multivariate time series forecasting.

    The raw data has columns: date, store, item, sales.
    We pivot it into a multivariate time series where each channel is a store or item.

    Args:
        data_path: directory containing store_demand_train.csv
        mode: how to create channels:
            'by_store'  - each channel = total daily sales for one store (10 channels)
            'by_item'   - each channel = total daily sales for one item across stores (up to 50)
            'store_items' - each channel = sales of one item in one specific store
        store_id: if mode='store_items', which store to use (1-10)
        item_id: if mode='by_store' or specific item analysis
        n_stores: number of stores to include (default 10 = all)
        n_items: number of items to include for 'by_item' mode (default 10)

    Returns:
        train_data, val_data, test_data: numpy arrays (timesteps, n_channels)
        channel_names: list of channel name strings
    """
    filepath = os.path.join(data_path, 'store_demand_train.csv')
    if not os.path.exists(filepath):
        # Try alternative name
        filepath = os.path.join(data_path, 'train.csv')

    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date')

    if mode == 'by_store':
        # Aggregate: daily total sales per store -> 10 channels
        pivot = df.groupby(['date', 'store'])['sales'].sum().reset_index()
        pivot = pivot.pivot(index='date', columns='store', values='sales')
        # Select subset of stores
        stores = sorted(pivot.columns)[:n_stores]
        pivot = pivot[stores]
        channel_names = [f'store_{s}' for s in stores]

    elif mode == 'by_item':
        # Aggregate: daily total sales per item (across all stores) -> n_items channels
        pivot = df.groupby(['date', 'item'])['sales'].sum().reset_index()
        pivot = pivot.pivot(index='date', columns='item', values='sales')
        # Select top n_items by total sales volume
        top_items = pivot.sum().nlargest(n_items).index
        pivot = pivot[top_items]
        channel_names = [f'item_{i}' for i in top_items]

    elif mode == 'store_items':
        # Single store, each item as a channel -> n_items channels
        if store_id is None:
            store_id = 1
        store_df = df[df['store'] == store_id]
        pivot = store_df.pivot(index='date', columns='item', values='sales')
        # Select top items
        top_items = pivot.sum().nlargest(n_items).index
        pivot = pivot[top_items]
        channel_names = [f's{store_id}_item_{i}' for i in top_items]

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'by_store', 'by_item', or 'store_items'")

    # Fill any NaN with 0 (shouldn't happen but safety)
    pivot = pivot.fillna(0)
    data = pivot.values.astype(np.float32)

    # Split: 3 years train, 1 year val, 1 year test
    # Total ~1826 days (5 years) -> ~1096 train, ~365 val, ~365 test
    # This gives enough room for seq_len up to ~200 in val/test
    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print(f"  Mode: {mode}")
    print(f"  Channels ({len(channel_names)}): {channel_names}")
    print(f"  Total days: {n}")
    print(f"  Split -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data, channel_names


# ============================================================
# Step 3: Favorita Grocery Sales (Marketing Extension)
# ============================================================

def download_favorita_data(save_dir='./data'):
    """
    Download Favorita Kaggle competition data.

    Files needed:
      - train.csv (~120 MB) — sales data
      - stores.csv (~2 KB) — store metadata (city, type)

    Try Kaggle CLI first; if not configured, print manual instructions.
    Saves both files into save_dir as favorita_train.csv and favorita_stores.csv.
    """
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, 'favorita_train.csv')
    stores_path = os.path.join(save_dir, 'favorita_stores.csv')

    if os.path.exists(train_path) and os.path.exists(stores_path):
        print(f"  Favorita data already exists")
        return train_path, stores_path

    print("Attempting to download Favorita data via Kaggle CLI...")
    try:
        import subprocess
        # Download the whole competition zip and extract train.csv + stores.csv
        subprocess.run([
            'kaggle', 'competitions', 'download',
            '-c', 'store-sales-time-series-forecasting',
            '-p', save_dir,
        ], check=True, capture_output=True, text=True)

        zip_path = os.path.join(save_dir, 'store-sales-time-series-forecasting.zip')
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extract('train.csv', save_dir)
                z.extract('stores.csv', save_dir)
            os.rename(os.path.join(save_dir, 'train.csv'), train_path)
            os.rename(os.path.join(save_dir, 'stores.csv'), stores_path)
            os.remove(zip_path)
            print(f"  Saved {train_path}")
            print(f"  Saved {stores_path}")
            return train_path, stores_path
    except Exception as e:
        print(f"  Kaggle CLI download failed: {e}")

    print("\nManual download instructions:")
    print("  1. Go to: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data")
    print("  2. Download train.csv and stores.csv")
    print(f"  3. Save them as: {train_path} and {stores_path}")
    return None, None


def load_favorita_data(data_path, city='Quito', train_ratio=0.6, val_ratio=0.2,
                       verbose=True):
    """
    Step 3a: Load Favorita data and pivot to PatchTST format.

    Strategy:
      1. Find the largest store in `city` (default Quito) by total sales volume
      2. Filter train data to that store
      3. Pivot to (date, family) → matrix of shape (~1684, 33)
      4. Apply 60/20/20 chronological split (more val/test than 70/10/20 since
         the dataset is short — needed to fit seq_len=200 sliding windows)

    Returns:
        train_data, val_data, test_data: numpy arrays (timesteps, 33)
        family_names: list of 33 product family names (the channels)
        store_info: dict with selected store's metadata (store_nbr, city, type, etc.)
    """
    train_path = os.path.join(data_path, 'favorita_train.csv')
    stores_path = os.path.join(data_path, 'favorita_stores.csv')

    if not (os.path.exists(train_path) and os.path.exists(stores_path)):
        raise FileNotFoundError(
            f"Favorita data not found at {train_path}. "
            f"Run download_favorita_data() first or download manually from Kaggle."
        )

    if verbose:
        print(f"  Loading Favorita stores metadata...")
    stores = pd.read_csv(stores_path)
    city_stores = stores[stores['city'] == city]
    if len(city_stores) == 0:
        raise ValueError(f"No stores found in city={city}. Available cities: "
                         f"{sorted(stores['city'].unique())}")

    if verbose:
        print(f"  Found {len(city_stores)} stores in {city}: "
              f"{sorted(city_stores['store_nbr'].tolist())}")

    if verbose:
        print(f"  Loading sales data (this may take ~30 sec)...")
    df = pd.read_csv(train_path, parse_dates=['date'],
                     dtype={'store_nbr': 'int32', 'sales': 'float32'},
                     usecols=['date', 'store_nbr', 'family', 'sales'])

    # Filter to city stores, then find largest by total sales volume
    df_city = df[df['store_nbr'].isin(city_stores['store_nbr'])]
    sales_by_store = df_city.groupby('store_nbr')['sales'].sum().sort_values(ascending=False)
    largest_store = int(sales_by_store.index[0])

    store_info = stores[stores['store_nbr'] == largest_store].iloc[0].to_dict()
    if verbose:
        print(f"  Selected largest store: store_nbr={largest_store} "
              f"({store_info.get('type','?')} type, cluster {store_info.get('cluster','?')})")
        print(f"  Total sales: {sales_by_store.iloc[0]:,.0f}")

    # Filter to that store and pivot
    df_store = df[df['store_nbr'] == largest_store].copy()
    pivot = df_store.pivot_table(
        index='date', columns='family', values='sales', fill_value=0,
    )
    pivot = pivot.sort_index()
    family_names = pivot.columns.tolist()

    data = pivot.values.astype(np.float32)
    if verbose:
        print(f"  Pivot shape: {data.shape} (days x product_families)")
        print(f"  Date range: {pivot.index.min().date()} to {pivot.index.max().date()}")
        print(f"  Families: {family_names[:5]}... (+{len(family_names)-5} more)")

    # Chronological split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    if verbose:
        print(f"  Split -> Train: {len(train_data)}, Val: {len(val_data)}, "
              f"Test: {len(test_data)}")

    return train_data, val_data, test_data, family_names, store_info


def load_custom_csv(data_path, filename, date_col=None, target_cols=None,
                    train_ratio=0.7, val_ratio=0.1):
    """
    Load a custom CSV dataset (e.g., marketing/sales data).

    Args:
        data_path: directory containing the CSV
        filename: CSV filename
        date_col: name of date column to exclude (optional)
        target_cols: list of column names to use (optional, uses all if None)
        train_ratio: fraction for training
        val_ratio: fraction for validation
    """
    df = pd.read_csv(os.path.join(data_path, filename))

    if date_col and date_col in df.columns:
        df = df.drop(columns=[date_col])

    if target_cols:
        df = df[target_cols]

    # Drop non-numeric columns
    df = df.select_dtypes(include=[np.number])
    data = df.values.astype(np.float32)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return data[:train_end], data[train_end:val_end], data[val_end:]


def create_dataloaders(train_data, val_data, test_data, seq_len, pred_len,
                       batch_size=32, num_workers=0):
    """
    Create train/val/test DataLoaders with StandardScaler normalization.

    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(train_data)

    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    train_dataset = TimeSeriesDataset(train_scaled, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_scaled, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_scaled, seq_len, pred_len)

    # Safety check: ensure datasets have positive length
    for name, ds in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        if len(ds) <= 0:
            raise ValueError(
                f"{name} dataset has {len(ds)} samples (need >0). "
                f"Data has {len(scaler.transform(train_data)) if name == 'train' else '?'} rows, "
                f"but seq_len={seq_len} + pred_len={pred_len} = {seq_len + pred_len} is too large. "
                f"Reduce seq_len or use more data."
            )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader, scaler


def download_ett_data(save_dir='./data'):
    """
    Download ETT datasets from the official source.
    Call this in Colab to get the data.
    """
    import urllib.request
    import ssl

    os.makedirs(save_dir, exist_ok=True)
    base_url = 'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/'

    datasets = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']

    # Handle SSL certificate issues (common on macOS)
    try:
        ssl_context = ssl.create_default_context()
    except Exception:
        ssl_context = ssl._create_unverified_context()

    for name in datasets:
        filepath = os.path.join(save_dir, name)
        if not os.path.exists(filepath):
            print(f"Downloading {name}...")
            try:
                urllib.request.urlretrieve(base_url + name, filepath)
            except Exception:
                # Fallback: disable SSL verification (safe for public GitHub raw files)
                ctx = ssl._create_unverified_context()
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ctx)
                )
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(base_url + name, filepath)
            print(f"  Saved to {filepath}")
        else:
            print(f"  {name} already exists")

    print("Done!")


def download_weather_data(save_dir='./data'):
    """
    Download Weather dataset (21 meteorological indicators, 52696 timesteps).

    The Weather dataset is from the Max Planck Institute for Biogeochemistry.
    It's commonly distributed via Google Drive in time series benchmarks.

    In Colab, run:
        !gdown 1r4h8qSWcy5jbhSbTEXBKbEYL6Vjt2q1Z -O ./data/weather.csv
    Or download from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, 'weather.csv')

    if os.path.exists(filepath):
        print(f"  weather.csv already exists")
        return

    # Try gdown (works well in Colab)
    try:
        import subprocess
        result = subprocess.run(
            ['gdown', '1r4h8qSWcy5jbhSbTEXBKbEYL6Vjt2q1Z', '-O', filepath],
            capture_output=True, text=True, timeout=60
        )
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
            print(f"  Downloaded weather.csv via gdown")
            return
    except Exception:
        pass

    print("  Auto-download failed. To get weather.csv:")
    print("  In Colab: !pip install gdown && !gdown 1r4h8qSWcy5jbhSbTEXBKbEYL6Vjt2q1Z -O ./data/weather.csv")
    print("  Or download from: https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy")


if __name__ == '__main__':
    # Quick test with synthetic data
    np.random.seed(42)
    data = np.random.randn(1000, 7).astype(np.float32)
    train_data = data[:700]
    val_data = data[700:800]
    test_data = data[800:]

    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        train_data, val_data, test_data,
        seq_len=96, pred_len=96, batch_size=32
    )

    for x, y in train_loader:
        print(f"Input batch:  {x.shape}")  # (32, 96, 7)
        print(f"Target batch: {y.shape}")  # (32, 96, 7)
        break
