import json
import math
import os
import sys
from html import escape

import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from data_loader import load_favorita_data, create_dataloaders
from patchtst import PatchTST


COLORS = {
    'PatchTST': '#1565C0',
    'DLinear': '#43A047',
    'Transformer': '#EF6C00',
}


def _load_clean_results(path):
    with open(path) as f:
        rows = json.load(f)
    return [r for r in rows if 'error' not in r]


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _fmt(num):
    return f'{num:.3f}'


def _hex_to_rgb(color):
    color = color.lstrip('#')
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(max(0, min(255, int(round(v)))) for v in rgb)


def _interp_color(stops, t):
    t = max(0.0, min(1.0, t))
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    seg = t * (len(stops) - 1)
    i = int(math.floor(seg))
    frac = seg - i
    a = _hex_to_rgb(stops[i])
    b = _hex_to_rgb(stops[i + 1])
    rgb = tuple(a[j] + (b[j] - a[j]) * frac for j in range(3))
    return _rgb_to_hex(rgb)


def _line_path(points):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)


def _scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def _add_text(parts, x, y, text, size=14, weight='normal', fill='#222', anchor='start',
              italic=False):
    style = 'font-style:italic;' if italic else ''
    parts.append(
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="{anchor}" style="{style}">{escape(text)}</text>'
    )


def _draw_marker(parts, x, y, color, marker):
    if marker == 'o':
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}" />')
    elif marker == 's':
        parts.append(f'<rect x="{x - 4.3:.2f}" y="{y - 4.3:.2f}" width="8.6" height="8.6" fill="{color}" />')
    else:
        pts = f'{x:.2f},{y - 5:.2f} {x - 5:.2f},{y + 4.5:.2f} {x + 5:.2f},{y + 4.5:.2f}'
        parts.append(f'<polygon points="{pts}" fill="{color}" />')


def _draw_line_panel(parts, x0, y0, w, h, title, series, x_ticks, x_labels, xlabel, ylabel,
                     legend=None):
    plot_left = x0 + 58
    plot_right = x0 + w - 16
    plot_top = y0 + 26
    plot_bottom = y0 + h - 40
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    all_y = [y for s in series for y in s['y']]
    y_min = min(all_y)
    y_max = max(all_y)
    y_pad = 0.12 * (y_max - y_min) if y_max > y_min else 0.1
    y0v = y_min - y_pad
    y1v = y_max + y_pad

    x_min = min(x_ticks)
    x_max = max(x_ticks)

    _add_text(parts, x0 + w / 2, y0 + 14, title, size=16, weight='bold', anchor='middle')

    for frac in np.linspace(0, 1, 5):
        yv = y0v + frac * (y1v - y0v)
        py = _scale(yv, y0v, y1v, plot_bottom, plot_top)
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{py:.2f}" x2="{plot_right:.2f}" y2="{py:.2f}" '
            f'stroke="#d6d6d6" stroke-width="1" opacity="0.7" />'
        )
        _add_text(parts, plot_left - 8, py + 4, f'{yv:.2f}', size=10, anchor='end', fill='#555')

    for xt, xl in zip(x_ticks, x_labels):
        px = _scale(xt, x_min, x_max, plot_left, plot_right)
        parts.append(
            f'<line x1="{px:.2f}" y1="{plot_top:.2f}" x2="{px:.2f}" y2="{plot_bottom:.2f}" '
            f'stroke="#ececec" stroke-width="1" />'
        )
        _add_text(parts, px, plot_bottom + 18, xl, size=10, anchor='middle', fill='#444')

    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" '
        f'stroke="#333" stroke-width="1.5" />'
    )
    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" y2="{plot_bottom:.2f}" '
        f'stroke="#333" stroke-width="1.5" />'
    )

    for s in series:
        pts = [
            (_scale(xv, x_min, x_max, plot_left, plot_right), _scale(yv, y0v, y1v, plot_bottom, plot_top))
            for xv, yv in zip(s['x'], s['y'])
        ]
        parts.append(
            f'<polyline fill="none" stroke="{s["color"]}" stroke-width="{s["width"]}" '
            f'points="{_line_path(pts)}" />'
        )
        label_dy = plot_h * 0.03
        for (px, py), yv in zip(pts, s['y']):
            _draw_marker(parts, px, py, s['color'], s['marker'])
            _add_text(parts, px, py - label_dy, _fmt(yv), size=10, weight='bold',
                      fill=s['color'], anchor='middle')

    _add_text(parts, x0 + w / 2, y0 + h - 8, xlabel, size=12, anchor='middle')
    parts.append(
        f'<text x="{x0 + 18:.2f}" y="{y0 + h / 2:.2f}" font-size="12" fill="#222" '
        f'text-anchor="middle" transform="rotate(-90 {x0 + 18:.2f},{y0 + h / 2:.2f})">{escape(ylabel)}</text>'
    )

    if legend:
        lx = x0 + w - 158 if legend == 'right' else x0 + 78
        ly = y0 + 30
        parts.append(
            f'<rect x="{lx - 10:.2f}" y="{ly - 16:.2f}" width="134" height="62" rx="6" '
            f'fill="white" fill-opacity="0.86" stroke="#cfcfcf" />'
        )
        for i, s in enumerate(series):
            yy = ly + i * 18
            parts.append(
                f'<line x1="{lx:.2f}" y1="{yy:.2f}" x2="{lx + 28:.2f}" y2="{yy:.2f}" '
                f'stroke="{s["color"]}" stroke-width="{s["width"]}" />'
            )
            _draw_marker(parts, lx + 14, yy, s['color'], s['marker'])
            _add_text(parts, lx + 36, yy + 4, s['label'], size=10, fill='#333')


def export_benchmark_trends_svg():
    rows = _load_clean_results('results/benchmark/benchmark_results.json')
    pred_lens = [96, 192, 336, 720]
    models = [('PatchTST', 'o', 2.8), ('DLinear', 's', 2.2), ('Transformer', '^', 2.2)]
    datasets = ['ETTm1', 'Weather']

    width, height = 1040, 330
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]
    _add_text(parts, width / 2, 24, 'Benchmark Trends: MSE vs Forecast Horizon',
              size=20, weight='bold', anchor='middle')

    panel_w, panel_h, top = 480, 260, 36
    x_positions = [24, 536]
    for ds, x0 in zip(datasets, x_positions):
        series = []
        for model, marker, lw in models:
            subset = sorted(
                [r for r in rows if r['dataset'] == ds and r['model'] == model and r['pred_len'] in pred_lens],
                key=lambda r: r['pred_len'],
            )
            series.append({
                'label': model,
                'x': [r['pred_len'] for r in subset],
                'y': [r['test_mse'] for r in subset],
                'color': COLORS[model],
                'marker': marker,
                'width': lw,
            })
        _draw_line_panel(parts, x0, top, panel_w, panel_h, ds, series, pred_lens,
                         [str(v) for v in pred_lens], 'Prediction Horizon', 'MSE',
                         legend='left' if ds == 'ETTm1' else None)

    parts.append('</svg>')
    out = 'results/benchmark/benchmark_trends.svg'
    _ensure_dir(out)
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print(f'Saved: {out}')


def export_favorita_trends_svg():
    rows = _load_clean_results('results/favorita/favorita_results.json')
    pred_lens = [7, 14, 30]
    models = [('PatchTST', 'o', 2.8), ('DLinear', 's', 2.2), ('Transformer', '^', 2.2)]
    series = []
    for model, marker, lw in models:
        subset = sorted([r for r in rows if r['model'] == model and r['pred_len'] in pred_lens],
                        key=lambda r: r['pred_len'])
        series.append({
            'label': model,
            'x': [r['pred_len'] for r in subset],
            'y': [r['test_mse'] for r in subset],
            'color': COLORS[model],
            'marker': marker,
            'width': lw,
        })

    width, height = 720, 420
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]
    _draw_line_panel(parts, 18, 24, 684, 360, 'Favorita', series, pred_lens,
                     [f'{v}d' for v in pred_lens], 'Prediction Horizon (days)', 'MSE',
                     legend='right')
    parts.append('</svg>')
    out = 'results/favorita/favorita_trends.svg'
    _ensure_dir(out)
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print(f'Saved: {out}')


def export_ablation_weather_336_svg():
    rows = _load_clean_results('results/ablation/ablation_results.json')
    rows = [r for r in rows if r['dataset'] == 'Weather' and r['pred_len'] == 336]
    label_to_pos = {'Original': (0, 0), 'P Only': (0, 1), 'CI Only': (1, 0), 'P+CI': (1, 1)}
    mat = np.full((2, 2), np.nan)
    for r in rows:
        rr, cc = label_to_pos[r['ablation_label']]
        mat[rr, cc] = r['test_mse']

    vmin = float(np.nanmin(mat))
    vmax = float(np.nanmax(mat))
    best = float(np.nanmin(mat))
    stops = ['#2F6E73', '#5D9A9D', '#9DCCCD', '#D8ECEB', '#F6FBFB']

    width, height = 430, 300
    x0, y0, cell = 86, 48, 110
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]
    _add_text(parts, width / 2, 20, 'Weather - T=336', size=18, weight='bold',
              anchor='middle', fill='#334144')

    for rr in range(2):
        for cc in range(2):
            val = float(mat[rr, cc])
            t = 0 if vmax == vmin else (val - vmin) / (vmax - vmin)
            fill = _interp_color(stops[::-1], t)
            rx = x0 + cc * cell
            ry = y0 + rr * cell
            parts.append(
                f'<rect x="{rx}" y="{ry}" width="{cell}" height="{cell}" fill="{fill}" stroke="none" />'
            )
            label = [k for k, pos in label_to_pos.items() if pos == (rr, cc)][0]
            best_line = ' BEST' if abs(val - best) < 1e-9 else ''
            _add_text(parts, rx + cell / 2, ry + cell / 2 - 8, label, size=13,
                      weight='bold', anchor='middle')
            _add_text(parts, rx + cell / 2, ry + cell / 2 + 12, f'MSE={val:.3f}{best_line}',
                      size=13, weight='bold', anchor='middle')

    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{2 * cell}" height="{2 * cell}" fill="none" stroke="#333" stroke-width="1.5" />'
    )
    _add_text(parts, x0 + cell / 2, y0 + 2 * cell + 24, 'No Patching', size=12, anchor='middle')
    _add_text(parts, x0 + 1.5 * cell, y0 + 2 * cell + 24, 'Patching', size=12, anchor='middle')
    _add_text(parts, x0 - 34, y0 + cell / 2 + 4, 'No CI', size=12, anchor='middle')
    _add_text(parts, x0 - 18, y0 + 1.5 * cell + 4, 'CI', size=12, anchor='middle')

    cb_x, cb_y, cb_h, cb_w = 345, 70, 170, 18
    for i in range(80):
        yy = cb_y + i * cb_h / 80
        frac = i / 79
        fill = _interp_color(stops, frac)
        parts.append(
            f'<rect x="{cb_x}" y="{yy:.2f}" width="{cb_w}" height="{cb_h / 80 + 0.5:.2f}" fill="{fill}" stroke="none" />'
        )
    for val in np.linspace(vmin, vmax, 4):
        py = _scale(val, vmin, vmax, cb_y + cb_h, cb_y)
        parts.append(f'<line x1="{cb_x + cb_w}" y1="{py:.2f}" x2="{cb_x + cb_w + 4}" y2="{py:.2f}" stroke="#666" />')
        _add_text(parts, cb_x + cb_w + 8, py + 4, f'{val:.2f}', size=10, fill='#555')
    _add_text(parts, cb_x + 30, cb_y + cb_h / 2, 'MSE', size=11, fill='#334144',
              anchor='middle')
    parts.append('</svg>')

    out = 'results/ablation/ablation_Weather_336_single.svg'
    _ensure_dir(out)
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print(f'Saved: {out}')


def _load_favorita_forecast_series(category='BEVERAGES', history_days=30):
    torch.set_num_threads(1)
    checkpoint_path = 'results/favorita/PatchTST_Favorita_30_best.pt'
    results_json = 'results/favorita/PatchTST_Favorita_30_results.json'
    data_path = 'data'

    with open(results_json) as f:
        info = json.load(f)
    seq_len = info['seq_len']
    pred_len = info['pred_len']

    train_data, val_data, test_data, family_names, _ = load_favorita_data(data_path, verbose=False)
    n_families = len(family_names)
    matches = [i for i, name in enumerate(family_names) if category.upper() in name.upper()]
    if matches:
        ch = matches[0]
        category_name = family_names[ch]
    else:
        ch = int(np.argmax(test_data.std(axis=0)))
        category_name = family_names[ch]

    cfg = info['config']
    model = PatchTST(
        enc_in=n_families,
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=cfg.get('patch_len', 16),
        stride=cfg.get('stride', 8),
        d_model=cfg.get('d_model', 16),
        n_heads=cfg.get('n_heads', 4),
        e_layers=cfg.get('e_layers', 3),
        d_ff=cfg.get('d_ff', 128),
        dropout=cfg.get('dropout', 0.2),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()

    _, _, test_loader, scaler = create_dataloaders(
        train_data, val_data, test_data, seq_len=seq_len, pred_len=pred_len, batch_size=32
    )
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y in test_loader:
            o = model(x)
            preds_list.append(o.numpy())
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
    shown_history = min(history_days, seq_len)
    context = context_full[-shown_history:]
    actual_future = targets_raw[idx, :, ch]
    predicted = preds_raw[idx, :, ch]
    return category_name, context, actual_future, predicted


def export_favorita_forecast_svg():
    category_name, context, actual_future, predicted = _load_favorita_forecast_series(
        category='BEVERAGES', history_days=30
    )
    shown_history = len(context)
    pred_len = len(actual_future)
    x_vals = list(range(shown_history + pred_len))
    y_all = np.concatenate([context, actual_future, predicted])
    y_min = float(y_all.min())
    y_max = float(y_all.max())
    y_pad = 0.08 * (y_max - y_min)
    y0v = y_min - y_pad
    y1v = y_max + y_pad

    width, height = 760, 430
    x0, y0, w, h = 20, 24, 720, 380
    plot_left = x0 + 56
    plot_right = x0 + w - 16
    plot_top = y0 + 26
    plot_bottom = y0 + h - 44

    def px_x(v):
        return _scale(v, 0, shown_history + pred_len - 1, plot_left, plot_right)

    def px_y(v):
        return _scale(v, y0v, y1v, plot_bottom, plot_top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]
    _add_text(parts, width / 2, 20, f'Favorita Forecast Example - {category_name}',
              size=18, weight='bold', anchor='middle', fill='#B31B1B')

    hist_left = px_x(0)
    hist_right = px_x(shown_history)
    parts.append(
        f'<rect x="{hist_left:.2f}" y="{plot_top:.2f}" width="{hist_right - hist_left:.2f}" '
        f'height="{plot_bottom - plot_top:.2f}" fill="#F5F1EC" opacity="0.95" />'
    )

    for frac in np.linspace(0, 1, 5):
        yv = y0v + frac * (y1v - y0v)
        py = px_y(yv)
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{py:.2f}" x2="{plot_right:.2f}" y2="{py:.2f}" '
            f'stroke="#d8d8d8" stroke-width="1" opacity="0.7" />'
        )
        _add_text(parts, plot_left - 8, py + 4, f'{int(round(yv))}', size=10, anchor='end', fill='#555')

    for xt in range(0, shown_history + pred_len, 10):
        px = px_x(xt)
        parts.append(
            f'<line x1="{px:.2f}" y1="{plot_top:.2f}" x2="{px:.2f}" y2="{plot_bottom:.2f}" '
            f'stroke="#ececec" stroke-width="1" />'
        )
        _add_text(parts, px, plot_bottom + 18, str(xt), size=10, anchor='middle', fill='#444')

    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_bottom:.2f}" x2="{plot_right:.2f}" y2="{plot_bottom:.2f}" stroke="#333" stroke-width="1.5" />'
    )
    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" y2="{plot_bottom:.2f}" stroke="#333" stroke-width="1.5" />'
    )

    context_pts = [(px_x(i), px_y(v)) for i, v in enumerate(context)]
    actual_pts = [(px_x(shown_history + i), px_y(v)) for i, v in enumerate(actual_future)]
    pred_pts = [(px_x(shown_history + i), px_y(v)) for i, v in enumerate(predicted)]

    parts.append(f'<polyline fill="none" stroke="#666666" stroke-width="2.0" points="{_line_path(context_pts)}" />')
    parts.append(f'<polyline fill="none" stroke="#1565C0" stroke-width="2.8" points="{_line_path(actual_pts)}" />')
    parts.append(f'<polyline fill="none" stroke="#F28E2B" stroke-width="2.8" stroke-dasharray="8 6" points="{_line_path(pred_pts)}" />')

    forecast_x = px_x(shown_history)
    parts.append(
        f'<line x1="{forecast_x:.2f}" y1="{plot_top:.2f}" x2="{forecast_x:.2f}" y2="{plot_bottom:.2f}" stroke="#111" stroke-width="1.2" stroke-dasharray="3 4" />'
    )
    _add_text(parts, forecast_x + 8, plot_top + 16, 'Forecast start', size=11)

    _add_text(parts, x0 + w / 2, y0 + h - 8, 'Day', size=12, anchor='middle')
    parts.append(
        f'<text x="{x0 + 16:.2f}" y="{y0 + h / 2:.2f}" font-size="12" fill="#222" text-anchor="middle" '
        f'transform="rotate(-90 {x0 + 16:.2f},{y0 + h / 2:.2f})">Daily Sales</text>'
    )

    lx, ly = x0 + 24, y0 + 30
    parts.append(
        f'<rect x="{lx - 10:.2f}" y="{ly - 16:.2f}" width="150" height="64" rx="6" fill="white" fill-opacity="0.9" stroke="#d0d0d0" />'
    )
    legend_rows = [
        ('History', '#666666', False),
        ('Actual', '#1565C0', False),
        ('PatchTST forecast', '#F28E2B', True),
    ]
    for i, (label, color, dashed) in enumerate(legend_rows):
        yy = ly + i * 18
        dash = ' stroke-dasharray="8 6"' if dashed else ''
        parts.append(f'<line x1="{lx:.2f}" y1="{yy:.2f}" x2="{lx + 26:.2f}" y2="{yy:.2f}" stroke="{color}" stroke-width="2.6"{dash} />')
        _add_text(parts, lx + 34, yy + 4, label, size=10, fill='#333')

    parts.append('</svg>')
    out = 'results/favorita/favorita_forecast_beverages_pred30.svg'
    _ensure_dir(out)
    with open(out, 'w') as f:
        f.write('\n'.join(parts))
    print(f'Saved: {out}')


def main():
    export_benchmark_trends_svg()
    export_favorita_trends_svg()
    export_ablation_weather_336_svg()
    export_favorita_forecast_svg()


if __name__ == '__main__':
    main()
