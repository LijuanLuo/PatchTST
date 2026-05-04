import base64
import os
from pathlib import Path


FILES = [
    'results/benchmark/benchmark_trends.png',
    'results/favorita/favorita_trends.png',
    'results/ablation/ablation_Weather_336_single.png',
    'results/favorita/favorita_forecast_beverages_pred30.png',
]


def get_png_size(path: Path):
    with path.open('rb') as f:
        sig = f.read(24)
    if sig[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'Not a PNG: {path}')
    width = int.from_bytes(sig[16:20], 'big')
    height = int.from_bytes(sig[20:24], 'big')
    return width, height


def wrap_png_as_svg(path_str: str):
    path = Path(path_str)
    width, height = get_png_size(path)
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image width="{width}" height="{height}" href="data:image/png;base64,{data}" />
</svg>
'''
    out = path.with_suffix('.svg')
    out.write_text(svg)
    print(f'Saved: {out}')


def main():
    for path in FILES:
        if os.path.exists(path):
            wrap_png_as_svg(path)
        else:
            print(f'Skipped missing file: {path}')


if __name__ == '__main__':
    main()
