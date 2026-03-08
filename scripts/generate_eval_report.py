#!/usr/bin/env python3
import os
import re
import math
import argparse
from datetime import datetime

DEFAULT_LOG_PATH = '/mnt/data/logs/eval_babyai_multi.log'
DEFAULT_OUT_DIR = '/mnt/data/saves/eval_reports'
DEFAULT_PREFIX = 'babyai_multi_eval_metrics'


def to_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float('nan')


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate eval TXT/PNG report from log.')
    parser.add_argument('--log', default=DEFAULT_LOG_PATH)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--prefix', default=DEFAULT_PREFIX)
    args = parser.parse_args()

    log_path = args.log
    out_dir = args.out_dir
    txt_name = f'{args.prefix}.txt'
    png_name = f'{args.prefix}.png'

    os.makedirs(out_dir, exist_ok=True)

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    total_block = re.findall(
        r'============Total Task Evaluation============\s*'
        r'Avg@1:\s*([0-9.]+|nan)\s*'
        r'Pass@1:\s*([0-9.]+|nan)',
        text,
        flags=re.MULTILINE,
    )
    if total_block:
        avg_total = to_float(total_block[-1][0])
        pass_total = to_float(total_block[-1][1])
    else:
        avg_total = float('nan')
        pass_total = float('nan')

    cat_pattern = re.compile(
        r'Category:\s*([^\n]+)\nAvg@1:\s*([0-9.]+|nan)(?:\nPass@1:\s*([0-9.]+|nan))?',
        re.MULTILINE,
    )
    cat_vals = {}
    for m in cat_pattern.finditer(text):
        cat = m.group(1).strip()
        a = to_float(m.group(2))
        p = to_float(m.group(3)) if m.group(3) is not None else float('nan')
        cat_vals[cat] = (a, p)

    had_error = (
        'Traceback (most recent call last):' in text
        or 'Error executing job with overrides' in text
    )

    txt_path = os.path.join(out_dir, txt_name)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('BabyAI Multi-Agent Eval Metrics\n')
        f.write(f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}\n')
        f.write(f'Log file: {log_path}\n')
        f.write(f'Eval had error: {had_error}\n\n')
        f.write(f'Avg@1: {avg_total}\n')
        f.write(f'Pass@1: {pass_total}\n\n')
        if cat_vals:
            f.write('Per-category\n')
            for c, (a, p) in sorted(cat_vals.items()):
                f.write(f'- {c}: Avg@1={a}, Pass@1={p}\n')

    png_path = os.path.join(out_dir, png_name)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        labels = ['Avg@1', 'Pass@1']
        values = [
            0.0 if math.isnan(avg_total) else avg_total,
            0.0 if math.isnan(pass_total) else pass_total,
        ]

        plt.figure(figsize=(7, 4.5))
        bars = plt.bar(labels, values, color=['#3b82f6', '#10b981'])
        plt.ylim(0, 1)
        plt.ylabel('Score')
        title = 'BabyAI Multi-Agent Eval (Latest Run)'
        if had_error:
            title += ' - run ended with error'
        plt.title(title)
        for b, v in zip(bars, values):
            plt.text(
                b.get_x() + b.get_width() / 2,
                min(v + 0.02, 0.98),
                f'{v:.4f}',
                ha='center',
                va='bottom',
                fontsize=10,
            )
        plt.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        print('PNG_WRITTEN', png_path)
    except Exception as e:
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write('\n[WARN] PNG was not generated with matplotlib.\n')
            f.write(f'Reason: {e}\n')
        print('PNG_FAILED', str(e))

    print('TXT_WRITTEN', txt_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
