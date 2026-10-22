"""Summarize measured timing and numerical agreement without fixed result values."""
import csv
import json
from pathlib import Path

import numpy as np


def summarize(report):
    if not report.get('complete'):
        raise ValueError('Cannot summarize an incomplete benchmark')
    rows = []
    steps = sorted({row['steps'] for row in report['rows']})
    for count in steps:
        values = {'steps': count}
        for mode in ('reference', 'cached', 'cuda_graph'):
            cases = [row for row in report['rows'] if row['steps'] == count and row['mode'] == mode]
            samples = []
            for case in cases:
                samples.extend(case['samples_ms'])
            if not samples:
                raise ValueError(f'Missing {mode} measurements at {count} steps')
            values[mode + '_ms'] = float(np.mean(samples))
            values[mode + '_p95_ms'] = float(np.percentile(samples, 95))
        values['optimized_hz'] = 1000 / values['cuda_graph_ms']
        values['speedup'] = values['reference_ms'] / values['cuda_graph_ms']
        errors = [row['action_difference_max_rad'] for row in report['rows']
                  if row['steps'] == count and row['mode'] == 'cuda_graph']
        values['max_action_deviation_rad'] = max(errors)
        rows.append(values)
    return rows


def write_tables(rows, output):
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Use one measured curve per implementation and a paired numerical-error panel.
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        figure, axes = plt.subplots(1, 2, figsize=(8, 3), layout='constrained')
        steps = [row['steps'] for row in rows]
        for mode, label in [('reference', 'Reference'), ('cached', 'KV cache'), ('cuda_graph', '+ Graph')]:
            values = [row[mode + '_ms'] for row in rows]
            axes[0].plot(steps, values, 'o-', label=label)
        axes[0].axhline(1000 / 24, color='.6', ls='--', lw=1, label='24 Hz budget')
        axes[0].set(xlabel='Flow steps', ylabel='Latency (ms)')
        axes[0].legend(frameon=False, fontsize=8)
        errors = [row['max_action_deviation_rad'] * 1000 for row in rows]
        axes[1].bar(range(len(steps)), errors, color='#7294a4')
        axes[1].set(xticks=range(len(steps)), xticklabels=steps,
                    xlabel='Flow steps', ylabel='Max. action deviation (mrad)')
        figure.savefig(output / 'summary.png', dpi=180)
        figure.savefig(output / 'summary.pdf')
        plt.close(figure)


def run(source, output):
    report = json.loads(Path(source).read_text())
    rows = summarize(report)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_tables(rows, output)
    plot(rows, output)
    return rows
