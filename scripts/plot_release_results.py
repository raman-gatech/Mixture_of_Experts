"""Plot frozen aggregate proxies; these figures do not imply reviewed accuracy."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result = json.loads(args.results.read_text())
    names = ['parent-baseline-256', 'parent-precision-512', 'candidate-precision-512']
    labels = ['Parent 256', 'Parent 512', 'Stage 3 512']
    colors = ['#64748b', '#2563eb', '#d97706']
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), layout='constrained')
    for ax, field, denominator, title in zip(axes,
        ['math_numeric_proxy_matches', 'science_phrase_proxy_matches', 'truncated'],
        [200, 100, 500], ['Math numeric proxy', 'Science phrase proxy', 'Truncated outputs']):
        counts = [result['arms'][name][field] for name in names]
        bars = ax.bar(labels, [100 * n / denominator for n in counts], color=colors)
        ax.set_ylim(0, 100)
        ax.set_title(title)
        ax.set_ylabel('Percent of cases')
        ax.bar_label(bars, labels=[f'{n}/{denominator}' for n in counts], padding=4, fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Frozen comparison: automatic proxies, not independently reviewed accuracy', fontsize=11)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        fig.savefig(args.output / ('frozen-comparison.' + suffix), dpi=180)
    plt.close(fig)
    print('Wrote PNG and SVG aggregate figures')


if __name__ == '__main__':
    main()
