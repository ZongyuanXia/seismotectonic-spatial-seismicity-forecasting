"""Alternative Figure 1(e): both Japan monthly count axes are logarithmic.

Reads the same audited CSV as the existing figure. Zero-count months cannot
appear on a logarithmic axis and are masked without adding pseudo-counts.
"""
import json
from pathlib import Path

from redraw_figure1 import BASE, load_plotting, sha
import matplotlib.pyplot as plt
import pandas as pd


def main():
    _, _, _, _, e = load_plotting()
    source = BASE / 'data/monthly_counts.csv'
    provenance = json.loads((BASE / 'provenance.json').read_text())
    assert sha(source) == provenance['data_sha256'][source.name]
    table = pd.read_csv(source, parse_dates=['month'])
    fig = plt.figure(figsize=(e.FIG_WIDTH_CM / 2.54, e.FIG_HEIGHT_CM / 2.54))
    top, bottom = fig.add_axes(e.AX_TOP), fig.add_axes(e.AX_BOTTOM)
    ticks = ['1976-01-01', '1990-01-01', '2000-01-01', '2010-01-01', '2024-12-01']
    masked = {}
    for ax, field, color, title in [
        (top, 'usgs_japan_mw_ge_4', e.USGS_COLOR, 'USGS catalog: Japan'),
        (bottom, 'wsm_japan_records', e.WSM_COLOR, 'WSM2025 records: Japan')]:
        y = table[field]
        masked[field] = int(y.eq(0).sum())
        e.plot_series(ax, table.month, y.where(y.gt(0)), color)
        positive = y.gt(0)
        isolated = positive & ~positive.shift(1, fill_value=False) & ~positive.shift(-1, fill_value=False)
        # A single positive month between gaps has no line segment; retain it
        # as a small dot rather than silently dropping a real observation.
        ax.plot(table.loc[isolated, 'month'], y[isolated], linestyle='none',
                marker='.', markersize=1.8, color=color, alpha=e.RAW_ALPHA)
        e.format_time_axis(ax, x_start='1976-01-01', x_end='2024-12-31',
                           show_xlabel=ax is bottom, major_ticks=ticks)
        e._configure_y_axis(ax, y, True)
        ax.set_ylabel('Counts', fontsize=e.AXIS_LABEL_SIZE)
        e.add_axis_top_label(ax, title)
        ax.axvline(pd.Timestamp('1978-01-01'), color=e.REFERENCE_LINE_COLOR,
                   lw=e.REFERENCE_LINEWIDTH, ls=e.REFERENCE_LINESTYLE, zorder=4)
        ax.text(pd.Timestamp('1978-01-01'), .93, '1978-01',
                transform=ax.get_xaxis_transform(), ha='left', va='top',
                fontsize=6, color=e.REFERENCE_LINE_COLOR)
    fig.text(.015, .965, '(e)', ha='left', va='top', fontsize=14, fontweight='bold')
    stem = 'Fig1e_Monthly_Counts_Japan_both_log'
    for folder, suffix, dpi in [('figures', '.pdf', None), ('figures', '.png', 600),
                                ('preview', '.png', 140)]:
        path = BASE / folder / (stem + suffix)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.03)
        print(path)
    plt.close(fig)
    record = {'source_sha256': sha(source), 'axes': ['log', 'log'],
              'zero_count_months_masked': masked, 'pseudo_counts_added': False,
              'outputs': {suffix: sha(BASE / 'figures' / (stem + suffix))
                          for suffix in ['.png', '.pdf']}}
    (BASE / 'both_log_verification.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
