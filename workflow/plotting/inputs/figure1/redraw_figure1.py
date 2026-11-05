"""Redraw Figure 1 from audited experiment data; no model execution.

Coordinates are degrees, magnitudes are archived Mw values, counts are records.
Original plotting functions and their style constants are retained verbatim in
original_scripts. Run with --reuse-data to plot the supplied CSV snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(BASE / 'tmp' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd

DEFAULT_ARCHIVE = Path('/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/九项指标重构归档')
NAMES = {'japan': 'Japan', 'greece': 'Greece', 'indonesia': 'Indonesia',
         'mexico': 'Mexico', 'central_chile': 'Chile', 'italy': 'Italy',
         'new_zealand': 'New Zealand', 'taiwan': 'Taiwan', 'iran': 'Iran',
         'southern_california': 'Southern California'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def prepare(archive):
    """Read existing audit tables and reproduce only the common WSM filter."""
    from source_rules import load_wsm
    data = BASE / 'data'
    data.mkdir(exist_ok=True)
    rf = archive / 'datasets/rf_wsm'
    cl = archive / 'datasets/cl10_mapping_v2'
    manifest = json.loads((rf / 'manifest.json').read_text())
    wsm_path = Path(manifest['wsm_path'])
    assert sha(wsm_path) == manifest['wsm_sha256']
    assert sha(BASE / 'source_rules.py') == manifest['core_sha256']
    hist, _, audit = load_wsm(wsm_path)
    assert audit == manifest['preprocessing'] or all(
        audit[k] == v for k, v in manifest['preprocessing'].items() if k in audit)
    assert len(hist) == 65513
    hist = hist[['ID', 'ISC_ID', 'DAY', 'LAT', 'LON', 'TYPE', 'QUALITY']].copy()
    hist.to_csv(data / 'wsm_global_history.csv', index=False)
    # Use the exact nominal Japan limits used for the WSM regional analysis.
    japan = hist[hist.LAT.between(30.05, 45.95) & hist.LON.between(128.10, 146.80)].copy()
    japan.to_csv(data / 'wsm_japan_history.csv', index=False)
    usgs = pd.read_csv(cl / 'usgs_event_audit.csv')
    assert len(usgs) == 6077 and usgs.MW.ge(4).all()
    usgs.to_csv(data / 'usgs_japan_events.csv', index=False)
    train = pd.read_csv(rf / 'rf_training_1978_2024.csv')
    assert train.all_nine_valid.all() and len(train) == 13204
    assert pd.to_datetime(train.date).between('1978-01-01', '2024-12-31').all()
    counts = train.groupby('region').size().rename('rf_event_count').reset_index()
    recorded = pd.read_csv(rf / 'rf_region_counts.csv').set_index('region')
    for row in counts.itertuples():
        assert row.rf_event_count == recorded.loc[row.region, 'all_nine_valid']
    counts['display_name'] = counts.region.map(NAMES)
    counts = counts.sort_values('rf_event_count', ascending=False)
    counts.to_csv(data / 'rf_region_counts.csv', index=False)
    # Boundaries come from the RF configuration recorded by its manifest.
    cfg_path = Path(manifest['config_path'])
    assert sha(cfg_path) == manifest['config_sha256']
    cfg = module(cfg_path, 'figure1_rf_config')
    regions = {NAMES[k]: {'lat': [r.lat_min, r.lat_max],
                          'lon': [r.lon_min, r.lon_max]} for k, r in cfg.REGIONS.items()}
    (data / 'regions.json').write_text(json.dumps(regions, indent=2) + '\n')
    months = pd.date_range('1976-01-01', '2024-12-01', freq='MS')
    def monthly(series):
        dt = pd.to_datetime(series, utc=True).dt.tz_convert(None).dt.to_period('M').dt.to_timestamp()
        return dt.value_counts().reindex(months, fill_value=0).sort_index()
    uc = monthly(usgs.time).astype(float)
    uc.loc[uc.index < '1978-01-01'] = np.nan  # unselected period, not zero earthquakes
    wc = monthly(japan.DAY)
    monthly_table = pd.DataFrame({'month': months, 'usgs_japan_mw_ge_4': uc.values,
                                 'wsm_japan_records': wc.values})
    assert int(monthly_table.usgs_japan_mw_ge_4.sum()) == len(usgs)
    assert int(monthly_table.wsm_japan_records.sum()) == len(japan)
    monthly_table.to_csv(data / 'monthly_counts.csv', index=False)
    sources = [wsm_path, rf / 'manifest.json', rf / 'rf_training_1978_2024.csv',
               rf / 'rf_region_counts.csv', cl / 'usgs_event_audit.csv', cl / 'manifest.json', cfg_path]
    record = {'sources': {str(p): sha(p) for p in sources},
              'counts': {'wsm_global_history': len(hist), 'wsm_japan_history': len(japan),
                         'wsm_japan_before_1978': int((pd.to_datetime(japan.DAY, utc=True).dt.year < 1978).sum()),
                         'usgs_japan': len(usgs), 'rf_all_regions': len(train),
                         'rf_japan': int(counts.loc[counts.region.eq('japan'), 'rf_event_count'].iloc[0])},
              'wsm_date_min': str(hist.DAY.min()), 'wsm_date_max': str(hist.DAY.max()),
              'panel_d_definition': 'Nine-valid-feature RF event rows, 1978-2024; before balancing, not summed across magnitude tasks.',
              'region_bounds': regions,
              'data_sha256': {p.name: sha(p) for p in data.iterdir() if p.is_file()}}
    (BASE / 'provenance.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('Verified data:', record['counts'], flush=True)


def load_plotting():
    for p in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
        font_manager.fontManager.addfont(str(p))
    font = font_manager.findfont(font_manager.FontProperties(family='Times New Roman'), fallback_to_default=False)
    filenames = ['Fig1a_USGS_Japan.py', 'Fig1b_WSM_Japan.py', 'Fig1c_Regions_WSM.py',
                 'Fig1d_WSM_Counts.py', 'Fig1e_Monthly_Counts.py']
    a, b, c, d, e = [module(BASE / 'original_scripts' / n, 'figure1_' + n[:-3]) for n in filenames]
    for m in (a, b, c, d, e):
        m.OUT_DIR = BASE / 'figures'
        m.SAVE_TIFF = False
        m.DPI = 600
    c.LEGEND_LABEL_WSM = 'WSM records (1976-2024)'
    regions = json.loads((BASE / 'data/regions.json').read_text())
    for name, limits in regions.items():
        c.REGIONS[name].update(limits)
    # Reposition labels only where the formal RF bounds differ from old map boxes.
    c.REGIONS['Southern California']['label'] = (-109., 42.0)
    d.BAR_XLABEL = 'RF event count (1978-2024)'
    e.FORMAL_START_DATE = pd.Timestamp('1978-01-01')
    a.apply_matplotlib_style()
    print('Font:', font, flush=True)
    return a, b, c, d, e


def draw(a, b, c, d, e):
    data = BASE / 'data'
    usgs = pd.read_csv(data / 'usgs_japan_events.csv').rename(columns={'MW': 'Mw'})
    japan = pd.read_csv(data / 'wsm_japan_history.csv')
    hist = pd.read_csv(data / 'wsm_global_history.csv')
    counts = pd.read_csv(data / 'rf_region_counts.csv')
    bars = pd.DataFrame({'region': counts.display_name, 'wsm_record_count': counts.rf_event_count,
                         'role': np.where(counts.region.eq('japan'), 'main', 'aux')})
    monthly = pd.read_csv(data / 'monthly_counts.csv', parse_dates=['month'])
    def pa(ax):
        a.style_geo_axes(ax); a.plot_events(ax, usgs); a.add_size_legend(ax)
        ax.text(.02, .98, '(a)', transform=ax.transAxes, ha='left', va='top', fontsize=a.PANEL_LABEL_SIZE, fontweight='bold', zorder=20)
    def pb(ax):
        b.style_geo_axes(ax)
        ax.scatter(japan.LON, japan.LAT, s=b.WSM_MARKER_SIZE, c=b.WSM_BEFORE_COLOR,
                   alpha=b.WSM_ALPHA_BEFORE, edgecolors='none', linewidths=0,
                   transform=b.ccrs.PlateCarree(), zorder=4, rasterized=True)
        ax.text(.02, .98, '(b)', transform=ax.transAxes, ha='left', va='top', fontsize=b.PANEL_LABEL_SIZE, fontweight='bold', zorder=20)
    def pc(ax): c.plot_region_map(ax, hist)
    def pd_(ax, add_label=True):
        d.plot_region_bars(ax, bars)
        ax.set_xlim(0, bars.wsm_record_count.max() * 1.32)
        if add_label:
            ax.text(-.45, 1.045, '(d)', transform=ax.transAxes, ha='left', va='bottom', fontsize=14, fontweight='bold')
    def pe(top, bottom):
        ticks = ['1976-01-01', '1990-01-01', '2000-01-01', '2010-01-01', '2024-12-01']
        for ax, field, color, title, log in [
            (top, 'usgs_japan_mw_ge_4', e.USGS_COLOR, 'USGS catalog: Japan', True),
            (bottom, 'wsm_japan_records', e.WSM_COLOR, 'WSM2025 records: Japan', True)]:
            y = monthly[field]
            e.plot_series(ax, monthly.month, y.where(y.gt(0)) if log else y, color)
            if log:
                positive = y.gt(0)
                isolated = positive & ~positive.shift(1, fill_value=False) & ~positive.shift(-1, fill_value=False)
                ax.plot(monthly.loc[isolated, 'month'], y[isolated], linestyle='none',
                        marker='.', markersize=1.8, color=color, alpha=e.RAW_ALPHA)
            e.format_time_axis(ax, x_start='1976-01-01', x_end='2024-12-31',
                               show_xlabel=ax is bottom, major_ticks=ticks)
            e._configure_y_axis(ax, y, log)
            ax.set_ylabel('Counts', fontsize=e.AXIS_LABEL_SIZE)
            e.add_axis_top_label(ax, title)
            ax.axvline(pd.Timestamp('1978-01-01'), color=e.REFERENCE_LINE_COLOR,
                       lw=e.REFERENCE_LINEWIDTH, ls=e.REFERENCE_LINESTYLE, zorder=4)
            ax.text(pd.Timestamp('1978-01-01'), .93, '1978-01', transform=ax.get_xaxis_transform(),
                    ha='left', va='top', fontsize=6, color=e.REFERENCE_LINE_COLOR)
    out = BASE / 'figures'; out.mkdir(exist_ok=True)
    previews = BASE / 'preview'; previews.mkdir(exist_ok=True)
    def save(fig, stem):
        fig.savefig(out / (stem + '.pdf'), bbox_inches='tight', pad_inches=.03)
        fig.savefig(out / (stem + '.png'), dpi=600, bbox_inches='tight', pad_inches=.03)
        fig.savefig(previews / (stem + '.png'), dpi=140, bbox_inches='tight', pad_inches=.03)
        plt.close(fig); print('Saved', stem, flush=True)
    for name, painter, projection in [('Fig1a_USGS_Japan', pa, a.PROJECTION),
                                      ('Fig1b_WSM_Japan', pb, b.PROJECTION)]:
        fig = plt.figure(figsize=(8.5 / 2.54, 8.2 / 2.54))
        painter(fig.add_axes([.10, .10, .86, .86], projection=projection));save(fig, name)
    fig = plt.figure(figsize=(c.MAP_WIDTH_CM / 2.54, c.MAP_HEIGHT_CM / 2.54))
    pc(fig.add_axes(c.MAP_AX, projection=c.PROJECTION)); save(fig, 'Fig1c_Regions_WSM')
    fig = plt.figure(figsize=(d.BAR_WIDTH_CM / 2.54, d.BAR_HEIGHT_CM / 2.54))
    pd_(fig.add_axes(d.BAR_AX));save(fig, 'Fig1d_RF_Counts')
    fig = plt.figure(figsize=(e.FIG_WIDTH_CM / 2.54, e.FIG_HEIGHT_CM / 2.54))
    pe(fig.add_axes(e.AX_TOP), fig.add_axes(e.AX_BOTTOM))
    fig.text(.015, .965, '(e)', ha='left', va='top', fontsize=14, fontweight='bold')
    save(fig, 'Fig1e_Monthly_Counts_Japan')
    # Same five-panel composition as the manuscript: two Japan maps, global map,
    # then regional bars beside two monthly series.
    fig = plt.figure(figsize=(18 / 2.54, 25.5 / 2.54))
    axa = fig.add_axes([.065, .727, .415, .261], projection=a.PROJECTION)
    axb = fig.add_axes([.56, .727, .415, .261], projection=b.PROJECTION)
    axc = fig.add_axes([.065, .365, .91, .322], projection=c.PROJECTION)
    axd = fig.add_axes([.12, .048, .205, .252])
    axt = fig.add_axes([.425, .205, .55, .105])
    axw = fig.add_axes([.425, .048, .55, .105])
    pa(axa);pb(axb);pc(axc);pd_(axd, False);pe(axt, axw)
    fig.text(.015, .333, '(d)', ha='left', va='bottom', fontsize=14, fontweight='bold')
    fig.text(.425, .333, '(e)', ha='left', va='bottom', fontsize=14, fontweight='bold')
    for lon, corner in [(128.10, (0, 0)), (146.80, (1, 0))]:
        connector = ConnectionPatch(xyA=(lon, 45.95), xyB=corner,
            coordsA=c.DATA_CRS._as_mpl_transform(axc), coordsB=axb.transAxes,
            color='#555555', lw=1.5, ls='--', zorder=15, clip_on=False)
        fig.add_artist(connector)
    save(fig, 'Figure1_updated')
    summary = {'font_family': 'Times New Roman', 'panel_d_counts': dict(zip(counts.display_name, counts.rf_event_count)),
               'panel_e_axes': ['log', 'log'], 'zero_count_months_masked_without_pseudocounts': True,
               'global_wsm_count': len(hist), 'japan_wsm_count': len(japan), 'usgs_count': len(usgs),
               'original_plot_scripts_sha256': {p.name: sha(p) for p in (BASE / 'original_scripts').glob('*.py')},
               'outputs_sha256': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
               'model_training_inference_or_evaluation': False}
    (BASE / 'plot_verification.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument('--reuse-data', action='store_true', help='Use supplied CSVs without reading the original archive')
    args = parser.parse_args()
    if not args.reuse_data: prepare(args.archive)
    else:
        manifest = json.loads((BASE / 'provenance.json').read_text())
        for name, digest in manifest['data_sha256'].items():
            assert sha(BASE / 'data' / name) == digest, name
    draw(*load_plotting())
