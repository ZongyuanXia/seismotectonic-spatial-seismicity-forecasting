"""Redraw the five installed figures from frozen data, without model execution.

All existing text sizes increase by 1 pt and use bold Times New Roman (including
math). Remove only the W1 date footer in Figure 3 and the footer in Figure 4.
Archived plotting scripts and experiment outputs remain read-only.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import shutil
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import cartopy
cartopy.config['data_dir'] = Path(os.environ.get('PAPER_CARTOPY_DATA', str(ROOT / 'cartopy')))
from cartopy.io.shapereader import NEShpDownloader
def _missing_offline_map(self, target_path, format_dict):
    raise FileNotFoundError('Bundled Natural Earth map asset is missing: ' + str(target_path))
NEShpDownloader.acquire_resource = _missing_offline_map
PROJECT = ROOT / 'outputs'
for _p in (PROJECT/'figures', PROJECT/'tmp'):
    _p.mkdir(parents=True,exist_ok=True)
ARCHIVE = ROOT
SOURCE = ROOT / 'plotting/inputs/convlstm'
BAR = ROOT / 'plotting/inputs/barline'
AUDIT = []


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def math_bold():
    # Font weight on Text alone does not affect math expressions.
    plt.rcParams.update({
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman:style=normal:weight=bold',
        'mathtext.it': 'Times New Roman:style=italic:weight=bold',
        'mathtext.bf': 'Times New Roman:style=normal:weight=bold',
        'mathtext.bfit': 'Times New Roman:style=italic:weight=bold',
        'mathtext.sf': 'Times New Roman:style=normal:weight=bold',
        'mathtext.tt': 'Times New Roman:style=normal:weight=bold',
        'mathtext.cal': 'STIXGeneral:style=italic:weight=bold',
    })


def finish(fig, filename, tight=False):
    math_bold()
    fig.canvas.draw()
    # Cartopy generates its own labels during drawing and reuses its style dicts.
    grid_text = set()
    for obj in fig.findobj():
        if obj.__class__.__name__ == 'Gridliner':
            for field in ('xlabel_style', 'ylabel_style'):
                style = getattr(obj, field)
                size = style.get('size', style.get('fontsize', plt.rcParams['font.size']))
                style.pop('size', None)
                style['fontsize'] = float(size) + 1
                style['fontweight'] = 'bold'
            for t in obj.label_artists:
                grid_text.add(id(t))
    records = []
    for t in fig.findobj(Text):
        before = t.get_fontsize()
        if id(t) not in grid_text:
            t.set_fontsize(before + 1)
        t.set_fontweight('bold')
        t.set_math_fontfamily('custom')
        if t.get_visible() and t.get_text():
            records.append({'text': t.get_text(), 'before_pt': before,
                            'after_pt': t.get_fontsize(), 'weight': t.get_fontweight()})
    if tight:
        fig.tight_layout(w_pad=1.2)
    fig.canvas.draw()
    assert all(t.get_fontweight() == 'bold' for t in fig.findobj(Text)
               if t.get_visible() and t.get_text())
    if filename.startswith('figure3'):
        assert not any(t.get_text().startswith('W1:') for t in fig.findobj(Text))
    if filename.startswith('figure4'):
        assert not fig.texts
    path = PROJECT / 'figures' / filename
    if path.exists():
        old = hashlib.sha256(path.read_bytes()).hexdigest()
        backup = PROJECT / 'tmp' / f'{path.stem}_before_bold_fonts_{old[:12]}.png'
        if not backup.exists():
            shutil.copy2(path, backup)
    fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=.04)
    fig.savefig(PROJECT / 'tmp' / f'{path.stem}_font_preview.png',
                dpi=140, bbox_inches='tight', pad_inches=.04)
    AUDIT.append({'file': str(path), 'text': records,
                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    print('Updated:', filename, flush=True)


def figure1():
    plt.rcdefaults()
    m = module('fig1_layout', ROOT / 'plotting/inputs/figure1/redraw_figure1.py')
    modules = m.load_plotting()
    # Retain the existing panel painters and exact combined layout. Omit the
    # standalone panel exports and reports; save only the installed manuscript PNG.
    src = inspect.getsource(m.draw)
    start = src.index("    out = BASE / 'figures'; out.mkdir(exist_ok=True)")
    combined = src.index('    # Same five-panel composition')
    end = src.index("    save(fig, 'Figure1_updated')")
    src = src[:start] + src[combined:end] + "    return fig\n"
    exec(compile(src, str(m.__file__), 'exec'), m.__dict__)
    fig = m.draw(*modules)
    finish(fig, 'figure1.png')


def figure2():
    plt.rcdefaults()
    m = module('fig2_layout', ROOT / 'plotting/inputs/rf/plot_revised_rf.py')
    m.style()
    m.REGIONS['智利中部'] = 'Chile'
    data = pd.read_csv(m.DATA / '图1_地区分类.csv')
    assert len(data) == 23
    m.save = lambda fig, *args: finish(fig, 'figure2.png', tight=True)
    m.multiregion(data, PROJECT / 'figures', [])


def figure3():
    plt.rcdefaults()
    m = module('fig3_layout', SOURCE / 'plot_Fig3_six_windows.py')
    m.SUMMARY=ROOT/'convlstm/runs/compact_core_direction_s0_summary'
    m.TRAINING=ROOT/'convlstm/training'
    m.style()
    summary = m.SUMMARY
    manifest = json.loads((summary / 'summary_manifest.json').read_text())
    assert m.sha(summary / 'six_window_maps.npz') == manifest['files_sha256']['six_window_maps.npz']
    with np.load(summary / 'six_window_maps.npz', allow_pickle=False) as z:
        arrays = {k: z[k][0].copy() for k in z.files}
    with np.load(SOURCE / 'map_background.npz', allow_pickle=False) as z:
        land, coast = m.geometry(z)
    grid = json.loads((summary / 'grid_and_protocol.json').read_text())
    with (SOURCE / 'Fig3_six_windows_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        stats = [r for r in csv.DictReader(f) if r['window'] == '1']
    assert len(stats) == 4
    # Expand saved binary masks only for the established display geometry.
    # Precision/recall annotations are read from the frozen table, not reevaluated.
    sys.path.insert(0, str(m.TRAINING))
    from spatial_symmetric import symmetric_expand
    expand = lambda a: symmetric_expand(a.astype(np.uint8), d=3) > 0
    truth = arrays['targets'] > 0
    assert np.array_equal(truth, arrays['targets'] * 255 >= 100)
    et = expand(truth)
    display = []
    for name, r in zip(m.MODELS, stats):
        pred = arrays[name + '_predictions'] * 255 >= 100
        em = expand(pred & et)
        ef = expand(pred & ~et)
        assert int(em.sum()) == int(r['expanded_matched_grid_count'])
        assert int(ef.sum()) == int(r['expanded_FP_grid_count'])
        c = np.zeros((340, 200), dtype=np.uint8)
        c[em] = 1
        c[ef] = 2
        display.append(c)
        r['precision'] = float(r['precision'])
        r['recall'] = float(r['recall'])
    src = inspect.getsource(m.plot_window)
    footer = '    fig.text(.52,.015,'
    assert src.count(footer) == 1
    src = '\n'.join(line for line in src.splitlines() if not line.startswith(footer))
    src = src[:src.index('    fig.savefig(')] + '    return fig\n'
    exec(compile(src, str(m.__file__), 'exec'), m.__dict__)
    args = argparse.Namespace(red='#E45756', purple='#4E79A7')
    fig = m.plot_window(display, truth, stats, {}, grid['region'], land, coast, args, '')
    # The date footer is removed; keep the existing three-item legend.
    finish(fig, 'figure3_first_window.png')


def figure4():
    plt.rcdefaults()
    m = module('fig4_layout', BAR / 'plot_precision_barline.py')
    data = json.loads((BAR / 'barline_data.json').read_text())
    assert data['sample_count'] == 2473 and data['direction_channels'] == [0, 3, 4]
    m.style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.5), sharey=True)
    for i, (ax, group) in enumerate(zip(axes, m.GROUPS)):
        m.draw(ax, group, data, ['(a)', '(b)', '(c)'][i])
        if i:
            ax.set_ylabel('')
    fig.subplots_adjust(left=.055, right=.99, top=.90, bottom=.22, wspace=.15)
    # No bottom footer. Panel names remain at their original top positions.
    finish(fig, 'figure4_gain_sources.png')


def figure5():
    plt.rcdefaults()
    m = module('fig5_layout', SOURCE / 'plot_section44_precision_sensitivity.py')
    data = json.loads((SOURCE / 'section44_plot_data.json').read_text())
    models = ['M2', 'M3', 'M4']
    records = data['records']
    matrices = []
    for kind, x in [('threshold', data['threshold_order']), ('scale', data['scale_order'])]:
        matrices.append(np.array([[next(r['precision_gain_pp'] for r in records
            if r['model'] == model and (r['d'] == 3 and r['pred_threshold'] == t
            if kind == 'threshold' else r['pred_threshold'] == 100 and r['d'] == t))
            for t in x] for model in models]))
    m.style()
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 5], wspace=.30)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    m.panel(a, matrices[0], data['threshold_order'], models, 'Prediction threshold', '(a)')
    im = m.panel(b, matrices[1], data['scale_order'], models, 'Spatial expansion scale, d', '(b)')
    fig.subplots_adjust(left=.065, right=.89, bottom=.20, top=.87)
    box = b.get_position()
    cax = fig.add_axes([box.x1 + .012, box.y0, .015, box.height])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 3, 6, 9, 12])
    cb.set_label(r'$\Delta P$ (percentage points)', fontsize=20)
    cb.ax.tick_params(length=5, width=1)
    cb.outline.set_linewidth(1.2)
    finish(fig, 'figure5_precision_sensitivity.png')


if __name__ == '__main__':
    for draw in (figure1, figure2, figure3, figure4, figure5):
        draw()
    (PROJECT / 'tmp/figure_font_update_verification.json').write_text(
        json.dumps({'font_increment_pt': 1, 'weight': 'bold', 'figures': AUDIT,
                    'no_model_execution': True}, ensure_ascii=False, indent=2) + '\n')
