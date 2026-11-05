"""Paths, explicit source selection and file provenance (standard library)."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

SOFTWARE = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())
MODELS = {m['model']: m for m in PROTOCOL['models']}
RECOMPUTED = ('rf_analysis/results_1000', 'rf_pairs/results', 'rf_ablation/results',
              'convlstm/runs', 'compare/compact_compare_summary', 'paper_results',
              'baseline/final_comparison', 'baseline/results',
              'baseline/etas/MC_sensitivity_1000_5000_20261030/results')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No rows to export: ' + str(path))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def module(name, path):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj; spec.loader.exec_module(obj)
    return obj


class Layout:
    def __init__(self, data=None, output=None, source='archived', weights='frozen'):
        self.source = source; self.weights = weights
        self.data = Path(data).resolve() if data else SOFTWARE.parent / 'data'
        self.out = Path(output).resolve() if output else SOFTWARE.parent / 'reproduced' / source
        self.project = self.out / 'project'
        self.results = self.project / 'paper_results'
        self.tables = self.out / 'tables'
        self.figures = self.out / 'figures'
        for protected in (self.data, SOFTWARE):
            if self.out == protected or self.out in protected.parents or protected in self.out.parents:
                raise ValueError('Output must be separate from software and data: ' + str(self.out))

    def prepare(self):
        """Copy into a working directory; never modify released inputs/sources."""
        if not (self.data / 'datasets/cl10_mapping_v2/manifest.json').is_file():
            raise FileNotFoundError('Companion data not found: ' + str(self.data))
        identity = dict(source=self.source, data=str(self.data), software=str(SOFTWARE), weights=self.weights)
        marker = self.out / 'workspace.json'
        if marker.exists() and load(marker) != identity:
            raise ValueError('Output belongs to a different configuration; choose a new --output-root')
        self.project.mkdir(parents=True, exist_ok=True)
        for source, code in [(SOFTWARE / 'workflow', True), (self.data, False)]:
            for path in sorted(source.rglob('*')):
                if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:
                    continue
                rel = path.relative_to(source)
                # install_data.py from earlier snapshots may have merged data
                # into workflow/. The explicit companion tree stays authoritative.
                if code and path.suffix!='.py' and (self.data/rel).is_file():
                    continue
                if rel.parts[0] in ('outputs', 'generated') or path.name in ('FILE_MANIFEST.csv', 'SHA256SUMS.txt', '.DS_Store'):
                    continue
                if not code and self.source == 'computed' and any(rel == Path(p) or Path(p) in rel.parents for p in RECOMPUTED):
                    continue
                dst = self.project / rel
                if dst.exists():
                    # Data in a computed workspace may have been rebuilt there.
                    if code and sha(dst) != sha(path):
                        raise ValueError('Working source differs: ' + str(dst))
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dst)
        aliases_file=self.data/'file_aliases.json'
        if aliases_file.exists():
            for destination,item in load(aliases_file)['aliases'].items():
                rel=Path(destination);canonical=Path(item['source'])
                if rel.is_absolute() or canonical.is_absolute() or '..' in rel.parts or '..' in canonical.parts:
                    raise ValueError('Unsafe shared-file path')
                if self.source=='computed' and any(rel==Path(p) or Path(p) in rel.parents for p in RECOMPUTED):
                    continue
                # Read the preserved canonical input, not a newly calculated
                # result at the same working path, for a frozen reference alias.
                src=self.data/canonical;dst=self.project/rel
                if not src.is_file() or sha(src)!=item['sha256']:
                    raise ValueError('Shared canonical input differs: '+str(canonical))
                if dst.exists():
                    if sha(dst)!=item['sha256']:raise ValueError('Shared working file differs: '+str(rel))
                else:
                    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        dump(marker, identity)
        return self.project

    def run_dir(self, model):
        return self.project / ('trained' if self.weights == 'retrained' else 'checkpoints') / model

    def check_model(self, model):
        entry = MODELS[model]; run = self.run_dir(model)
        for name in ('best.pt', 'run.json', 'summary.json'):
            if not (run / name).is_file():
                raise FileNotFoundError('Missing %s/%s; import the cloud delivery first' % (model, name))
        if self.weights == 'frozen' and sha(run / 'best.pt') != entry['sha256']:
            raise ValueError('Wrong frozen checkpoint: ' + model)
        config = load(run / 'run.json')
        if config['selected_channels'] != entry['selected_channels']:
            raise ValueError('Wrong channels: ' + model)
        for key in ('init_seed', 'order_seed'):
            if config[key] != PROTOCOL[key]: raise ValueError('Wrong seed: ' + model)
        names={'manifest':'manifest.json','sample_index':'sample_index.csv','usgs_daily':'usgs_daily.npz'}
        for key, name in names.items():
            actual=sha(self.project/'datasets/cl10_mapping_v2'/name)
            if config['dataset_sha256'][key] != actual: raise ValueError('Model/data mismatch: ' + model)
            if self.weights=='frozen' and actual!=PROTOCOL['expected_dataset_sha256'][key]:
                raise ValueError('Frozen model requires the original processed data identity: ' + model)
        return run

    def predictions(self, model, split):
        return self.results / 'predictions' / model / split

    def evaluation(self, model, split):
        return self.results / 'evaluations' / model / split

    def sensitivity(self, model, split):
        return self.results / 'sensitivity' / model / split


def import_cloud(layout, folder, include_results=True):
    """Verify downloaded bytes, then stage models/results without loading models."""
    folder = Path(folder).resolve(); report = load(folder / 'collection_report.json')
    records = report['copied_files']
    root_record = next(r for r in records if Path(r['destination']).name == 'formal_checkpoints_required.json')
    cloud_root = Path(root_record['destination']).parent
    ledger = {}
    for r in records:
        rel = Path(r['destination']).relative_to(cloud_root)
        if '..' in rel.parts: raise ValueError('Unsafe delivery path')
        src = folder / rel
        if src.is_symlink() or not src.is_file() or sha(src) != r['sha256']:
            raise ValueError('Downloaded file differs from report: ' + str(rel))
        ledger[str(rel)] = r
    added = []
    def copy(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha(dst) != sha(src): raise ValueError('Conflicting file: ' + str(dst))
        if not dst.exists(): shutil.copy2(src, dst)
        added.append(dict(path=str(dst.relative_to(layout.project)), sha256=sha(dst)))
    for model, entry in MODELS.items():
        src = folder / model
        if not (src / 'best.pt').exists(): continue
        if sha(src / 'best.pt') != entry['sha256']: raise ValueError('Wrong cloud model: ' + model)
        for p in src.iterdir():
            if p.is_file() and str(p.relative_to(folder)) in ledger:
                copy(p, layout.project / 'checkpoints' / model / p.name)
    chosen = {};excluded=[]
    for e in report['evaluations']:
        if not include_results:continue
        key = e['model'], e['split'], e['kind']
        rel = Path(e['destination']).relative_to(cloud_root)
        src = folder / rel
        expected=MODELS[e['model']].get('analysis_identities',{}).get(e['split'],{}).get('prediction_sha256')
        if e['kind']=='standard' and expected:
            actual=load(src/'prediction_manifest.json').get('predictions_sha256')
            if actual!=expected:
                excluded.append(dict(source=str(rel),reason='Prediction array identity differs from the final manuscript receipt',
                                     expected_prediction_sha256=expected,actual_prediction_sha256=actual))
                continue
        # Keep one identical delivery per case; reject conflicting alternatives.
        primary = 'per_window.csv' if e['kind'] == 'sensitivity' else 'by_sample.csv'
        if key in chosen:
            if sha(src / primary) != sha(chosen[key] / primary):
                raise ValueError('Different verified results for the same model/split; select explicitly: ' + str(key))
            continue
        chosen[key] = src
    # Resolve all candidate ambiguities before writing evaluation files.
    for (model,split,kind),src in chosen.items():
        dst = (layout.sensitivity(model,split) if kind=='sensitivity' else layout.evaluation(model,split))
        for p in src.iterdir():
            if p.is_file() and str(p.relative_to(folder)) in ledger: copy(p, dst / p.name)
    dump(layout.out / 'cloud_import.json', dict(cloud_collection_complete=report['complete'],
         imported=added,excluded_nonfinal_deliveries=excluded,
         missing_from_collection=report.get('missing', []), model_execution=False))
    return len(added)
