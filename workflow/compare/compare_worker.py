"""One isolated training or prediction worker. Run through run_compare.py."""
import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from compare_common import (HERE, frozen, plan, read_json, sha, write_json,
                            source_run, fingerprint, intervention, task_run, completed_run)


def train(task, project, data, out, device):
    import hashlib
    from compare_data import dataset_class
    channels = task['reference']['selected_channels']
    module = importlib.import_module('train_cl10_mag_plus_one' if len(channels) == 2
                                     else 'train_cl10_mag_plus_two')
    source = source_run(project, task['reference']['model'], task['reference']['seed'])
    original = read_json(source / 'run.json')
    expected = intervention(task, project)
    run = task_run(out, task)
    if completed_run(run, expected):
        print('Verified completed training:', run, flush=True)
        return
    module.CL10Data = dataset_class(task['group'])
    old_config, old_model = module.create_run_config, module.build_model
    holder = {}

    def create_config(*a, **kw):
        config = old_config(*a, **kw)
        if fingerprint(config) != fingerprint(original):
            raise ValueError('Training conditions differ from real-feature model')
        if config['selected_channels'] != original['selected_channels']:
            raise ValueError('Control channels differ from its source')
        config['input_intervention'] = expected
        holder['config'] = config
        return config

    def build_model(count):
        model = old_model(count)
        h = hashlib.sha256()
        for key, value in model.state_dict().items():
            h.update(key.encode('utf-8'))
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        actual = h.hexdigest()
        recorded = original.get('initialization', {}).get('model_initial_sha256')
        if recorded is not None and actual != recorded:
            raise ValueError('Initial model weights differ from real-feature reference')
        holder['config']['compare_initial_sha256'] = actual
        return model

    module.create_run_config, module.build_model = create_config, build_model
    args = SimpleNamespace(channels=','.join(map(str, channels)), out=str(run),
        baseline_run=str(project / 'runs/mag_w0_000175' / task['reference']['seed']),
        data=str(data), device=device, resume=run.exists(), smoke=False)
    module.train(args)
    write_json(run / 'training_done.json', {
        'intervention': expected, 'initial_sha256': holder['config']['compare_initial_sha256'],
        'files_sha256': {n: sha(run / n) for n in
                         ('run.json', 'summary.json', 'best.pt', 'history.csv', 'train_mask.npz')}})


def predict(task, project, data, out, device, split):
    from compare_data import dataset_class
    import predict_cl10
    run = task_run(out, task)
    expected = intervention(task, project)
    if not completed_run(run, expected):
        raise ValueError('Complete and verify training first')
    pred = out / 'evaluations' / task['task_id'] / split / 'predictions'
    marker = pred / 'manifest.json'
    if pred.exists():
        if not marker.is_file():
            raise ValueError(f'Incomplete prediction export preserved: {pred}')
        m = read_json(marker)
        if not m.get('complete') or m.get('input_intervention') != expected:
            raise ValueError('Existing predictions use a different input transform')
        if m.get('best_checkpoint_sha256') != sha(run / 'best.pt') or m.get('split') != split:
            raise ValueError('Existing predictions use a different model or split')
        if sha(pred / 'predictions.npy') != m.get('predictions_sha256'):
            raise ValueError('Prediction array hash mismatch')
        return
    predict_cl10.CL10Data = dataset_class(task['group'])
    predict_cl10.predict(SimpleNamespace(run=str(run), data=str(data), split=split,
                                        out=str(pred), device=device))
    m = read_json(marker)
    m['input_intervention'] = expected
    m['run_json_sha256'] = sha(run / 'run.json')
    write_json(marker, m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('train', 'predict'), required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda', choices=('auto', 'cpu', 'cuda'))
    p.add_argument('--split', choices=('validation', 'test'), default='validation')
    a = p.parse_args()
    f = frozen()
    tasks, _ = plan(f)
    task = tasks[a.task]
    sys.path.insert(0, str(HERE / 'vendor'))
    if a.action == 'train':
        train(task, a.project, a.data, a.out, a.device)
    else:
        predict(task, a.project, a.data, a.out, a.device, a.split)


if __name__ == '__main__':
    main()
