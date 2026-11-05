"""Train and evaluate fixed-seed A=0/B=100/C=255 controls.

Examples: --dry-run; --models pair_04_08 --device cuda; --device cuda.
Original real-feature checkpoints are read for verification, never retrained.
"""
import argparse
import fcntl
import subprocess
import sys
from pathlib import Path

from compare_common import (HERE, DEFAULT_PROJECT, DEFAULT_OUTPUT, PIN, frozen, plan,
                            preflight, read_json, write_json, software_identity,
                            task_run, intervention, completed_run, source_run)


def command(arguments, log):
    print('Running:', ' '.join(map(str, arguments)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as handle:
        with subprocess.Popen(list(map(str, arguments)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                print(line, end='', flush=True)
            code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, arguments)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=DEFAULT_PROJECT)
    p.add_argument('--data', type=Path, help='Default project/cl10_mapping_v2')
    p.add_argument('--out', type=Path, help='Default project/runs/compare_abc_v1')
    p.add_argument('--models', nargs='+', help='Default all 19 fixed combinations')
    p.add_argument('--groups', nargs='+', choices=('A', 'B', 'C'), default=['A', 'B', 'C'])
    p.add_argument('--stage', choices=('all', 'train', 'evaluate', 'summary'), default='all')
    p.add_argument('--splits', nargs='+', choices=('validation', 'test'), default=['validation', 'test'])
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--dry-run', action='store_true', help='Verify inputs; no training, scoring or output')
    a = p.parse_args()
    f = frozen()
    tasks, aliases = plan(f)
    selected = a.models or list(aliases)
    if len(selected) != len(set(selected)) or set(selected) - set(aliases):
        p.error('Unknown or duplicate model names')
    if len(a.groups) != len(set(a.groups)) or len(a.splits) != len(set(a.splits)):
        p.error('Duplicate group or split')
    selected_ids = {aliases[m][g] for m in selected for g in a.groups}
    chosen = [t for key, t in tasks.items() if key in selected_ids]
    project = a.project_root.resolve()
    data = a.data.resolve() if a.data else project / 'cl10_mapping_v2'
    out = a.out.resolve() if a.out else project / DEFAULT_OUTPUT
    sources = [data, HERE, *(source_run(project, m['model'], m['seed']).resolve() for m in f['models'])]
    for source in sources:
        if out == source or source in out.parents or out in source.parents:
            raise ValueError(f'Output overlaps inputs: {source}')
    preflight(project, data, f, tasks, aliases)
    print(f'{len(selected) * len(a.groups)} logical controls; {len(chosen)} unique tasks selected; '
          f'{len(tasks)} total unique tasks. Output={out}', flush=True)
    for task in chosen:
        m = task['reference']
        print(task['task_id'], m['selected_channels'], m['seed'], m['init_seed'], m['order_seed'], flush=True)
    if a.dry_run:
        print('Preflight passed; no model execution or outputs.', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'version': 'compare_abc_v1', 'selection_sha256': PIN,
                    'compare_code_sha256': software_identity(),
                    'vendor_sha256': f['vendor_sha256'], 'dataset_sha256': f['dataset_sha256'],
                    'project_root': str(project), 'data_root': str(data),
                    'all_model_names': list(aliases), 'unique_training_tasks': len(tasks),
                    'groups': {'A': 0, 'B': 100, 'C': 255},
                    'seed_selection': 'fixed original validation selection; no reselection',
                    'training_selection': 'lowest full validation weighted loss; original early stopping'}
        marker = out / 'experiment_manifest.json'
        if marker.exists():
            if read_json(marker) != identity:
                raise ValueError('Experiment code or inputs changed; use a new --out')
        elif any(path.name != '.run.lock' for path in out.iterdir()):
            raise ValueError('Unrecognized nonempty output directory')
        else:
            write_json(marker, identity)
            write_json(out / 'task_aliases.json', aliases)
        if a.stage != 'summary':
            sys.path.insert(0, str(HERE / 'vendor'))
            import run_symmetric_evaluation as engine
            for task in chosen:
                args = [sys.executable, '-u', HERE / 'compare_worker.py', '--task', task['task_id'],
                        '--project', project, '--data', data, '--out', out, '--device', a.device]
                if a.stage in ('all', 'train'):
                    command(args + ['--action', 'train'], out / 'logs' / (task['task_id'] + '_train.log'))
                if a.stage in ('all', 'evaluate'):
                    for split in a.splits:
                        command(args + ['--action', 'predict', '--split', split],
                                out / 'logs' / (task['task_id'] + '_' + split + '.log'))
                        run = task_run(out, task)
                        if not completed_run(run, intervention(task, project)):
                            raise ValueError('Incomplete training')
                        result_root = out / 'evaluations' / task['task_id'] / split
                        pred = result_root / 'predictions'
                        case = engine.Case(task['task_id'], task['reference']['seed'], run,
                                           tuple(task['reference']['selected_channels']), pred)
                        # Shared engine writes cases/MODEL/SEED/evaluation; use the
                        # fixed scoring output directly here to keep one case tree.
                        target = result_root / 'evaluation'
                        expected = engine.evaluation_identity(case, data, split, HERE / 'vendor')
                        if target.exists():
                            engine.verify_evaluation(target, expected)
                        else:
                            command([sys.executable, '-u', HERE / 'vendor/evaluate_cl10_symmetric.py',
                                     '--data', data, '--run', run, '--predictions', pred,
                                     '--split', split, '--out', target],
                                    out / 'logs' / (task['task_id'] + '_' + split + '_score.log'))
                            engine.verify_evaluation(target, expected)
        from summarize_compare import collect
        collect(project, out, f, tasks, aliases)


if __name__ == '__main__':
    main()
