"""Infer frozen MAG run_03 for six validation windows and compare with ETAS/PI.

Use the existing ConvLSTM Python environment (PyTorch + NumPy). ETAS forecasts
are supplied as reference maps, so no ETAS/scientific dependencies are required.
"""
import argparse
from pathlib import Path
import platform
import zipfile

import numpy as np
import common_six as common
import core_one_window as core


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', type=Path, default=Path('/share/home/22460734/project_MP_1978_2024'))
    ap.add_argument('--data', type=Path)
    ap.add_argument('--training-dir', type=Path)
    ap.add_argument('--run', type=Path)
    ap.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    ap.add_argument('--dry-run', action='store_true', help='Check model/data/reference identities; no prediction or output')
    ap.add_argument('--out', type=Path, default=common.BASE/'results/MAG_ETAS_PI_validation_six')
    args = ap.parse_args()
    frozen, training, data, scorer, windows, area = common.load_inputs(args)
    run, cfg = core.locate_run(args.project_root.resolve(), args.run, frozen)
    plan = {'run':str(run),'selected_rows':list(common.ROWS),'windows':windows,
            'checkpoint_sha256':frozen['model']['checkpoint_sha256'],
            'selected_channels':[0],'input_history_days':4050,'target_days':450}
    print('Frozen MAG run:',run,flush=True)
    for w in windows:
        print(f'row {w["row"]}: {w["target_start"]} to {w["target_end"]}',flush=True)
    if args.dry_run:
        print('Dry run passed; no inference performed.',flush=True)
        return
    out = args.out.resolve()
    archive = out.parent/(out.name+'.zip')
    for protected in (data.root,training,run,common.BASE/'reference'):
        if out==protected or out in protected.parents or protected in out.parents:
            raise ValueError('Output overlaps protected source: '+str(protected))
    if out.exists() or archive.exists():
        raise FileExistsError('Results already exist; use a new --out directory')
    import torch
    from model_cl10 import build_model
    device = torch.device(args.device if args.device!='auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = build_model(1).to(device)
    checkpoint = torch.load(run/'best.pt',map_location=device,weights_only=False)
    if checkpoint['run_config']!=cfg or checkpoint['epoch']!=frozen['model']['best_epoch']:
        raise ValueError('Checkpoint metadata differs from frozen run')
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    del checkpoint
    out.mkdir(parents=True,exist_ok=False)
    core.write_json(out/'manifest.json',{'complete':False,**plan})
    (out/'maps').mkdir()
    details,protocols,native_checks,input_hashes,raw_ranges = [],[],[],{},{}
    for w in windows:
        index = w['row']; row=data.rows[index]
        x = np.zeros((1,9,1,340,200),np.float32)
        for j in range(9):
            keys,values=data._load_frame(row[f'x{j}_start'])
            x[0,j,0].flat[keys]=values[:,0]
            name=row[f'x{j}_start']+'.npz'
            if name not in input_hashes:
                input_hashes[name]=core.sha(data.root/'frames'/name)
        x/=255
        with torch.no_grad():
            prediction=model(torch.from_numpy(x).to(device))
        if tuple(prediction.shape)!=(1,1,340,200) or not torch.isfinite(prediction).all():
            raise ValueError(f'Invalid model prediction at row {index}')
        raw=prediction[0,0].detach().cpu().numpy().astype(np.float32)
        # Save the raw MAG map before target retrieval or scoring.
        np.savez_compressed(out/'maps'/f'MAG_{index:05d}.npz',raw_prediction=raw,row=np.array(index),
            target_start=np.array(row['target_start']),target_end=np.array(row['target_end']),
            checkpoint_sha256=np.array(frozen['model']['checkpoint_sha256']))
        raw_ranges[str(index)]=[float(raw.min()),float(raw.max())]
        maps=common.reference_maps(w)
        maps['ConvLSTM_MAG']=raw
        d,p,target=common.analyze_window(w,maps,data,scorer,area)
        details.extend(d);protocols.extend(p)
        native,_=core.score_alarms(scorer,target,area,[('ConvLSTM_MAG','original_gray_threshold',100,raw*255>=100)])
        old=next(r for r in frozen['archived_nonoverlap_windows'] if int(r['row'])==index)
        native_checks.append({'row':index,'target_start':row['target_start'],'target_end':row['target_end'],
            **{k:native[0][k] for k in ('precision','recall','CSI')},
            **{'archived_'+k:float(old[k]) for k in ('precision','recall','CSI')},
            'matches_within_1e_6':all(np.isclose(native[0][k],float(old[k]),rtol=0,atol=1e-6) for k in ('precision','recall','CSI'))})
        np.savez_compressed(out/'maps'/f'comparison_{index:05d}.npz',
            ETAS_probability=maps['ETAS'],PI_score=maps['PI'],ConvLSTM_MAG_raw=raw,
            target_normalized_MAG=target,cell_area_km2=area,row=np.array(index),
            target_start=np.array(row['target_start']),target_end=np.array(row['target_end']))
        print(f'Inferred and scored {len(native_checks)}/6 windows; row {index}',flush=True)
    summary=common.write_reports(out,details,protocols)
    core.write_csv(out/'native_MAG_window_check.csv',native_checks)
    all_match=all(r['matches_within_1e_6'] for r in native_checks)
    outputs={str(p.relative_to(out)):core.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    core.write_json(out/'manifest.json',{'complete':True,**plan,'device':str(device),
        'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),
        'training_executed':False,'MAG_inference_window_count':6,'model_count':3,
        'detail_rows':len(details),'protocol_rows':len(protocols),'summary_rows':len(summary),
        'all_archived_native_scores_match':bool(all_match),'input_frame_sha256':input_hashes,
        'raw_output_ranges':raw_ranges,'dataset_sha256':frozen['dataset_sha256'],
        'shared_code_sha256':frozen['code_sha256'],
        'reference_manifest_sha256':core.sha(common.BASE/'reference/reference_manifest.json'),
        'exporter_sha256':core.sha(Path(__file__)),'common_sha256':core.sha(common.BASE/'common_six.py'),
        'core_sha256':core.sha(common.BASE/'core_one_window.py'),'output_sha256':outputs})
    temporary=archive.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
    temporary.replace(archive)
    print('Completed. Download only:',archive,flush=True)
    if not all_match:
        print('NOTICE: some native scores differ from archive; return zip for inspection.',flush=True)


if __name__=='__main__':
    main()
