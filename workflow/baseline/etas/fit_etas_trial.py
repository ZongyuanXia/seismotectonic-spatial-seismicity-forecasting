"""Fit the author's spatial ETAS to pre-origin Japan events only; one-window pilot."""
from pathlib import Path
import argparse, hashlib, json, logging, pickle, sys
import numpy as np
import pandas as pd
from scipy.optimize import NonlinearConstraint

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'vendor/etas-main'))
import etas.inversion as inv

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=BASE/'results/validation_00000_20090101_v1')
    args=ap.parse_args(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'fit.pkl').exists():
        raise FileExistsError('Fit already exists; do not overwrite it.')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(out/'fit.log'),logging.StreamHandler()])
    np.random.seed(20261029)
    data=args.root/'datasets/cl10_mapping_v2'
    if not data.exists(): data=args.root/'cl10_mapping_v2'
    manifest=json.loads((data/'manifest.json').read_text())
    region_path=BASE/'region_meta.json'
    if not region_path.exists():region_path=Path(manifest['region_path'])
    region=json.loads(region_path.read_text())
    events=pd.read_csv(data/'usgs_event_audit.csv')
    events['time']=pd.to_datetime(events['time'],format='ISO8601',utc=True).dt.tz_localize(None)
    origin=pd.Timestamp('2009-01-01'); aux=pd.Timestamp('1978-01-01'); fitstart=pd.Timestamp('1989-02-02')
    # Critical: discard target/future rows before any fit, beta estimate, or diagnostics.
    history=events.loc[(events.time>=aux)&(events.time<origin)].copy()
    assert history.id.is_unique and (history.MW>=4).all() and history.time.max()<origin
    catalog=history.rename(columns={'MW':'magnitude'}).set_index('id')[['latitude','longitude','time','magnitude']]
    catalog=catalog.sort_values('time'); catalog.to_csv(out/'historical_catalog.csv')
    b=region['effective_bounds']; shape=[[b['lat_min'],b['lon_min']],[b['lat_max'],b['lon_min']],
        [b['lat_max'],b['lon_max']],[b['lat_min'],b['lon_max']]]
    primary=catalog.loc[catalog.time>=fitstart]
    beta=float(inv.estimate_beta_tinti(primary.magnitude.to_numpy(),4,delta_m=0))
    # Trial bandwidth 25 km, not selected from this validation target.
    theta={'log10_mu':-7.3,'log10_k0':0.,'a':.6*beta+1.,'log10_c':-2.,
        'omega':.15,'log10_tau':3.,'log10_d':1.4,'gamma':1.,'rho':1.}
    theta['log10_k0']=float(np.log10(.3/inv.branching_ratio(inv.parameter_dict2array(theta),beta)))
    config={'id':'japan_one_window_trial_20261029','name':'Japan ETAS single-origin pilot',
        'fn_catalog':str(out/'historical_catalog.csv'),'auxiliary_start':str(aux),
        'timewindow_start':str(fitstart),'timewindow_end':str(origin),'mc':4.,'delta_m':0.,
        'coppersmith_multiplier':100,'earth_radius':6378.1,'shape_coords':shape,
        'free_background':True,'free_productivity':False,'bw_sq':625.,'theta_0':theta,
        'beta':beta,'three_dim':False}
    (out/'fit_config.json').write_text(json.dumps(config,indent=2)+'\n')
    obj=inv.ETASParameterCalculation(config); obj.prepare()
    # Stationary trial: alpha=a-rho*gamma < beta and mean offspring ratio <= .99.
    # Constraints apply to optimizer's 8 triggering parameters; mu is updated by EM.
    def alpha(x): return x[1]-x[7]*x[6]
    def br(x):
        full=np.array([-7.,None,*x],dtype=object)
        if alpha(x)>=beta: return 1e6+(alpha(x)-beta)*1e6
        return float(inv.branching_ratio(full,beta))
    obj.constraints=[NonlinearConstraint(alpha,-np.inf,beta-.01),NonlinearConstraint(br,0.,.99)]
    original_minimize=inv.minimize; steps=[]
    def tracked_minimize(*a,**kw):
        kw['tol']=1e-9; kw['options']={'maxiter':500,'ftol':1e-9}
        res=original_minimize(*a,**kw)
        steps.append({'success':bool(res.success),'message':str(res.message),'nit':int(res.nit),
            'objective':float(res.fun),'branching_ratio':br(res.x),'alpha':alpha(res.x)})
        (out/'optimizer_steps.json').write_text(json.dumps(steps,indent=2)+'\n')
        if not np.isfinite(res.fun): raise RuntimeError('Nonfinite ETAS M-step objective')
        return res
    inv.minimize=tracked_minimize
    original_expectation=obj.expectation_step; nstep=[0]
    def limited_expectation(*a,**kw):
        nstep[0]+=1
        if nstep[0]>151: raise RuntimeError('ETAS did not converge within 150 EM iterations')
        return original_expectation(*a,**kw)
    obj.expectation_step=limited_expectation
    fitted=obj.invert()
    obj.expectation_step=original_expectation
    inv.minimize=original_minimize
    eta=float(inv.branching_ratio(inv.parameter_dict2array(fitted),obj.beta))
    assert np.isfinite(eta) and 0<=eta<=.99001 and obj.inversion_done
    # Persist only numerical state/data tables, never local optimizer callbacks.
    state={'theta':fitted,'catalog':obj.catalog,'target_events':obj.target_events,
        'source_events':obj.source_events,'beta':float(obj.beta),'origin':str(origin),
        'shape_coords':shape,'background_bandwidth_km':25.,'EM_iterations':obj.i}
    temporary=out/'fit.pkl.tmp'
    with temporary.open('wb') as f:pickle.dump(state,f)
    temporary.replace(out/'fit.pkl')
    info={'complete':True,'trial_only':True,'origin':str(origin),'target_end_exclusive':str(origin+pd.Timedelta(days=450)),
        'history_start_for_triggering':str(origin-pd.Timedelta(days=4050)),
        'fit_target_count':len(obj.target_events),'fit_source_count':len(obj.source_events),
        'training_min_time':str(catalog.time.min()),'training_max_time':str(catalog.time.max()),
        'theta':fitted,'beta':float(obj.beta),'b_value':float(obj.beta/np.log(10)),
        'branching_ratio':eta,'EM_iterations':obj.i,'background_bandwidth_km':25.,
        'optimizer_failed_steps':sum(not s['success'] for s in steps),
        'completeness':'Current frozen converted-Mw>=4 catalog; completeness assumption not yet established. Pilot, not final baseline.',
        'boundary':'Observed parents restricted to Japan; author spatial likelihood uses infinite-plane trigger integral. Boundary effects remain a limitation.',
        'sha256':{'event_audit':sha(data/'usgs_event_audit.csv'),'dataset_manifest':sha(data/'manifest.json'),
            'sample_index':sha(data/'sample_index.csv'),'fit_script':sha(__file__),
            'author_archive':sha(BASE/'vendor/etas_main.tar.gz'),'author_inversion':sha(BASE/'vendor/etas-main/etas/inversion.py')}}
    (out/'fit_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__': main()
