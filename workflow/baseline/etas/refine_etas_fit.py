"""Numerical polishing of the same constrained fit, using training data only.

The upstream SLSQP tolerance 1e-9 gave false-status line-search exits at the
active branching constraint. Retry from the fitted state with absolute 1e-6
tolerance, retain complete diagnostics, and require every final M-step success.
"""
import argparse,json,pickle,shutil,sys,logging
from pathlib import Path
import numpy as np
from scipy.optimize import NonlinearConstraint
import fit_etas_trial as f
inv=f.inv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--from-fit',type=Path,default=f.BASE/'results/validation_00000_20090101_v1')
    ap.add_argument('--out',type=Path,default=f.BASE/'results/validation_00000_20090101_v2')
    args=ap.parse_args();old=args.from_fit;out=args.out
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(out/'fit.log'),logging.StreamHandler()])
    parent=json.loads((old/'fit_manifest.json').read_text())
    with (old/'fit.pkl').open('rb') as h:state=pickle.load(h)
    config=json.loads((old/'fit_config.json').read_text())
    shutil.copy2(old/'historical_catalog.csv',out/'historical_catalog.csv')
    config['fn_catalog']=str((out/'historical_catalog.csv').resolve());config['theta_0']=state['theta']
    (out/'fit_config.json').write_text(json.dumps(config,indent=2)+'\n')
    obj=inv.ETASParameterCalculation(config);obj.prepare()
    obj.target_events=state['target_events'].copy();obj.source_events=state['source_events'].copy()
    beta=obj.beta
    def alpha(x):return x[1]-x[7]*x[6]
    def br(x):
        if alpha(x)>=beta:return 1e6+(alpha(x)-beta)*1e6
        return float(inv.branching_ratio(np.array([-7.,None,*x],dtype=object),beta))
    obj.constraints=[NonlinearConstraint(alpha,-np.inf,beta-.01),NonlinearConstraint(br,0.,.99)]
    original=inv.minimize;steps=[]
    def tracked(*a,**kw):
        kw['tol']=1e-6;kw['options']={'maxiter':1000,'ftol':1e-6}
        res=original(*a,**kw)
        steps.append({'success':bool(res.success),'message':str(res.message),'nit':int(res.nit),
            'objective':float(res.fun),'branching_ratio':br(res.x),'alpha':alpha(res.x)})
        (out/'optimizer_steps.json').write_text(json.dumps(steps,indent=2)+'\n')
        if not res.success or not np.isfinite(res.fun) or br(res.x)>.990001:
            raise RuntimeError('Refined ETAS M-step not accepted: '+str(steps[-1]))
        if len(steps)>30:raise RuntimeError('Polishing did not converge within 30 iterations')
        return res
    inv.minimize=tracked;theta=obj.invert();inv.minimize=original
    state.update(theta=theta,target_events=obj.target_events,source_events=obj.source_events,
        catalog=obj.catalog,EM_iterations=parent['EM_iterations']+obj.i)
    with (out/'fit.pkl').open('wb') as h:pickle.dump(state,h)
    info={**parent,'theta':theta,'branching_ratio':float(inv.branching_ratio(inv.parameter_dict2array(theta),beta)),
        'EM_iterations':state['EM_iterations'],'refinement_EM_iterations':obj.i,
        'optimizer_failed_steps':0,'initial_fit_optimizer_warnings':parent['optimizer_failed_steps'],
        'optimizer_tolerance_refinement':1e-6,'parent_fit_manifest_sha256':f.sha(old/'fit_manifest.json'),
        'sha256':{**parent['sha256'],'refinement_script':f.sha(__file__)}}
    (out/'fit_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
