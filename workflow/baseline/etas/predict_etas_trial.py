"""One causal 450-day ETAS forecast using the authors' all-generation simulator.

The local adapter samples the inferred background mixture with exact normalized
weights and a 25 km Gaussian, without the upstream function's entropy reseed.
All triggering kernels and generation recursion remain the upstream algorithms.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, pickle, sys, warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'vendor/etas-main'))
import etas.simulation as simulation
from etas.inversion import expected_aftershocks, parameter_dict2array

CTX={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(p,rows):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def exact_background(latitudes,longitudes,background_probs,scale=None,grid=False,bsla=None,bslo=None,n=1):
    """Weighted background epicentres, then 25 km N/S and E/W Gaussian offsets.

    Locations are conditioned on falling inside the Japan polygon. This avoids
    the upstream insufficient-draw fallback to a spatially uniform background.
    """
    assert not grid
    weights=np.asarray(background_probs,dtype=float); weights=weights/weights.sum()
    lat=np.asarray(latitudes);lon=np.asarray(longitudes);a,b,c,d=CTX['bounds']
    outlat=[];outlon=[];remaining=n;radius=6378.1
    while remaining:
        count=max(32,remaining*2)
        idx=np.random.choice(len(lat),size=count,p=weights)
        la=lat[idx]+np.degrees(np.random.normal(0,25.,count)/radius)
        lo=lon[idx]+np.degrees(np.random.normal(0,25.,count)/(radius*np.cos(np.radians(lat[idx]))))
        ok=(la>=a)&(la<c)&(lo>=b)&(lo<d)
        la=la[ok][:remaining];lo=lo[ok][:remaining]
        outlat.extend(la);outlon.extend(lo);remaining-=len(la)
    return pd.Series(outlat),pd.Series(outlon)

def initialize(fit_path,region_path):
    with open(fit_path,'rb') as f:state=pickle.load(f)
    region=json.loads(Path(region_path).read_text());b=region['effective_bounds']
    bounds=(b['lat_min'],b['lon_min'],b['lat_max'],b['lon_max'])
    origin=pd.Timestamp(state['origin']);history_start=origin-pd.Timedelta(days=4050)
    history=state['catalog'].loc[(state['catalog'].time>=history_start)&(state['catalog'].time<origin)].copy()
    assert (history.time<origin).all() and history.magnitude.min()>=4
    history['xi_plus_1']=1.
    CTX.update(state=state,history=history,origin=origin,end=origin+pd.Timedelta(days=450),
        history_start=history_start,bounds=bounds,region=region,polygon=Polygon(state['shape_coords']))
    simulation.simulate_background_location=exact_background
    logging=__import__('logging');logging.getLogger('etas.simulation').setLevel(logging.ERROR)
    warnings.filterwarnings('ignore',category=FutureWarning)

def grid_keys(lat,lon,region):
    gx=np.floor((np.asarray(lon)+180)/360*region['global_W']).astype(np.int64)
    gy=np.floor((np.asarray(lat)+90)/180*region['global_H']).astype(np.int64)
    c=region['crop_box_closed'];x=gx-c['x_min'];y=gy-c['y_min']
    ok=(x>=0)&(x<200)&(y>=0)&(y<340)
    return (y[ok]*200+x[ok]).astype(np.int32),ok

def generate_one(index):
    c=CTX;s=c['state'];seed=202610290+int(index)
    np.random.seed(seed)
    # All generations included; outside-Japan simulated events may trigger back inside.
    catalog=simulation.simulate_catalog_continuation(c['history'],c['history_start'],c['origin'],
        c['polygon'],c['end'],s['theta'],4.,s['beta'],delta_m=0,m_max=None,
        background_lats=s['target_events'].latitude,background_lons=s['target_events'].longitude,
        background_probs=s['target_events'].P_background,gaussian_scale=.1,
        filter_polygon=False,approx_times=False)
    future=catalog.loc[(catalog.time>=c['origin'])&(catalog.time<c['end'])&(catalog.magnitude>=4)].copy()
    keys,ok=grid_keys(future.latitude,future.longitude,c['region'])
    if len(future)>1000000:raise RuntimeError('Excessively large simulation: no truncated/censored result accepted')
    return index,np.unique(keys),keys,{'simulation':int(index),'seed':seed,
        'events_in_region':int(len(keys)),'occupied_cells':int(len(np.unique(keys))),
        'events_all_locations':len(future),'maximum_generation':int(future.generation.max()) if len(future) else 0,
        'maximum_magnitude_in_region':float(future.loc[ok,'magnitude'].max()) if len(keys) else None}

def areas(region):
    c=region['crop_box_closed'];edges=(np.arange(341)+c['y_min'])/region['global_H']*180-90
    perrow=6378.1**2*np.radians(360/region['global_W'])*np.diff(np.sin(np.radians(edges)))
    return np.broadcast_to(perrow[:,None],(340,200)).copy()

def ranked_alarm(score,area,fraction):
    # Seeded tie order is independent of future targets; nearest prefix differs by <= one cell area.
    tie=np.random.default_rng(2901).random(score.size)
    order=np.lexsort((tie,-score.ravel().astype(float)))
    cumulative=np.cumsum(area.ravel()[order]);budget=float(area.sum()*fraction)
    hi=int(np.searchsorted(cumulative,budget));choices=[max(0,hi),min(len(order),hi+1)]
    k=min(choices,key=lambda j:abs((cumulative[j-1] if j else 0)-budget))
    alarm=np.zeros(score.size,dtype=bool);alarm[order[:k]]=True
    return alarm.reshape(score.shape)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=BASE/'results/validation_00000_20090101_v2')
    ap.add_argument('--simulations',type=int,default=2048);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();out=args.out.resolve();n=args.simulations
    if n<2 or args.workers<1:ap.error('Positive workers and at least two simulations required')
    if (out/'prediction.npz').exists():raise FileExistsError('Prediction already exists; do not overwrite it.')
    fm=json.loads((out/'fit_manifest.json').read_text());assert fm['complete']
    data=args.root/'datasets/cl10_mapping_v2'
    if not data.exists():data=args.root/'cl10_mapping_v2'
    ds=json.loads((data/'manifest.json').read_text())
    assert fm['sha256']['event_audit']==sha(data/'usgs_event_audit.csv')
    region_path=BASE/'region_meta.json'
    if not region_path.exists():region_path=Path(ds['region_path'])
    region=json.loads(region_path.read_text())
    training=args.root/'convlstm/training'
    if not training.exists():training=args.root/'training'
    sys.path.insert(0,str(training))
    import cl10_data as reader
    import evaluate_cl10_symmetric as scorer
    assert scorer.EVALUATION_VERSION=='symmetric_band_union_v1' and scorer.EXPAND_D==3 and scorer.THRESHOLD_GRAY==100
    assert fm['sha256']['dataset_manifest']==sha(data/'manifest.json')
    score_shape=(340,200);presence=np.zeros(score_shape,np.int64);eventcounts=np.zeros(score_shape,np.int64)
    half_presence=np.zeros((2,*score_shape),np.int64);records=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
         initializer=initialize,initargs=(str(out/'fit.pkl'),region_path)) as pool:
        for i,unique,keys,record in pool.map(generate_one,range(n),chunksize=1):
            presence.flat[unique]+=1;np.add.at(eventcounts.ravel(),keys,1)
            half_presence[int(i>=n//2)].flat[unique]+=1;records.append(record)
            if (i+1)%64==0 or i+1==n:print(f'ETAS simulations {i+1}/{n}',flush=True)
    probability=(presence/n).astype(np.float32);mean_count=(eventcounts/n).astype(np.float32)
    assert np.all((probability>=0)&(probability<=1)) and np.isfinite(mean_count).all()
    # Commit predictions BEFORE reading this window's true target.
    np.savez_compressed(out/'prediction.npz',probability=probability,expected_event_count=mean_count,
        presence_count=presence,event_count=eventcounts,half_presence_count=half_presence,
        origin=np.array('2009-01-01'),end_exclusive=np.array('2010-03-27'),simulations=np.array(n))
    prediction_hash=sha(out/'prediction.npz');writecsv(out/'simulation_counts.csv',records)
    with (out/'prediction_manifest.json').open('w') as f:json.dump({'complete':True,'trial_only':True,
        'simulations':n,'seed_formula':'202610290 + zero_based_simulation_index',
        'forecast_days':450,'origin':'2009-01-01','end_exclusive':'2010-03-27',
        'prediction_sha256':prediction_hash,'fit_manifest_sha256':sha(out/'fit_manifest.json'),
        'script_sha256':sha(__file__),'author_simulation_sha256':sha(BASE/'vendor/etas-main/etas/simulation.py'),
        'background_adapter':'normalized P_background weighted Gaussian mixture; 25 km; rejection inside Japan; seeded RNG',
        'history_trigger_days':4050,'all_generations':True,'magnitude_maximum':'unbounded Gutenberg-Richter',
        'evaluation_status':'Not yet evaluated; target has not been loaded by this stage'},f,indent=2)
    d=reader.CL10Data(data,'validation',(0,),cache_size=0);row=d.rows[0]
    assert row['target_start']=='2009-01-01' and row['target_end']=='2010-03-26'
    keys,gray=d.target_keys(row);target=np.zeros(score_shape,np.float32);target.flat[keys]=gray/255.
    area=areas(region);outputs=[];protocols=[]
    def evaluate(label,alarm,setting,value):
        expanded=scorer.archived_expand(alarm.astype(np.float32));truth_exp=scorer.archived_expand(target)
        a={}
        for protocol,pe,te in scorer.PROTOCOLS:
            metrics=scorer.counts(expanded if pe else alarm.astype(np.float32),truth_exp if te else target)
            protocols.append({'model':label,'setting':setting,'value':value,'protocol':protocol,**metrics});a[protocol]=metrics
        outputs.append({'model':label,'setting':setting,'value':value,'sample_count':1,
            'precision':a['target_only']['precision'],'recall':a['prediction_only']['recall'],'CSI':a['both']['CSI'],
            'precision_exact':a['neither']['precision'],'recall_exact':a['neither']['recall'],
            'alarm_cells':int(alarm.sum()),'alarm_grid_fraction':float(alarm.mean()),
            'alarm_km2':float(area[alarm].sum()),'alarm_area_fraction':float(area[alarm].sum()/area.sum()),
            'expanded_alarm_area_fraction':float(area[expanded>0].sum()/area.sum()),'target_cells':len(keys)})
    # Predefined descriptive thresholds, never selected for best target performance.
    for threshold in (.001,.005,.01,.02,.05,.1):evaluate('ETAS',probability>=threshold,'probability_threshold',threshold)
    pi_dir=args.root/'baseline/results/pi_japan_4050d_change450d_positive_v1/validation'
    if not pi_dir.exists():pi_dir=BASE/'reference/pi_validation'
    with np.load(pi_dir/'maps/sample_00000_2009-01-01.npz') as z:pi=z['pi_map']
    pi_manifest=json.loads((pi_dir/'manifest.json').read_text())
    assert pi_manifest['dataset_sha256']['manifest']==sha(data/'manifest.json')
    for name,h in pi_manifest['shared_code_sha256'].items():assert sha(training/name)==h
    evaluate('PI',pi>0,'original_PI_positive',0)
    for fraction in (.005,.01,.02,.03):
        for label,score in [('ETAS',probability),('PI',pi)]:evaluate(label,ranked_alarm(score,area,fraction),'equal_actual_area',fraction)
    writecsv(out/'evaluation.csv',outputs);writecsv(out/'evaluation_protocols.csv',protocols)
    stability=[]
    for fraction in (.005,.01,.02,.03):
        a=ranked_alarm(half_presence[0]/(n//2),area,fraction);b=ranked_alarm(half_presence[1]/(n-n//2),area,fraction)
        stability.append({'area_fraction':fraction,'alarm_jaccard_two_halves':float((a&b).sum()/(a|b).sum())})
    diagnostics={'simulations':n,'events_mean':float(np.mean([x['events_in_region'] for x in records])),
        'events_median':float(np.median([x['events_in_region'] for x in records])),
        'events_p05_p95':np.percentile([x['events_in_region'] for x in records],[5,95]).tolist(),
        'events_max':max(x['events_in_region'] for x in records),'observed_events':int(np.sum((d.day>= (pd.Timestamp(row['target_start']).date()-reader.START).days)&(d.day<= (pd.Timestamp(row['target_end']).date()-reader.START).days))),
        'observed_occupied_cells':len(keys),'probability_max':float(probability.max()),
        'expected_occupied_cells':float(probability.sum()),'split_half_stability':stability,
        'MC_standard_error_max':float(np.sqrt(probability*(1-probability)/n).max()),
        'formal_convergence_claim':False,'prediction_sha256_unchanged_after_evaluation':sha(out/'prediction.npz')==prediction_hash}
    (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps({'diagnostics':diagnostics,'evaluation':outputs},indent=2),flush=True)

if __name__=='__main__':main()
