"""Build the Japan MAG + nine WSM feature frame dataset (no model training).

Sparse frames and a dated sample index can be loaded directly by CLDataset.
The target is generated from USGS MAG only. Files are checkpointed by frame.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"shared"))
from core import (FEATURES, GridDirectionCache, HistoryIndex, features, gray_linear,
                  gray_mag, haversine, angular_bounds, load_usgs_months, load_wsm, sha256)

DEFAULT_WSM=Path("/Users/xiazongyuan/1 课题/4 全球地震预测项目资料汇总/Phase1/世界应力数据/wsm2025/WSM_Database_2025.csv")
DEFAULT_USGS=Path("/Users/xiazongyuan/Documents/ChatGPT/论文撰写/usgs_1973_2024_refresh/monthly_csv")
DEFAULT_REGION=Path("/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/第二阶段-ConvLSTM模型/data_mag_wsm7c_100km_jp/region_meta.json")
CHANNELS=("MAG","DEPTH_LOCAL_MEAN","DEPTH_LOCAL_STD","MAX_SPREAD_100KM_DEG","STD_LOCAL",
          "REGIME_ENTROPY","REGIME_GINI","MEDIAN_LOCAL_DEPTH","CONSISTENCY_R_MEAN","MAX_ROTATION_DEG")
PHYSICS=("MEAN_LOCAL_DEPTH","STD_LOCAL_DEPTH","MAX_SPREAD_100KM_DEG","STD_LOCAL",
         "REGIME_ENTROPY","REGIME_GINI","MEDIAN_LOCAL_DEPTH","CONSISTENCY_R_MEAN","MAX_ROTATION_DEG")
# Physical-value upper bounds for the nine WSM channels (km, degrees, or unitless).
# An active valid zero maps to 100; an invalid value maps to 0.
UPPER=(30.,15.,90.,75.,float(np.log(5)),.8,35.,1.,20.)
START=pd.Timestamp("1978-01-01",tz="UTC")
END=pd.Timestamp("2024-12-31",tz="UTC")
HORIZON=450
INPUT_FRAMES=9


def sample_index():
    periods=(
        ("train","1989-02-02","2008-12-31",6824),
        ("validation","2009-01-01","2016-12-31",2473),
        ("test","2017-01-01","2024-12-31",2473),
    )
    rows=[]
    for name,lo,hi,expected in periods:
        lo=pd.Timestamp(lo,tz="UTC");hi=pd.Timestamp(hi,tz="UTC")
        current=lo
        while current+pd.Timedelta(days=HORIZON-1)<=hi:
            first=current-pd.Timedelta(days=HORIZON*INPUT_FRAMES)
            if first<START:raise ValueError(f"Incomplete historical input for {current}")
            row={"split":name,"target_start":current.date().isoformat(),
                 "target_end":(current+pd.Timedelta(days=HORIZON-1)).date().isoformat()}
            for j in range(INPUT_FRAMES):
                row[f"x{j}_start"]=(first+pd.Timedelta(days=HORIZON*j)).date().isoformat()
                row[f"x{j}_end"]=(first+pd.Timedelta(days=HORIZON*(j+1)-1)).date().isoformat()
            rows.append(row)
            current+=pd.Timedelta(days=1)
        actual=len(rows)-sum(x["split"]!=name for x in rows)
        if actual!=expected:raise AssertionError((name,actual,expected))
    return pd.DataFrame(rows)


def _monthday(date_str):
    return (pd.Timestamp(date_str,tz="UTC")-START).days


def _prepare_usgs(events,region):
    x0,x1,y0,y1=[region["crop_box_closed"][k] for k in ("x_min","x_max","y_min","y_max")]
    H=int(region["global_H"]);W=int(region["global_W"])
    lat=events.latitude.to_numpy(float);lon=events.longitude.to_numpy(float)
    # Same anchored 3840 x 3840 mother-grid map as finaldata_7c_v4.py.
    gx=np.floor((lon+180)/360*W).astype(np.int32)
    gy=np.floor((lat+90)/180*H).astype(np.int32)
    keep=(gx>=x0)&(gx<=x1)&(gy>=y0)&(gy<=y1)
    e=events.loc[keep].copy()
    e["x"]=(gx[keep]-x0).astype(np.int32)
    e["y"]=(gy[keep]-y0).astype(np.int32)
    e["key"]=(e.y*(x1-x0+1)+e.x).astype(np.int32)
    e["day_index"]=(e.DAY-START).dt.days.astype(np.int32)
    e["gray"]=[gray_mag(v) for v in e.MW]
    return e.sort_values("day_index",kind="stable").reset_index(drop=True)


class Grid:
    def __init__(self,region):
        self.x0,self.x1,self.y0,self.y1=[int(region["crop_box_closed"][k]) for k in ("x_min","x_max","y_min","y_max")]
        self.W=self.x1-self.x0+1;self.H=self.y1-self.y0+1
        self.globalW=int(region["global_W"]);self.globalH=int(region["global_H"])
        x=np.arange(self.W,dtype=float)+self.x0
        y=np.arange(self.H,dtype=float)+self.y0
        self.lon=(x+.5)/self.globalW*360-180
        self.lat=(y+.5)/self.globalH*180-90
        self.candidates={}
        self.candidate_keys={}

    def center(self,key):
        y,x=divmod(int(key),self.W)
        return self.lat[y],self.lon[x]

    def direction_grid(self,key):
        key=int(key)
        if key in self.candidates:return self.candidates[key]
        y,x=divmod(key,self.W)
        la,lo=self.center(key)
        dy,dx=angular_bounds(la,100)
        ylo=max(0,int(np.searchsorted(self.lat,la-dy,side="left")))
        yhi=min(self.H,int(np.searchsorted(self.lat,la+dy,side="right")))
        xlo=max(0,int(np.searchsorted(self.lon,lo-dx,side="left")))
        xhi=min(self.W,int(np.searchsorted(self.lon,lo+dx,side="right")))
        yy,xx=np.meshgrid(np.arange(ylo,yhi),np.arange(xlo,xhi),indexing="ij")
        yy,xx=yy.ravel(),xx.ravel()
        keep=haversine(la,lo,self.lat[yy],self.lon[xx])<=100+1e-9
        yy,xx=yy[keep],xx[keep]
        cells=(yy*self.W+xx).astype(np.int32)
        target=np.flatnonzero(cells==key)
        if len(target)!=1:raise AssertionError("Center cell missing in its own 100 km circle")
        out=(self.lat[yy],self.lon[xx],int(target[0]))
        self.candidates[key]=out
        self.candidate_keys[key]=cells
        return out


class CenterCache:
    """A center only needs recomputation when its 200 km WSM history changes."""
    def __init__(self,index,grid):
        self.index=index;self.grid=grid
        self.daylists={};self.last={}
        self.direction=GridDirectionCache(index)
        self.frame_mean=None

    def _history_count(self,key,day):
        key=int(key)
        if key not in self.daylists:
            la,lo=self.grid.center(key)
            ids,_=self.index.neighbors(la,lo,200,cache_key=("center",key))
            self.daylists[key]=np.sort(self.index.day[ids])
        di=int((np.datetime64(day.date())-np.datetime64("1970-01-01")).astype(int))
        return int(np.searchsorted(self.daylists[key],di,side="right"))

    def prepare(self,keys,day):
        """Calculate each requested grid direction once for the whole frame."""
        pending=[]
        for key in keys:
            key=int(key)
            n=self._history_count(key,day)
            old=self.last.get(key)
            if old is None or old[0]!=n:
                self.grid.direction_grid(key)
                pending.append(key)
        self.frame_mean=np.full(self.grid.W*self.grid.H,np.nan,dtype=np.float32)
        if not pending:return
        query_keys=np.unique(np.concatenate([self.grid.candidate_keys[k] for k in pending]))
        for cell in query_keys:
            la,lo=self.grid.center(cell)
            mean,_,std,n=self.direction(la,lo,day,True)
            if n>=5 and np.isfinite(std) and std<=25:
                self.frame_mean[cell]=mean

    def compute(self,key,day):
        key=int(key)
        n=self._history_count(key,day)
        old=self.last.get(key)
        if old is not None and old[0]==n:return old[1],old[2]
        la,lo=self.grid.center(key)
        grid_values=(self.frame_mean[self.grid.candidate_keys[key]]
                     if self.frame_mean is not None else None)
        vals,counts=features(self.index,la,lo,day,candidate_grid=self.grid.direction_grid(key),
                             inclusive=True,center_cache_key=("center",key),
                             direction_lookup=self.direction if grid_values is None else None,
                             grid_azis=grid_values)
        self.last[key]=(n,vals,counts)
        return vals,counts


def _frame_mag(day,key,gray,start,end):
    lo=np.searchsorted(day,start,side="left");hi=np.searchsorted(day,end,side="right")
    if lo==hi:return np.empty(0,np.int32),np.empty(0,np.float32)
    keys=key[lo:hi]
    order=np.argsort(keys,kind="stable")
    sorted_key=keys[order]
    uniq,idx=np.unique(sorted_key,return_index=True)
    mx=np.maximum.reduceat(gray[lo:hi][order],idx)
    return uniq.astype(np.int32),mx.astype(np.float32)


def build(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    frame_dir=output/"frames";frame_dir.mkdir(exist_ok=True)
    mapping_config=output/"mapping_config.json"
    existing_frames=any(frame_dir.glob("*.npz"))
    if args.resume and existing_frames:
        previous=mapping_config if mapping_config.exists() else output/"manifest.json"
        if not previous.exists():
            raise ValueError("Cannot resume frames without a recorded mapping; use a new output directory")
        previous_mapping=json.loads(previous.read_text(encoding="utf-8")).get("mapping_upper")
        if previous_mapping is None or not np.allclose(previous_mapping,UPPER,rtol=0,atol=1e-12):
            raise ValueError("Cached frames use a different gray mapping; use a new output directory")
        manifest_path=output/"manifest.json"
        if manifest_path.exists():
            recorded=json.loads(manifest_path.read_text(encoding="utf-8"))
            if recorded.get("complete") and len(list(frame_dir.glob("*.npz")))==recorded.get("unique_input_frames"):
                print(f"[CL] already complete; output={output}",flush=True)
                return
    mapping_config.write_text(json.dumps({"channel_order":CHANNELS,"mapping_upper":UPPER},indent=2),encoding="utf-8")
    region=json.loads(Path(args.region).read_text(encoding="utf-8"))
    if (region["global_H"],region["global_W"],region["output_H"],region["output_W"])!=(3840,3840,340,200):
        raise ValueError("Unexpected archived Japan grid")
    samples=sample_index()
    samples.to_csv(output/"sample_index.csv",index=False)
    e,audit,files=load_usgs_months(Path(args.usgs))
    e=_prepare_usgs(e,region)
    if e.empty:raise ValueError("No Japanese USGS Mw>=4 events")
    e[["id","time","latitude","longitude","mag","MAGTYPE_ORIGINAL","MAGTYPE_STRIPPED",
       "MW","MAG_BRANCH","x","y","key","day_index","gray"]].to_csv(output/"usgs_event_audit.csv",index=False)
    day=e.day_index.to_numpy(np.int32);key=e.key.to_numpy(np.int32);gray=e.gray.to_numpy(np.float32)
    np.savez_compressed(output/"usgs_daily.npz",day=day,key=key,gray=gray)
    unique_frames=sorted(set().union(*(set(samples[f"x{j}_start"]) for j in range(INPUT_FRAMES))))
    frames_total=len(unique_frames)
    print(f"[CL] samples={len(samples)}, unique input frames={frames_total}, USGS events={len(e)}",flush=True)
    history,_,wsm_audit=load_wsm(Path(args.wsm))
    index=HistoryIndex(history)
    grid=Grid(region)
    cache=CenterCache(index,grid)
    start_wall=time.monotonic()
    distribution=Counter()
    for i,frame_start in enumerate(unique_frames,1):
        path=frame_dir/f"{frame_start}.npz"
        if args.resume and path.is_file():
            if i%1000==0:print(f"[CL] resumed {i}/{frames_total}",flush=True)
            continue
        ai=_monthday(frame_start);bi=ai+HORIZON-1
        keys,mag=_frame_mag(day,key,gray,ai,bi)
        vals=np.full((len(keys),len(PHYSICS)),np.nan,dtype=np.float32)
        mask=np.zeros((len(keys),len(PHYSICS)),dtype=np.bool_)
        mapped=np.zeros((len(keys),len(CHANNELS)),dtype=np.float32)
        mapped[:,0]=mag
        cutoff=START+pd.Timedelta(days=bi)
        cache.prepare(keys,cutoff)
        for row,cell in enumerate(keys):
            f,c=cache.compute(cell,cutoff)
            for channel,(feature,upper) in enumerate(zip(PHYSICS,UPPER),1):
                value=f[feature]
                if np.isfinite(value):
                    vals[row,channel-1]=value
                    mask[row,channel-1]=True
                    mapped[row,channel]=gray_linear(value,upper)
            distribution["depth_valid"]+=int(c["depth_n"]>=5)
            distribution["direction_valid"]+=int(c["direction_n"]>=5)
            distribution["mechanism_valid"]+=int(c["regime_n"]>=5)
        tmp=frame_dir/f".{frame_start}.tmp.npz"
        np.savez_compressed(tmp,keys=keys,raw=vals,valid=mask,gray=mapped)
        tmp.replace(path)
        if i%100==0 or i==frames_total:
            print(f"[CL] {i}/{frames_total} frames; last active={len(keys)}; elapsed={time.monotonic()-start_wall:.0f}s",flush=True)
        if args.limit_frames and i>=args.limit_frames:break
    complete=all((frame_dir/f"{s}.npz").exists() for s in unique_frames)
    feature_summary={}
    if complete:
        raw_values=[[] for _ in PHYSICS]
        active_total=0
        for frame_start in unique_frames:
            with np.load(frame_dir/f"{frame_start}.npz") as z:
                raw=z["raw"];valid=z["valid"]
                active_total+=len(z["keys"])
                for j in range(len(PHYSICS)):
                    if valid[:,j].any():raw_values[j].append(raw[:,j][valid[:,j]])
        for name,upper,blocks in zip(PHYSICS,UPPER,raw_values):
            values=np.concatenate(blocks) if blocks else np.empty(0,np.float32)
            feature_summary[name]={"valid_grid_frames":int(len(values)),
                                   "above_mapping_upper":int((values>upper).sum()),
                                   "p95":float(np.percentile(values,95)) if len(values) else None,
                                   "p99":float(np.percentile(values,99)) if len(values) else None,
                                   "max":float(values.max()) if len(values) else None}
        feature_summary["active_grid_frames"]=int(active_total)
    manifest={"dataset":"CL-MAG+9","complete":complete,"wsm_path":str(Path(args.wsm).resolve()),
              "wsm_sha256":sha256(Path(args.wsm)),"usgs_dir":str(Path(args.usgs).resolve()),
              "usgs_month_count":len(files),"usgs_month_sha256":{p.name:sha256(p) for p in files},
              "region_path":str(Path(args.region).resolve()),"region_sha256":sha256(Path(args.region)),
              "script_sha256":sha256(Path(__file__)),"core_sha256":sha256(Path(__file__).resolve().parents[1]/"shared"/"core.py"),
              "channel_order":CHANNELS,"physical_feature_order":PHYSICS,"mapping_upper":UPPER,
              "mapping_revision":"2026-10-11 user-selected upper bounds",
              "invalid_gray":0,"valid_physical_zero_gray":100,"input_frames":INPUT_FRAMES,
              "frame_days":HORIZON,"normalized_divisor":255,"target":"MAG only, one 450 day frame",
              "split_counts":samples.split.value_counts().to_dict(),"unique_input_frames":frames_total,
              "usgs_audit":audit,"wsm_audit":wsm_audit,"newly_generated_frame_valid_counts":dict(distribution),
              "all_frame_feature_summary":feature_summary,
              "limited_debug_run":bool(args.limit_frames)}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    print(f"[CL] complete={complete}; output={output}",flush=True)


class CLDataset:
    """Materialize X=(9,10,340,200), Y=(1,1,340,200), float32/255."""
    def __init__(self,root,split=None):
        self.root=Path(root)
        self.manifest=json.loads((self.root/"manifest.json").read_text())
        self.rows=pd.read_csv(self.root/"sample_index.csv")
        if split:self.rows=self.rows[self.rows.split==split].reset_index(drop=True)
        daily=np.load(self.root/"usgs_daily.npz")
        self.day,self.key,self.gray=daily["day"],daily["key"],daily["gray"]
        self.H=340;self.W=200

    def __len__(self):return len(self.rows)

    def __getitem__(self,i):
        row=self.rows.iloc[i]
        X=np.zeros((9,10,self.H,self.W),np.float32)
        for j in range(9):
            with np.load(self.root/"frames"/f"{row[f'x{j}_start']}.npz") as f:
                keys=f["keys"];mapped=f["gray"]
            yy,xx=np.divmod(keys,self.W)
            for channel in range(10):
                X[j,channel,yy,xx]=mapped[:,channel]
        Y=np.zeros((1,1,self.H,self.W),np.float32)
        a=_monthday(row.target_start)
        keys,mag=_frame_mag(self.day,self.key,self.gray,a,a+449)
        yy,xx=np.divmod(keys,self.W)
        Y[0,0,yy,xx]=mag
        return X/255,Y/255


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--wsm",default=str(DEFAULT_WSM))
    ap.add_argument("--usgs",default=str(DEFAULT_USGS))
    ap.add_argument("--region",default=str(DEFAULT_REGION))
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--limit-frames",type=int,default=0,help="Debug only")
    build(ap.parse_args())
