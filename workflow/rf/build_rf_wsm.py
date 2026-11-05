"""Build event-level RF association tables with WSM magnitude labels.

Usage: python3 build_rf_wsm.py --output /path/to/output
Each region is checkpointed, so rerunning after an interruption resumes it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"shared"))
from core import FEATURES, HistoryIndex, features, load_wsm, sha256

DEFAULT_WSM=Path("/Users/xiazongyuan/1 课题/4 全球地震预测项目资料汇总/Phase1/世界应力数据/wsm2025/WSM_Database_2025.csv")
DEFAULT_CONFIG=Path("/Users/xiazongyuan/1 课题/7 地震震级预测/xzy-小论文1-文件汇总/代码与结果（归档）/seismotectonic-spatial-seismicity-forecasting/code/rf_association/config.py")


def source_config(path):
    spec=importlib.util.spec_from_file_location("archived_rf_config",path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def build(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    config=source_config(Path(args.config))
    regions=list(config.REGIONS) if args.regions==["all"] else args.regions
    if set(regions)-set(config.REGIONS):raise ValueError(f"Unknown regions: {set(regions)-set(config.REGIONS)}")
    history,events,preprocess=load_wsm(Path(args.wsm))
    index=HistoryIndex(history)
    per_region=[]
    for name in regions:
        r=config.REGIONS[name]
        file=out/f"rf_events_{name}.csv"
        sub=events[events.LAT.between(r.lat_min,r.lat_max)&events.LON.between(r.lon_min,r.lon_max)].copy()
        if args.resume and file.exists():
            got=pd.read_csv(file,low_memory=False)
            if len(got)!=len(sub):
                raise ValueError(f"Checkpoint {file} has {len(got)} events but source has {len(sub)}")
            per_region.append(got)
            print(f"[RF] resumed {name}: {len(got)} events",flush=True)
            continue
        rows=[]
        start=time.monotonic()
        for i,evt in enumerate(sub.itertuples(index=False),1):
            result,counts=features(index,float(evt.LAT),float(evt.LON),evt.DAY,inclusive=False)
            missing=[k for k in FEATURES if not np.isfinite(result[k])]
            row={"region":name,"WSM_ID":evt.ID,"ISC_ID":evt.ISC_ID,
                 "date":evt.DAY.strftime("%Y-%m-%d"),"latitude":evt.LAT,"longitude":evt.LON,
                 "EQ_MAG":evt.EQ_MAG,"MAG_TYPE":evt.MAG_TYPE,
                 "MAG_BRANCH":evt.MAG_BRANCH,"MW":evt.MW,
                 "label_ge_4":int(evt.MW>=4),"label_ge_5":int(evt.MW>=5),"label_ge_6":int(evt.MW>=6),
                 "all_nine_valid":len(missing)==0,"invalid_indicators":"|".join(missing)}
            row.update(result);row.update(counts)
            rows.append(row)
            if i%500==0:
                print(f"[RF] {name} {i}/{len(sub)} {time.monotonic()-start:.0f}s",flush=True)
            if args.limit and i>=args.limit:break
        table=pd.DataFrame(rows)
        tmp=file.with_suffix(".tmp.csv")
        table.to_csv(tmp,index=False)
        tmp.replace(file)
        per_region.append(table)
        print(f"[RF] saved {name}: {len(table)} events, valid={table.all_nine_valid.sum() if len(table) else 0}",flush=True)
    all_events=pd.concat(per_region,ignore_index=True) if per_region else pd.DataFrame()
    all_events.to_csv(out/"rf_event_audit_1978_2024.csv",index=False)
    valid=all_events[all_events.all_nine_valid].copy()
    valid.to_csv(out/"rf_training_1978_2024.csv",index=False)
    training=valid[valid.date <= "2008-12-31"].copy()
    training.to_csv(out/"rf_training_1978_2008.csv",index=False)
    counts=[]
    for r,threshold in config.FORMAL_RF_TASKS:
        if r not in regions:continue
        for period,table in (("1978_2024",valid),("1978_2008",training)):
            subset=table[table.region==r]
            labels=subset[f"label_ge_{int(threshold)}"]
            counts.append({"region":r,"threshold":threshold,"period":period,
                           "usable_events":len(subset),"positive":int(labels.sum()),"negative":int(len(labels)-labels.sum())})
    pd.DataFrame(counts).to_csv(out/"rf_formal_task_counts.csv",index=False)
    all_count=all_events.groupby("region",dropna=False).agg(candidate_events=("MW","size"),all_nine_valid=("all_nine_valid","sum")).reset_index()
    all_count.to_csv(out/"rf_region_counts.csv",index=False)
    invalid={k:int(all_events[k].isna().sum()) for k in FEATURES}
    manifest={"dataset":"RF-WSM","wsm_path":str(Path(args.wsm).resolve()),"wsm_sha256":sha256(Path(args.wsm)),
              "config_path":str(Path(args.config).resolve()),"config_sha256":sha256(Path(args.config)),
              "script_sha256":sha256(Path(__file__)),"core_sha256":sha256(Path(__file__).resolve().parents[1]/"shared"/"core.py"),
              "preprocessing":preprocess,"regions":regions,"feature_order":FEATURES,
              "periods":{"full":"1978-01-01/2024-12-31","CL_history":"1978-01-01/2008-12-31"},
              "events_with_all_nine_valid":len(valid),"invalid_count_by_indicator":invalid,
              "thresholds":[4,5,6],"limited_debug_run":bool(args.limit)}
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=int),encoding="utf-8")
    print(f"[RF] completed: candidate={len(all_events)} valid={len(valid)} history={len(training)}",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--wsm",default=str(DEFAULT_WSM))
    ap.add_argument("--config",default=str(DEFAULT_CONFIG))
    ap.add_argument("--regions",nargs="+",default=["all"])
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--limit",type=int,default=0,help="Debug only: stop after N events per region")
    build(ap.parse_args())
