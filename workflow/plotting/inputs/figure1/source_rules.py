"""Shared, dimensioned data rules for the WSM/USGS dataset rebuild.

Distance is surface km; depth is km; azimuth and rotation are degrees.
The source CSVs are never modified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
import hashlib
import math

import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0088
FEATURES = (
    "MEAN_LOCAL_DEPTH", "MEDIAN_LOCAL_DEPTH", "STD_LOCAL_DEPTH",
    "CONSISTENCY_R_MEAN", "STD_LOCAL", "MAX_SPREAD_100KM_DEG",
    "MAX_ROTATION_DEG", "REGIME_ENTROPY", "REGIME_GINI",
)
REGIMES = ("NF", "NS", "SS", "TS", "TF")
QUALITY_WEIGHT = {"A": 1.0, "B": .8, "C": .6}
DROP_MAG_TYPES = frozenset(("mint", "Unknown", "ma", "mun", "no", "uk", "fa", "lg", "mdl", "mlv"))
DIRECT_MAG_TYPES = frozenset(("mww", "mwc", "mwb", "mwr", "Mi", "mwp", "ms", "ms_20", "Ml", "ml", "mlr"))
MB_MAG_TYPES = frozenset(("Mb", "mb", "mb_Lg", "mb_lg", "mblg", "mlg", "mfa"))
MD_MAG_TYPES = frozenset(("Md", "md", "mc"))
EARLIEST = pd.Timestamp("1978-01-01", tz="UTC")
LAST = pd.Timestamp("2024-12-31", tz="UTC")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def haversine(lat1, lon1, lat2, lon2):
    a1, b1, a2, b2 = map(np.deg2rad, np.broadcast_arrays(lat1, lon1, lat2, lon2))
    dl = (b2 - b1 + np.pi) % (2 * np.pi) - np.pi
    aa = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))


def angular_bounds(lat: float, radius_km: float):
    delta = radius_km / R_EARTH_KM
    lat_deg = np.rad2deg(delta)
    if np.deg2rad(abs(lat)) + delta >= np.pi/2:
        return lat_deg, 180.0
    return lat_deg, np.rad2deg(np.arcsin(min(1.0, np.sin(delta)/np.cos(np.deg2rad(lat)))))


def magnitude(mag, magtype):
    """Zhang's finaldata.py transEng + transEngToMag after pro_csv exclusions.

    Arithmetic is done in log-energy space to avoid power overflow. The
    function preserves the original case-sensitive branch matching.
    """
    m = np.asarray(mag, dtype=np.float64)
    t = np.asarray(magtype, dtype=str)
    out = m.copy()
    branch = np.full(m.shape, "default_original", dtype="U18")
    mask = np.isin(t, tuple(MB_MAG_TYPES))
    out[mask] = 1.6 * m[mask] - 4.0
    branch[mask] = "mb_energy"
    mask = np.isin(t, tuple(MD_MAG_TYPES))
    out[mask] = .936 * m[mask] - .16
    branch[mask] = "md_energy"
    mask = t == "me"
    out[mask] = m[mask] - 4/15
    branch[mask] = "me_energy"
    branch[np.isin(t, tuple(DIRECT_MAG_TYPES))] = "direct_original"
    return out, branch


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def load_wsm(path: Path):
    raw = pd.read_csv(path, low_memory=False, dtype={"ISC_ID": "string", "MAG_TYPE": "string"})
    required = ("ISC_ID", "TYPE", "QUALITY", "DATE", "LAT", "LON", "AZI", "DEPTH", "REGIME", "EQ_MAG", "MAG_TYPE")
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"WSM missing columns {sorted(missing)}")
    n = len(raw)
    df = raw.copy()
    for c in ("TYPE", "QUALITY", "REGIME", "MAG_TYPE", "ISC_ID"):
        df[c] = _text(df[c])
    for c in ("TYPE", "QUALITY", "REGIME"):
        df[c] = df[c].str.upper()
    for c in ("LAT", "LON", "AZI", "DEPTH", "EQ_MAG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[~np.isfinite(df[c]), c] = np.nan
    date_text = _text(df["DATE"])
    syntactic = date_text.str.fullmatch(r"\d{2}/\d{2}/\d{4}").fillna(False)
    df["DAY"] = pd.to_datetime(date_text.where(syntactic), format="%d/%m/%Y", errors="coerce", utc=True)
    isc_ok = df["ISC_ID"].notna() & ~df["ISC_ID"].str.upper().isin(("", "NAN", "NONE", "NULL", "NA", "N/A", "-", "0"))
    coords_ok = df["LAT"].between(-90,90) & df["LON"].between(-180,180)
    common = isc_ok & df["TYPE"].eq("FMS") & df["QUALITY"].isin(QUALITY_WEIGHT) & df["DAY"].notna() & coords_ok
    hist = df.loc[common].copy().reset_index(drop=True)
    hist["DEPTH_OK"] = hist["DEPTH"].ge(0) & hist["DEPTH"].notna()
    hist["AZI_OK"] = hist["AZI"].between(0,180) & hist["AZI"].notna()
    hist["REGIME_OK"] = hist["REGIME"].isin(REGIMES)
    hist["AZI_NORM"] = hist["AZI"].where(hist["AZI_OK"]) % 180
    hist["Q_WEIGHT"] = hist["QUALITY"].map(QUALITY_WEIGHT).astype(float)
    time_text = _text(df["TIME"]) if "TIME" in df else pd.Series(pd.NA, index=df.index, dtype="string")
    clock_shape = time_text.str.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d+)?").fillna(False)
    audit = {"raw_rows": n, "common_history_rows": len(hist), "before_1978_history_rows": int((hist.DAY < EARLIEST).sum()),
             "bad_date_rows": int(df.DAY.isna().sum()), "bad_isc_rows": int((~isc_ok).sum()),
             "bad_coordinate_rows": int((~coords_ok).sum()),
             "history_depth_valid": int(hist.DEPTH_OK.sum()), "history_azi_valid": int(hist.AZI_OK.sum()),
             "history_regime_valid": int(hist.REGIME_OK.sum()),
             "time_missing_rows": int(time_text.isna().sum()),
             "time_nonstandard_rows": int((time_text.notna() & ~clock_shape).sum())}
    events = hist[(hist.DAY >= EARLIEST) & (hist.DAY <= LAST)].copy()
    mag_ok = events["EQ_MAG"].gt(0) & events["EQ_MAG"].le(10)
    type_ok = events["MAG_TYPE"].notna() & events["MAG_TYPE"].ne("") & ~events["MAG_TYPE"].isin(DROP_MAG_TYPES)
    audit["event_before_magnitude_filter"] = len(events)
    audit["event_bad_magnitude"] = int((~mag_ok).sum())
    audit["event_bad_magnitude_type"] = int((~type_ok).sum())
    events = events.loc[mag_ok & type_ok].copy().reset_index(drop=True)
    events["MW"], events["MAG_BRANCH"] = magnitude(events.EQ_MAG.to_numpy(), events.MAG_TYPE.to_numpy())
    events = events[np.isfinite(events.MW)].copy().reset_index(drop=True)
    audit["event_valid_global"] = len(events)
    audit["event_mag_type_counts"] = events.MAG_TYPE.value_counts().to_dict()
    return hist, events, audit


def load_usgs_months(directory: Path):
    """Validate the complete 1973–2024 download, then return 1978–2024 events."""
    all_months = [(y,m) for y in range(1973,2025) for m in range(1,13)]
    files = [directory / f"usgs{y}_{m:02d}.csv" for y,m in all_months]
    missing = [p.name for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"USGS missing months ({len(missing)}): {missing[:20]}")
    parts, bad_time, wrong_month, raw_count = [], 0, 0, 0
    for (y,m), p in zip(all_months, files):
        df = pd.read_csv(p, low_memory=False, dtype={"magType": "string", "id": "string"})
        for c in ("time", "latitude", "longitude", "mag", "magType"):
            if c not in df:
                raise ValueError(f"{p.name} missing {c}")
        raw_count += len(df)
        ts = _text(df.time)
        syntax = ts.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$").fillna(False)
        dt = pd.to_datetime(ts.where(syntax), format="ISO8601", errors="coerce", utc=True)
        bad_time += int(dt.isna().sum())
        wrong_month += int((dt.notna() & ((dt.dt.year != y) | (dt.dt.month != m))).sum())
        if y >= 1978:
            df["DAY"] = dt.dt.normalize()
            parts.append(df)
    if bad_time or wrong_month:
        raise ValueError(f"USGS invalid UTC times={bad_time}, records in wrong CSV month={wrong_month}")
    events = pd.concat(parts, ignore_index=True)
    id_dups = int(events.id.duplicated().sum()) if "id" in events else 0
    if id_dups:
        raise ValueError(f"USGS duplicate event IDs: {id_dups}")
    for c in ("latitude", "longitude", "mag"):
        events[c] = pd.to_numeric(events[c], errors="coerce")
    coord_ok = events.latitude.between(-90,90) & events.longitude.between(-180,180)
    mag_ok = np.isfinite(events.mag)
    mt = _text(events.magType)
    type_ok = mt.notna() & mt.ne("") & ~mt.isin(DROP_MAG_TYPES)
    event_type_ok = events.type.eq("earthquake") if "type" in events else pd.Series(True, index=events.index)
    region = coord_ok & events.latitude.between(30.046875,45.984375) & events.longitude.between(128.0625,146.8125)
    audit = {"months_present": len(files), "raw_rows_all_months": raw_count,
             "raw_rows_1978_2024": len(events), "bad_coordinates": int((~coord_ok).sum()),
             "missing_or_excluded_magtype": int((~type_ok).sum()), "bad_magnitude": int((~mag_ok).sum()),
             "outside_japan_grid": int((~region).sum()),
             "type_column_present": "type" in events, "other_event_type": int((~event_type_ok).sum())}
    events = events.loc[region & mag_ok & type_ok & event_type_ok & events.DAY.between(EARLIEST,LAST)].copy()
    events["MAGTYPE_ORIGINAL"] = events.magType
    events["MAGTYPE_STRIPPED"] = mt.loc[events.index]
    events["MW"], events["MAG_BRANCH"] = magnitude(events.mag.to_numpy(), events.MAGTYPE_STRIPPED.to_numpy())
    events = events[np.isfinite(events.MW) & (events.MW >= 4)].copy().reset_index(drop=True)
    audit["mw4_japan_rows"] = len(events)
    audit["mag_type_counts"] = events.MAGTYPE_STRIPPED.value_counts().to_dict()
    return events, audit, files


class HistoryIndex:
    """Global WSM history; spatial bins retrieve candidates, Haversine decides membership."""
    def __init__(self, history: pd.DataFrame):
        h = history.sort_values("DAY", kind="stable").reset_index(drop=True)
        self.h = h
        self.day = (h.DAY.to_numpy(dtype="datetime64[D]") - np.datetime64("1970-01-01")).astype(np.int32)
        self.lat = h.LAT.to_numpy(float)
        self.lon = h.LON.to_numpy(float)
        self.azi = h.AZI_NORM.to_numpy(float)
        self.depth = h.DEPTH.to_numpy(float)
        self.regime = h.REGIME.astype(str).to_numpy()
        self.qw = h.Q_WEIGHT.to_numpy(float)
        self.bins = defaultdict(list)
        for i, (lat, lon) in enumerate(zip(self.lat,self.lon)):
            longitude_bin=(int(np.floor(lon))+180)%360-180
            self.bins[(int(np.floor(lat)),longitude_bin)].append(i)
        self.bins = {k:np.asarray(v,dtype=np.int32) for k,v in self.bins.items()}
        self._neighbors = {}

    def neighbors(self, lat: float, lon: float, radius: float=100, cache_key=None):
        key = (cache_key, radius) if cache_key is not None else None
        if key is not None and key in self._neighbors:
            return self._neighbors[key]
        dy, dx = angular_bounds(lat, radius)
        ys = range(max(-90, math.floor(lat-dy)), min(90,math.floor(lat+dy))+1)
        xs = range(math.floor(lon-dx), math.floor(lon+dx)+1)
        ids = [self.bins.get((y, (x+180)%360-180)) for y in ys for x in xs]
        ids = [v for v in ids if v is not None]
        if not ids:
            out = (np.empty(0,dtype=np.int32), np.empty(0,dtype=float))
        else:
            cand = np.concatenate(ids)
            dist = haversine(lat,lon,self.lat[cand],self.lon[cand])
            keep = dist <= radius + 1e-9
            out = cand[keep], dist[keep]
        if key is not None:
            self._neighbors[key] = out
        return out

    def before(self, ids, day: pd.Timestamp, inclusive=False):
        d = int((day.tz_localize(None).to_datetime64().astype("datetime64[D]") - np.datetime64("1970-01-01")).astype(int))
        return ids[self.day[ids] <= d] if inclusive else ids[self.day[ids] < d]


def _direction(azi, qw, distance):
    ok = np.isfinite(azi) & np.isfinite(qw) & (qw > 0) & (distance < 100)
    if ok.sum() < 5:
        return np.nan, np.nan, np.nan, int(ok.sum())
    d = distance[ok]
    w = qw[ok] * np.where(d <= 10, 1., 1. - (d-10)/90)
    ok2 = np.isfinite(w) & (w > 0)
    if ok2.sum() < 5:
        return np.nan, np.nan, np.nan, int(ok2.sum())
    radians = np.deg2rad(2*azi[ok][ok2])
    c = np.sum(w[ok2]*np.cos(radians))
    s = np.sum(w[ok2]*np.sin(radians))
    R = float(np.clip(np.hypot(c,s)/np.sum(w[ok2]), 1e-12, 1))
    mean = float(np.degrees(np.arctan2(s,c))/2 % 180)
    std = float(np.degrees(np.sqrt(-np.log(R)/2)))
    return mean, R, std, int(ok2.sum())


def _max_axial_spread(values):
    v = np.sort(np.asarray(values,float) % 180)
    if len(v)<2:return np.nan
    both = np.concatenate((v,v+180))
    target = v+90
    idx = np.searchsorted(both,target)
    near = np.minimum(np.abs(both[np.clip(idx,0,len(both)-1)]-target),
                      np.abs(both[np.clip(idx-1,0,len(both)-1)]-target))
    return float(90-np.min(near))


def _axial_diff(a,b):
    d=abs(a-b)%180
    return float(min(d,180-d))


def _axial_mean(azi):
    a=np.deg2rad(2*np.asarray(azi,float))
    return float(np.degrees(np.arctan2(np.sin(a).sum(),np.cos(a).sum()))/2%180)


def _rf_grid_field(index, ids, glat, glon):
    """Direction field for one RF event, calculated in bounded NumPy blocks."""
    means=np.full(len(glat),np.nan)
    for a in range(0,len(glat),64):
        b=min(a+64,len(glat))
        dist=haversine(glat[a:b,None],glon[a:b,None],index.lat[ids][None,:],index.lon[ids][None,:])
        az=index.azi[ids]
        good=np.isfinite(az)[None,:] & (dist<100) & (index.qw[ids][None,:]>0)
        n=good.sum(axis=1)
        w=np.where(good,np.where(dist<=10,1.,1.-(dist-10)/90)*index.qw[ids][None,:],0.)
        wsum=w.sum(axis=1)
        rad=np.deg2rad(2*np.nan_to_num(az,nan=0.))
        s=w@np.sin(rad)
        c=w@np.cos(rad)
        R=np.clip(np.hypot(c,s)/np.where(wsum>0,wsum,1),1e-12,1)
        std=np.rad2deg(np.sqrt(-np.log(R)/2))
        valid=(n>=5)&(std<=25)&(wsum>0)
        means[a:b][valid]=(np.rad2deg(np.arctan2(s[valid],c[valid]))/2)%180
    return means


def rf_grid(lat, lon):
    """Event-relative 0.1-degree RF grid, including a fixed target grid cell."""
    dy,dx=angular_bounds(lat,100)
    lat0=lat-max(100/111.,dy)
    lon0=lon-max(100/(111*np.cos(np.deg2rad(lat))+1e-6),dx)
    lats=np.arange(lat0,lat+max(100/111.,dy)+1e-8,.1)
    lons=np.arange(lon0,lon+max(100/(111*np.cos(np.deg2rad(lat))+1e-6),dx)+1e-8,.1)
    la,lo=np.meshgrid(lats,lons,indexing="ij")
    la,lo=la.ravel(),lo.ravel()
    keep=haversine(lat,lon,la,lo)<=100+1e-9
    la,lo=la[keep],lo[keep]
    target=int(np.argmin(haversine(lat,lon,la,lo)))
    return la,lo,target


def features(index: HistoryIndex, lat: float, lon: float, day: pd.Timestamp,
             candidate_grid=None, inclusive=False, center_cache_key=None,
             direction_lookup=None, grid_azis=None):
    """Compute all nine raw indicators and evidence counts at one place and cutoff.

    candidate_grid=(latitudes,longitudes,target_cell_index); if omitted, the
    archived RF 0.1-degree event-relative grid is used.
    """
    vals={k:np.nan for k in FEATURES}
    counts={"depth_n":0,"direction_n":0,"regime_n":0,"valid_grid_n":0}
    ids,dists=index.neighbors(lat,lon,200, cache_key=(center_cache_key,"outer") if center_cache_key is not None else None)
    if len(ids)==0:return vals,counts
    dday = np.datetime64(day.tz_localize(None).date())
    di = int((dday-np.datetime64("1970-01-01")).astype(int))
    keep=index.day[ids] <= di if inclusive else index.day[ids] < di
    ids,dists=ids[keep],dists[keep]
    if len(ids)==0:return vals,counts
    inner=dists<=100+1e-9
    ci,cd=ids[inner],dists[inner]
    dep=index.depth[ci]
    dep=dep[np.isfinite(dep)&(dep>=0)]
    counts["depth_n"]=len(dep)
    if len(dep)>=5:
        vals["MEAN_LOCAL_DEPTH"]=float(np.mean(dep))
        vals["MEDIAN_LOCAL_DEPTH"]=float(np.median(dep))
        vals["STD_LOCAL_DEPTH"]=float(np.std(dep,ddof=0))
    reg=index.regime[ci]
    reg=reg[np.isin(reg,REGIMES)]
    counts["regime_n"]=len(reg)
    if len(reg)>=5:
        freq=np.asarray([(reg==r).sum() for r in REGIMES],float)/len(reg)
        nonzero=freq[freq>0]
        vals["REGIME_ENTROPY"]=float(-(nonzero*np.log(nonzero)).sum())
        vals["REGIME_GINI"]=float(1-(freq*freq).sum())
    _,R,std,n=_direction(index.azi[ci],index.qw[ci],cd)
    counts["direction_n"]=n
    if n>=5:
        vals["CONSISTENCY_R_MEAN"]=R
        vals["STD_LOCAL"]=std
    if candidate_grid is None:
        candidate_grid=rf_grid(lat,lon)
    glat,glon,target_cell=candidate_grid
    if grid_azis is not None:
        grid_azis=np.asarray(grid_azis,float)
    elif direction_lookup is None:
        grid_azis=_rf_grid_field(index,ids,glat,glon)
    else:
        grid_azis=np.full(len(glat),np.nan)
        for k,(ga,go) in enumerate(zip(glat,glon)):
            mean,_,gstd,gn=direction_lookup(ga,go,day,inclusive)
            if gn>=5 and np.isfinite(gstd) and gstd<=25:
                grid_azis[k]=mean
    valid_azis=grid_azis[np.isfinite(grid_azis)]
    target_azi=grid_azis[target_cell]
    counts["valid_grid_n"]=len(valid_azis)
    if len(valid_azis)>=2:
        vals["MAX_SPREAD_100KM_DEG"]=_max_axial_spread(valid_azis)
    if np.isfinite(target_azi) and len(valid_azis):
        vals["MAX_ROTATION_DEG"]=_axial_diff(_axial_mean(valid_azis),target_azi)
    return vals,counts


class GridDirectionCache:
    """Reuse a grid direction until a new WSM record enters its 100 km history."""
    def __init__(self,index:HistoryIndex):
        self.index=index
        self.cells={}
        self.values={}

    def __call__(self,lat,lon,day,inclusive=True):
        key=(round(float(lat),8),round(float(lon),8))
        if key not in self.cells:
            ids,dists=self.index.neighbors(lat,lon,100)
            order=np.argsort(self.index.day[ids],kind="stable")
            self.cells[key]=(ids[order],dists[order],self.index.day[ids[order]])
        ids,dists,days=self.cells[key]
        di=int((np.datetime64(day.tz_localize(None).date())-np.datetime64("1970-01-01")).astype(int))
        n=int(np.searchsorted(days,di,side="right" if inclusive else "left"))
        old=self.values.get(key)
        if old is None or old[0]!=n:
            old=(n,_direction(self.index.azi[ids[:n]],self.index.qw[ids[:n]],dists[:n]))
            self.values[key]=old
        return old[1]


def gray_linear(value, upper):
    return np.float32(100+155*np.clip(value/upper,0,1)) if np.isfinite(value) else np.float32(0)


def gray_mag(mw):
    if not np.isfinite(mw) or mw<4:return np.float32(0)
    return np.float32(100+155*np.clip(np.log10((mw-3.5)/.5)/np.log10(13),0,1))
