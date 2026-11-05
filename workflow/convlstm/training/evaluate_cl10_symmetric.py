"""Evaluate frozen CL10 predictions using symmetric Japan spatial matching.

Only predictions exported by predict_cl10.py are accepted. The script does not
load a model or change the training dataset. Distances here are archived pixel
expansion widths, not physical kilometres.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np

from cl10_data import CL10Data, HEIGHT, WIDTH, sha256
from spatial_symmetric import symmetric_expand, EVALUATION_VERSION, MATCHING_RULE


THRESHOLD_GRAY = 100.0  # Mw >= 4.0 target coding, in 0-255 gray units
EXPAND_D = 3  # archived formal evaluation scale, in grid cells
NONOVERLAP_DAYS = 450  # one complete target window
REGION_SHA256 = "13721464652124f9c8596130733b30a5d99188cf1ab503ec3efe612b25850ddf"

PROTOCOLS = (
    ("neither", False, False),
    ("target_only", False, True),
    ("prediction_only", True, False),
    ("both", True, True),
)
FIELDS = (
    "row", "target_start", "target_end", "protocol", "prediction_expanded",
    "target_expanded", "TP", "FP", "FN", "TN", "pred_pos", "target_pos",
    "precision", "recall", "CSI", "pred_area_ratio", "target_area_ratio",
    "nonoverlap",
)
METRICS = ("precision", "recall", "CSI", "pred_area_ratio", "target_area_ratio")


def seed_metadata(record: dict) -> dict:
    """Read either a new dual-seed run or an archived single-seed run."""
    if "init_seed" in record or "order_seed" in record:
        seeds = {name: record.get(name) for name in ("init_seed", "order_seed")}
        if any(type(value) is not int or not 0 <= value < 2**32 for value in seeds.values()):
            raise ValueError("Missing or invalid dual-seed metadata")
        return seeds
    if type(record.get("seed")) is int:
        return {"seed": record["seed"]}
    raise ValueError("Missing random seed metadata")


def archived_expand(image: np.ndarray) -> np.ndarray:
    """Compatibility name; the implementation now uses symmetric max expansion."""
    return symmetric_expand(image, d=EXPAND_D)


def counts(prediction: np.ndarray, target: np.ndarray) -> dict:
    pred = prediction * 255.0 >= THRESHOLD_GRAY
    truth = target * 255.0 >= THRESHOLD_GRAY
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    total = int(pred.size)
    tn = total - tp - fp - fn
    pred_pos, target_pos = tp + fp, tp + fn
    union = tp + fp + fn
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "pred_pos": pred_pos, "target_pos": target_pos,
        "precision": tp / pred_pos if pred_pos else None,
        "recall": tp / target_pos if target_pos else None,
        "CSI": tp / union if union else None,
        "pred_area_ratio": pred_pos / total,
        "target_area_ratio": target_pos / total,
    }


def _read_prediction_index(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row", "target_start", "target_end"]:
            raise ValueError("Unexpected prediction sample_index.csv columns")
        return list(reader)


def _validate_inputs(data: CL10Data, pred_dir: Path, split: str):
    manifest = json.loads((pred_dir / "manifest.json").read_text(encoding="utf-8"))
    dataset_manifest = data.manifest
    if dataset_manifest.get("region_sha256") != REGION_SHA256:
        raise ValueError("Expansion geometry does not match this dataset's Japan region")
    if not manifest.get("complete") or manifest.get("split") != split:
        raise ValueError("Prediction export is incomplete or has the wrong split")
    if manifest.get("dataset_manifest_sha256") != sha256(data.root / "manifest.json"):
        raise ValueError("Prediction export uses a different CL10 dataset")
    if (manifest.get("count"), manifest.get("shape"), manifest.get("prediction_dtype")) != (
        len(data), [len(data), HEIGHT, WIDTH], "float32"
    ):
        raise ValueError("Prediction count, shape or dtype does not match the dataset")
    if manifest.get("prediction_file") != "predictions.npy" or manifest.get("target_dates_file") != "sample_index.csv":
        raise ValueError("Unexpected prediction export file names")
    if manifest.get("prediction_scale") != "raw model output against targets normalized by 255; no clipping":
        raise ValueError("Prediction scale is not the normalized output expected by this evaluator")
    checkpoint_hash = manifest.get("best_checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_hash):
        raise ValueError("Missing or invalid best checkpoint identity")
    if not isinstance(manifest.get("best_epoch"), int) or manifest["best_epoch"] < 1:
        raise ValueError("Missing best epoch")
    if not isinstance(manifest.get("selected_channels"), list) or not manifest["selected_channels"]:
        raise ValueError("Missing selected channels")
    predicted_rows = _read_prediction_index(pred_dir / "sample_index.csv")
    if len(predicted_rows) != len(data):
        raise ValueError("Prediction date index length differs from CL10 split")
    for i, (predicted, expected) in enumerate(zip(predicted_rows, data.rows)):
        if (predicted["row"], predicted["target_start"], predicted["target_end"]) != (
            str(i), expected["target_start"], expected["target_end"]
        ):
            raise ValueError(f"Prediction target dates differ at row {i}")
    actual_prediction_sha256 = sha256(pred_dir / "predictions.npy")
    expected_prediction_sha256 = manifest.get("predictions_sha256")
    if expected_prediction_sha256 is not None and expected_prediction_sha256 != actual_prediction_sha256:
        raise ValueError("Prediction array differs from its recorded SHA256")
    array = np.load(pred_dir / "predictions.npy", mmap_mode="r", allow_pickle=False)
    if array.shape != (len(data), HEIGHT, WIDTH) or array.dtype != np.float32:
        raise ValueError("predictions.npy has unexpected shape or dtype")
    return manifest, array, actual_prediction_sha256


def _check_cleanup_path(pred_dir: Path, run_dir: Path, out: Path, split: str) -> None:
    """Restrict automatic removal to this run's validation prediction export."""
    if split != "validation" or pred_dir != run_dir / "validation_predictions":
        raise ValueError("Cleanup is restricted to this run's validation_predictions directory")
    if out == pred_dir or pred_dir in out.parents:
        raise ValueError("Evaluation output cannot be inside the prediction directory")
    if pred_dir.is_symlink() or set(path.name for path in pred_dir.iterdir()) != {
        "manifest.json", "sample_index.csv", "predictions.npy"
    } or any(not path.is_file() or path.is_symlink() for path in pred_dir.iterdir()):
        raise ValueError("Prediction directory contains unexpected files or links; refusing cleanup")


def _summary(rows: list[dict], protocol: str, cohort: str) -> dict:
    selected = [row for row in rows if row["protocol"] == protocol and
                (cohort == "daily" or row["nonoverlap"])]
    result = {
        "evaluation_version": EVALUATION_VERSION,
        "cohort": cohort, "protocol": protocol, "sample_count": len(selected),
        "threshold_gray": THRESHOLD_GRAY, "expand_d": EXPAND_D,
    }
    for metric in METRICS:
        values = np.array([row[metric] for row in selected if row[metric] is not None], dtype=np.float64)
        result[f"{metric}_valid_count"] = int(len(values))
        result[f"{metric}_mean"] = float(values.mean()) if len(values) else None
        result[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else None
        result[f"{metric}_min"] = float(values.min()) if len(values) else None
        result[f"{metric}_max"] = float(values.max()) if len(values) else None
    return result


def _write_csv(path: Path, rows: list[dict], fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    data = CL10Data(args.data, args.split, (0,), cache_size=0)
    pred_dir = Path(args.predictions).resolve()
    pred_manifest, predictions, prediction_sha256 = _validate_inputs(data, pred_dir, args.split)
    run_dir = None
    if args.run:
        run_dir = Path(args.run).resolve()
        run_config = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        run_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        for key, filename in (("manifest", "manifest.json"),
                              ("sample_index", "sample_index.csv"),
                              ("usgs_daily", "usgs_daily.npz")):
            if run_config.get("dataset_sha256", {}).get(key) != sha256(data.root / filename):
                raise ValueError(f"Dataset {filename} differs from training")
        if sha256(run_dir / "best.pt") != pred_manifest["best_checkpoint_sha256"]:
            raise ValueError("best.pt differs from the checkpoint used for predictions")
        if (run_summary.get("best_epoch") != pred_manifest["best_epoch"] or
                run_config.get("selected_channels") != pred_manifest["selected_channels"] or
                seed_metadata(run_config) != seed_metadata(pred_manifest)):
            raise ValueError("Training run metadata differs from prediction export")
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(f"Evaluation output already exists: {out}")
    if args.cleanup_predictions:
        if run_dir is None:
            raise ValueError("--cleanup-predictions requires --run")
        _check_cleanup_path(pred_dir, run_dir, out, args.split)

    rows = []
    for i, sample in enumerate(data.rows):
        raw_pred = np.asarray(predictions[i])
        if not np.isfinite(raw_pred).all():
            raise ValueError(f"Nonfinite prediction at row {i}")
        keys, gray = data.target_keys(sample)
        raw_target = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        raw_target.flat[keys] = gray / 255.0
        expanded_pred = archived_expand(raw_pred)
        expanded_target = archived_expand(raw_target)
        for protocol, pred_expanded, target_expanded in PROTOCOLS:
            stats = counts(expanded_pred if pred_expanded else raw_pred,
                           expanded_target if target_expanded else raw_target)
            rows.append({
                "row": i, "target_start": sample["target_start"],
                "target_end": sample["target_end"], "protocol": protocol,
                "prediction_expanded": pred_expanded,
                "target_expanded": target_expanded, **stats,
                "nonoverlap": i % NONOVERLAP_DAYS == 0,
            })
        if (i + 1) % 100 == 0:
            print(f"evaluated {i + 1}/{len(data)}", flush=True)

    nonoverlap = [row for row in rows if row["nonoverlap"]]
    intervals = [(date.fromisoformat(row["target_start"]), date.fromisoformat(row["target_end"]))
                 for row in nonoverlap if row["protocol"] == "neither"]
    if any(previous[1] >= following[0] for previous, following in zip(intervals, intervals[1:])):
        raise ValueError("Selected nonoverlap targets intersect")
    summary = [_summary(rows, p, cohort) for cohort in ("daily", "nonoverlap")
               for p, _, _ in PROTOCOLS]
    official = []
    for cohort in ("daily", "nonoverlap"):
        by_protocol = {row["protocol"]: row for row in summary if row["cohort"] == cohort}
        official.append({
            "evaluation_version": EVALUATION_VERSION,
            "metric_protocol": "symmetric_neighborhood_PR_and_expanded_both_CSI",
            "cohort": cohort, "sample_count": by_protocol["both"]["sample_count"],
            "threshold_gray": THRESHOLD_GRAY, "expand_d": EXPAND_D,
            "precision": by_protocol["target_only"]["precision_mean"],
            "precision_std": by_protocol["target_only"]["precision_std"],
            "precision_valid_count": by_protocol["target_only"]["precision_valid_count"],
            "recall": by_protocol["prediction_only"]["recall_mean"],
            "recall_std": by_protocol["prediction_only"]["recall_std"],
            "recall_valid_count": by_protocol["prediction_only"]["recall_valid_count"],
            "CSI": by_protocol["both"]["CSI_mean"],
            "CSI_expanded_both": by_protocol["both"]["CSI_mean"],
            "CSI_std": by_protocol["both"]["CSI_std"],
            "CSI_valid_count": by_protocol["both"]["CSI_valid_count"],
        })

    out.mkdir(parents=True, exist_ok=False)
    _write_csv(out / "by_sample.csv", rows, FIELDS)
    _write_csv(out / "nonoverlap_by_sample.csv", nonoverlap, FIELDS)
    _write_csv(out / "summary.csv", summary)
    _write_csv(out / "official_summary.csv", official)
    shutil.copyfile(pred_dir / "manifest.json", out / "prediction_manifest.json")
    evaluation_manifest = {
        "complete": True, "split": args.split, "sample_count": len(data),
        "evaluation_version": EVALUATION_VERSION,
        "matching_rule": MATCHING_RULE,
        "evaluator_sha256": sha256(Path(__file__)),
        "spatial_module_sha256": sha256(Path(__file__).with_name("spatial_symmetric.py")),
        "data_reader_sha256": sha256(Path(__file__).with_name("cl10_data.py")),
        "nonoverlap_sample_count": len(intervals),
        "nonoverlap_selection": "indices 0,450,900,... from split's first target start",
        "threshold_gray": THRESHOLD_GRAY, "expand_d": EXPAND_D,
        "expansion": "symmetric union of source and destination latitude-band rectangles; maximum over all matches; image-edge clipping only",
        "protocols": {p: {"prediction_expanded": pe, "target_expanded": te} for p, pe, te in PROTOCOLS},
        "undefined_denominator": "empty CSV cell, excluded from mean/std and counted separately",
        "standard_deviation_ddof": 0,
        "prediction_manifest_sha256": sha256(pred_dir / "manifest.json"),
        "predictions_sha256": prediction_sha256,
        "prediction_checkpoint_sha256": pred_manifest["best_checkpoint_sha256"],
        "prediction_best_epoch": pred_manifest["best_epoch"],
        "checkpoint_file_verified": bool(args.run),
        "selected_channels": pred_manifest["selected_channels"],
        **seed_metadata(pred_manifest),
        "dataset_manifest_sha256": pred_manifest["dataset_manifest_sha256"],
        "files": ["by_sample.csv", "nonoverlap_by_sample.csv", "summary.csv",
                  "official_summary.csv", "prediction_manifest.json"],
        "predictions_cleanup_requested": bool(args.cleanup_predictions),
        "predictions_cleanup_completed": False,
    }
    evaluation_manifest["output_sha256"] = {
        name: sha256(out / name) for name in evaluation_manifest["files"]
    }
    manifest_path = out / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(evaluation_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.cleanup_predictions:
        del predictions
        shutil.rmtree(pred_dir)
        evaluation_manifest["predictions_cleanup_completed"] = True
        manifest_path.write_text(json.dumps(evaluation_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out}: {len(data)} samples, {len(intervals)} nonoverlap targets", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Complete cl10_mapping_v2 directory")
    parser.add_argument("--predictions", required=True, help="Completed predict_cl10.py output directory")
    parser.add_argument("--run", help="Optional completed training run; verify best.pt against predictions")
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--out", required=True, help="New evaluation output directory")
    parser.add_argument("--cleanup-predictions", action="store_true",
                        help="After a successful validation evaluation, delete this run's temporary prediction export")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
