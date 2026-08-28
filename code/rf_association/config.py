"""Shared configuration for the random-forest association analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DERIVED_DATA_DIR = PROJECT_ROOT / "data" / "derived"
REGIONAL_RECORDS_DIR = DERIVED_DATA_DIR / "regional_records"
FEATURES_DIR = DERIVED_DATA_DIR / "event_features"
LABELS_DIR = DERIVED_DATA_DIR / "labeled_events"
RESULTS_DIR = PROJECT_ROOT / "results" / "random_forest"
RUNS_DIR = RESULTS_DIR / "runs"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

WSM_RAW_PATH = RAW_DATA_DIR / "WSM_Database_2025.csv"
USGS_RAW_DIR = RAW_DATA_DIR / "usgs_csv"
FILTERED_WSM_PATH = DERIVED_DATA_DIR / "wsm2025_filtered_records.csv"
REGION_RECORD_COUNTS_PATH = DERIVED_DATA_DIR / "region_record_counts.csv"
SAMPLE_AVAILABILITY_PATH = DERIVED_DATA_DIR / "sample_availability_by_region_threshold.csv"

YEAR_START = 1964
YEAR_END = 2024
RADIUS_KM = 100.0
KM_PER_DEG_LAT = 111.32
GRID_STEP_DEG = 0.1
MIN_ORIENTATION_RECORDS = 5
LOCAL_DIRECTION_STD_MAX_DEG = 25.0

FEATURE_COLUMNS = (
    "MEAN_LOCAL_DEPTH",
    "MEDIAN_LOCAL_DEPTH",
    "STD_LOCAL_DEPTH",
    "STD_LOCAL",
    "CONSISTENCY_R_MEAN",
    "MAX_SPREAD_100KM_DEG",
    "MAX_ROTATION_DEG",
    "REGIME_ENTROPY",
    "REGIME_GINI",
)
MAGNITUDE_THRESHOLDS = (4.0, 5.0, 6.0)
DEFAULT_REGION = "japan"

FORMAL_RF_TASKS = (
    ("southern_california", 4.0),
    ("southern_california", 5.0),
    ("japan", 4.0),
    ("japan", 5.0),
    ("japan", 6.0),
    ("new_zealand", 4.0),
    ("new_zealand", 5.0),
    ("greece", 4.0),
    ("greece", 5.0),
    ("indonesia", 5.0),
    ("indonesia", 6.0),
    ("iran", 5.0),
    ("iran", 6.0),
    ("italy", 4.0),
    ("italy", 5.0),
    ("mexico", 4.0),
    ("mexico", 5.0),
    ("mexico", 6.0),
    ("taiwan", 4.0),
    ("taiwan", 5.0),
    ("central_chile", 4.0),
    ("central_chile", 5.0),
    ("central_chile", 6.0),
)

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_LEAF = 5
RF_TEST_SIZE = 0.5
RF_REPEATS = 1000


@dataclass(frozen=True)
class Region:
    key: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


REGIONS = {
    "southern_california": Region("southern_california", "Southern California", 32.0, 38.3, -123.0, -115.0),
    "japan": Region("japan", "Japan", 30.05, 45.95, 128.10, 146.80),
    "new_zealand": Region("new_zealand", "New Zealand", -47.5, -34.5, 166.5, 178.5),
    "greece": Region("greece", "Greece", 33.0, 43.0, 18.0, 31.0),
    "indonesia": Region("indonesia", "Indonesia", -11.0, 6.0, 94.0, 123.0),
    "iran": Region("iran", "Iran", 25.0, 40.0, 44.0, 64.0),
    "italy": Region("italy", "Italy", 36.0, 48.0, 5.0, 20.0),
    "mexico": Region("mexico", "Mexico", 13.0, 33.0, -117.0, -91.0),
    "taiwan": Region("taiwan", "Taiwan", 21.0, 26.0, 119.0, 123.0),
    "central_chile": Region("central_chile", "Central Chile", -35.0, -29.0, -75.0, -69.0),
}


def region_keys(selection: list[str]) -> list[str]:
    """Resolve command-line region selections, including the ``all`` shortcut."""
    if "all" in selection:
        return list(REGIONS)
    unknown = sorted(set(selection) - set(REGIONS))
    if unknown:
        raise ValueError(f"Unknown region key(s): {', '.join(unknown)}")
    return selection


def ensure_output_directories() -> None:
    for directory in (
        DERIVED_DATA_DIR,
        REGIONAL_RECORDS_DIR,
        FEATURES_DIR,
        LABELS_DIR,
        RUNS_DIR,
        SUMMARIES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
