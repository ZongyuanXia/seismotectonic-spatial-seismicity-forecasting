from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from model import ConvLstm


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
CONVLSTM_DATA_DIR = REPO_ROOT / "data" / "derived" / "convlstm"
DATASET_DIR = CONVLSTM_DATA_DIR / "dataset"
STAGE4_PREFIX = DATASET_DIR / "stage4" / "second"
WEIGHT_MAP_PATH = CONVLSTM_DATA_DIR / "loss_weight_map_mw4.npz"
RESULTS_DIR = REPO_ROOT / "results" / "convlstm"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"

# Default for direct use. Formal scripts select the same lists by M1-M5 name;
# 02_train_convlstm.py also accepts an explicit --channels list.
SELECTED_CHANNELS = [0]
# M1: [0]
# M2: [0, 1, 2]
# M3: [0, 3, 4]
# M4: [0, 5, 6]
# M5: [0, 1, 2, 5, 6]

MODEL_CHANNELS = {
    "M1": [0],
    "M2": [0, 1, 2],
    "M3": [0, 3, 4],
    "M4": [0, 5, 6],
    "M5": [0, 1, 2, 5, 6],
}

EXCLUDED_RUNS = {"M1": 9, "M2": 6, "M3": 10, "M4": 9, "M5": 9}


class Config:
    def __init__(self, seq_in=9, seq_out=1, selected_channels=None):
        channels = SELECTED_CHANNELS if selected_channels is None else selected_channels
        self.selected_channels = list(channels)
        self.in_channels = len(self.selected_channels)

        self.sequences_in = seq_in
        self.sequences_out = seq_out
        self.pic = seq_in
        self.dtday = 450
        self.percent = 0.7

        # Global mother grid and the Japan crop used in the manuscript.
        self.width = 3840
        self.height = 3840
        self.region_mode = "anchored_crop"
        self.region_name = "japan"
        self.lat_min = 30.05
        self.lat_max = 45.95
        self.lon_min = 128.10
        self.lon_max = 146.80
        self.crop_x_min = 3286
        self.crop_x_max = 3485
        self.crop_y_min = 2561
        self.crop_y_max = 2900

        self.start_mag = 4.0
        self.start_value = 100.0
        self.depth_min = 0.0
        self.depth_max = 700.0
        self.time_start = "1973_01"
        self.time_end = "2021_05"

        # WSM-derived channels.
        self.wsm_neighbor_km = 100.0
        self.wsm_min_n_local = 5
        self.wsm_std_threshold = 25.0
        self.wsm_time_mode = "cumulative"
        self.std_local_map_max = 45.0
        self.depth_std_map_max = 100.0
        self.wsm_local_mean_cache = True
        self.wsm_effective_snapshot_cache = True
        self.wsm_center_metric_cache = True
        self.wsm_global_local_mean_cache = True
        self.wsm_precompute_direction_field = True
        self.wsm_direction_use_kdtree = True
        self.wsm_direction_c2_radius_factor = 2.05
        self.wsm_direction_tree_query_factor = 1.35

        # Training and formal evaluation.
        self.batch = 1
        self.num_workers = 0
        self.num_epochs = 40
        self.patience = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample = str(WEIGHT_MAP_PATH)
        self.balancing_weights = [0.000177210579159563, 0.01, 1.0]
        self.THRESHOLDS = np.array([4.0])
        self.mse_mae_weights = (1.0, 1.0)
        self.normal_loss_global_scale = 1.0 / 1875.0
        self.pre_parameters = (0.0001, 7879, 0.7)
        self.activity_radius = 3
        self.evaluation_threshold = 100.0
        self.layerAccess = ([427, 502, 611, 786, 1133], [2706, 3053, 3228, 3337, 3412])

        # The architecture is unchanged; only the first convolution follows the
        # selected input-channel count.
        self.encoder = [
            [
                OrderedDict({"conv1_leaky_1": [self.in_channels, 8, 7, 5, 3]}),
                OrderedDict({"conv2_leaky_1": [64, 192, 5, 4, 2]}),
                OrderedDict({"conv3_leaky_1": [192, 192, 3, 1, 1]}),
            ],
            [
                ConvLstm(8, 64, self.sequences_in, 3),
                ConvLstm(192, 192, self.sequences_in, 3),
                ConvLstm(192, 192, self.sequences_in, 3),
            ],
        ]
        self.decoder = [
            [
                OrderedDict({"deconv1_leaky_1": [192, 192, 3, 1, 1]}),
                OrderedDict({"deconv2_leaky_1": [192, 64, 4, 4, 0]}),
                OrderedDict(
                    {
                        "deconv3_relu_1": [64, 8, 7, 5, 1],
                        "conv3_leaky_2": [8, 8, 3, 1, 1],
                        "conv3_3": [8, 1, 1, 1, 0],
                    }
                ),
            ],
            [
                ConvLstm(192, 192, self.sequences_out, 3),
                ConvLstm(192, 192, self.sequences_out, 3),
                ConvLstm(64, 64, self.sequences_out, 3),
            ],
        ]


config = Config()
