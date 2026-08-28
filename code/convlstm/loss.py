from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import config


def mag_to_pixel(magnitude, start_mag=4.0, start_value=100.0):
    if magnitude < start_mag:
        return 0.0
    lower = np.log10(0.5)
    upper = np.log10(10.0 - (start_mag - 0.5))
    value = np.log10(magnitude - (start_mag - 0.5))
    return (value - lower) / (upper - lower) * (255.0 - start_value) + start_value


class WeightedMseMae(nn.Module):
    """Weighted MSE-MAE loss used for the ConvLSTM experiments."""

    def __init__(self, weight_map_path=None):
        super().__init__()
        path = Path(config.sample if weight_map_path is None else weight_map_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Loss weight map not found: {path}. Download loss_weight_map_mw4.npz "
                "from the Zenodo dataset and place it at this path."
            )
        with np.load(path) as data:
            area = np.asarray(data["sequence_array"], dtype=np.float32)
        self.register_buffer("area", torch.from_numpy(area))

    def forward(self, prediction, target):
        area = self.area
        while area.ndim < prediction.ndim:
            area = area.unsqueeze(0)
        area = area.expand_as(prediction)

        background, mapped_area, _ = config.balancing_weights
        weights = torch.full_like(prediction, background)
        weights = weights + area * (mapped_area - background)

        target_255 = target * 255.0
        for index, magnitude in enumerate(config.THRESHOLDS):
            threshold = mag_to_pixel(float(magnitude), config.start_mag, config.start_value)
            increment = config.balancing_weights[index + 2] - config.balancing_weights[index + 1]
            weights = weights + increment * (target_255 >= threshold).to(prediction.dtype)

        reduce_dims = tuple(range(1, prediction.ndim))
        mse = torch.sum(weights * (prediction - target) ** 2, dim=reduce_dims)
        mae = torch.sum(weights * torch.abs(prediction - target), dim=reduce_dims)
        mse_weight, mae_weight = config.mse_mae_weights
        return config.normal_loss_global_scale * (
            mse_weight * torch.mean(mse) + mae_weight * torch.mean(mae)
        )
