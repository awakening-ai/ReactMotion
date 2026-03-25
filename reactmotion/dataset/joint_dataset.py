#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JointDataset: merges ReactMotionNet + HumanML3DDataset for co-training.

Uses WeightedRandomSampler to control the ratio of T2M vs ReactMotion samples.
"""

from typing import List

import torch
from torch.utils.data import Dataset, WeightedRandomSampler, ConcatDataset


class JointDataset(Dataset):
    """
    Concatenates two datasets (ReactMotionNet + HumanML3DDataset).
    Adds `is_t2m` flag to ReactMotionNet items (False) since HumanML3D
    items already have it set to True.
    """

    def __init__(self, reactmotion_ds: Dataset, t2m_ds: Dataset):
        self.rm_ds = reactmotion_ds
        self.t2m_ds = t2m_ds
        self.rm_len = len(reactmotion_ds)
        self.t2m_len = len(t2m_ds)

    def __len__(self):
        return self.rm_len + self.t2m_len

    def __getitem__(self, idx: int):
        if idx < self.rm_len:
            item = self.rm_ds[idx]
            if "is_t2m" not in item:
                item["is_t2m"] = False
            return item
        else:
            return self.t2m_ds[idx - self.rm_len]


def build_weighted_sampler(
    rm_size: int,
    t2m_size: int,
    t2m_ratio: float = 0.33,
    num_samples: int = 0,
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that yields ~t2m_ratio fraction of T2M samples.

    Args:
        rm_size: number of ReactMotion samples
        t2m_size: number of T2M samples
        t2m_ratio: desired fraction of T2M samples in each epoch (e.g., 0.33)
        num_samples: total samples per epoch (0 = rm_size + t2m_size)
    """
    total = rm_size + t2m_size
    if num_samples <= 0:
        num_samples = total

    # Compute per-sample weights so the expected ratio matches t2m_ratio
    # P(pick T2M sample) = w_t2m * t2m_size / (w_rm * rm_size + w_t2m * t2m_size)
    # We want this = t2m_ratio
    # Set w_rm = (1 - t2m_ratio) / rm_size, w_t2m = t2m_ratio / t2m_size
    rm_ratio = 1.0 - t2m_ratio

    w_rm = rm_ratio / max(rm_size, 1)
    w_t2m = t2m_ratio / max(t2m_size, 1)

    weights = [w_rm] * rm_size + [w_t2m] * t2m_size

    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )
