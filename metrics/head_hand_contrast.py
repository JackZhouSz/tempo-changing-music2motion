#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Head–hand contrast metric.

This module implements a simplified version of the theatrical appearance
metric described in the paper, focusing on the contrast between head and
hand motion on beat-segmented trajectories.

The implementation follows the structure of the original analysis code:

- Project head and hand trajectories to low-dimensional PCA spaces,
- Compute:
  - cosine similarity between 2D directions,
  - mean absolute phase (angle) difference in 2D,
  - mean absolute difference between 1D PCA projections (Xp),
- Optionally map Xp values through a Gaussian function parameterized
  by (mu_x, sigma_x) learned from training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from . import tempo_utils


@dataclass
class HeadHandContrastFeatures:
    """Contrast descriptors aggregated over all beat segments."""

    xp_mean: float


def _contrast_features_per_sequence(
    head_positions: np.ndarray,
    hand_positions: np.ndarray,
    beat_times: np.ndarray,
    fps: float,
    smooth_sigma: float = 5.0,
) -> HeadHandContrastFeatures:
    """
    Compute contrast features from head and hand trajectories.

    This is adapted from the original `calc_head_hand` implementation.
    """
    if head_positions.shape != hand_positions.shape:
        raise ValueError("head_positions and hand_positions must share the same shape.")

    if head_positions.ndim != 2 or head_positions.shape[1] != 3:
        raise ValueError("positions must have shape (T, 3).")

    T = head_positions.shape[0]
    frame_times = np.arange(T) / float(fps)
    beat_frames = np.searchsorted(frame_times, beat_times).astype(int)

    xp_segments_head = []
    xp_segments_hand = []

    for i in range(len(beat_frames) - 2):
        try:
            # Following the original script:
            #   hand: segment between beats i+1 and i+2
            #   head: segment between beats i and i+1
            start_hand = beat_frames[i + 1]
            end_hand = beat_frames[i + 2]
            start_head = beat_frames[i]
            end_head = beat_frames[i + 1]

            start_hand = max(0, min(start_hand, T - 2))
            end_hand = max(start_hand + 1, min(end_hand, T))
            start_head = max(0, min(start_head, T - 2))
            end_head = max(start_head + 1, min(end_head, T))

            seg_hand = hand_positions[start_hand:end_hand]
            seg_head = head_positions[start_head:end_head]

            seg_hand = gaussian_filter1d(seg_hand, smooth_sigma, axis=0)
            seg_head = gaussian_filter1d(seg_head, smooth_sigma, axis=0)

            # PCA to 2D for normalization (optional but kept for consistency with PCA1 logic)
            pca2 = PCA(n_components=2)
            seg_hand_2d = pca2.fit_transform(seg_hand)
            pca2 = PCA(n_components=2)
            seg_head_2d = pca2.fit_transform(seg_head)

            # PCA to 1D for Xp (Theatrical Contrast definition)
            pca1 = PCA(n_components=1)
            seg_hand_1d = pca1.fit_transform(seg_hand_2d)
            pca1 = PCA(n_components=1)
            seg_head_1d = pca1.fit_transform(seg_head_2d)

            # Resample to the same length if they differ
            # We choose to resample the longer one to the shorter one's length,
            # or simply resample both to a fixed size. 
            # Here we resample the hand segment to match the head segment length.
            len_head = len(seg_head_1d)
            len_hand = len(seg_hand_1d)
            
            if len_hand != len_head:
                if len_hand > 1:
                    # Create interpolation grid
                    old_indices = np.linspace(0, len_hand - 1, num=len_hand)
                    new_indices = np.linspace(0, len_hand - 1, num=len_head)
                    # Interpolate column 0 (PCA 1D projection)
                    seg_hand_1d = np.interp(new_indices, old_indices, seg_hand_1d.flatten())[:, None]
                else:
                    # Degenerate case
                    continue

            xp_segments_head.append(seg_head_1d)
            xp_segments_hand.append(seg_hand_1d)
        except Exception:
            continue

    if not xp_segments_head:
        return HeadHandContrastFeatures(
            xp_mean=float("nan"),
        )

    # Aggregate Xp (1D PCA projections)
    head_all = np.concatenate(xp_segments_head, axis=0)
    hand_all = np.concatenate(xp_segments_hand, axis=0)
    xp_mean = float(np.mean(np.abs(head_all - hand_all)))

    return HeadHandContrastFeatures(
        xp_mean=xp_mean,
    )


def gaussian_contrast_score(
    x: float,
    mu_x: float,
    sigma_x: float,
) -> float:
    """
    Map a contrast feature value x to a [0, 1] score using a Gaussian:

        score = exp( - (x - mu_x)^2 / (2 * sigma_x^2) )

    The parameters (mu_x, sigma_x) should be fitted from training data,
    as described in the appendix.
    """
    if sigma_x <= 0:
        raise ValueError("sigma_x must be positive.")
    return float(np.exp(-((x - mu_x) ** 2) / (2.0 * sigma_x**2)))


def head_hand_contrast_from_audio_and_positions(
    audio_path: str,
    head_positions: np.ndarray,
    hand_positions: np.ndarray,
    fps: float,
    fps_madmom: int = 100,
    smooth_sigma: float = 5.0,
) -> HeadHandContrastFeatures:
    """
    High-level helper: extract beat times from audio, then compute
    aggregated contrast features from head and hand positions.
    """
    beat_times = tempo_utils.get_beat_times(audio_path, fps=fps_madmom)
    return _contrast_features_per_sequence(
        head_positions=head_positions,
        hand_positions=hand_positions,
        beat_times=beat_times,
        fps=fps,
        smooth_sigma=smooth_sigma,
    )


