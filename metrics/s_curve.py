#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion aesthetic (S-curve) metric.

This module implements a simplified version of the S-curve metric used
in the paper, operating directly on 3D joint position trajectories.

The basic idea:
- Project a 3D trajectory onto a 2D PCA plane,
- Measure how much the curve bends away from the straight line between
  its start and end (sagitta-based curvature percentage),
- Count beat segments whose curvature falls inside a desirable range
  (from user studies) and average over all segments.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA

from . import tempo_utils


def calculate_curvature_percentage(positions: np.ndarray) -> float:
    """
    Calculate curvature percentage of a 3D trajectory using PCA + sagitta.

    Parameters
    ----------
    positions : np.ndarray, shape (T, 3)
        3D positions of a single joint over time.

    Returns
    -------
    curvature_percentage : float
        Sagitta-based curvature percentage in [0, +inf). NaN if degenerate.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (T, 3).")

    if positions.shape[0] < 3:
        return float("nan")

    # Project to 2D PCA plane
    pca = PCA(n_components=2)
    motion_2d = pca.fit_transform(positions)
    x = motion_2d[:, 0]
    y = motion_2d[:, 1]

    # Line from first to last point: Ax + By + C = 0
    A = y[-1] - y[0]
    B = x[0] - x[-1]
    C = x[-1] * y[0] - x[0] * y[-1]

    # Distances of all points to the line
    denom = np.sqrt(A * A + B * B)
    if denom == 0.0:
        return float("nan")

    distances = np.abs(A * x + B * y + C) / denom
    max_distance = np.max(distances)
    total_height = np.linalg.norm(motion_2d[0] - motion_2d[-1])
    if total_height == 0.0:
        return float("nan")

    curvature_percentage = (max_distance / total_height) * 100.0
    return float(curvature_percentage)


def s_curve_score_from_positions(
    positions: np.ndarray,
    beat_times: np.ndarray,
    fps: float,
    min_pct: float = 20.0,
    max_pct: float = 60.0,
) -> float:
    """
    Compute S-curve score for a single joint trajectory given beat times.

    For each beat segment, we:
    - cut the trajectory between frame indices corresponding to two
      consecutive beats,
    - compute curvature percentage,
    - count it as 1 if min_pct <= curvature <= max_pct, else 0.

    The final score is the mean of these indicator values across all
    valid segments, consistent with the definition in the paper.

    Parameters
    ----------
    positions : np.ndarray, shape (T, 3)
        3D positions of a single joint (e.g., head or hand).
    beat_times : np.ndarray, shape (n_beats,)
        Beat times in seconds.
    fps : float
        Frame rate of the motion (frames per second).
    min_pct : float, optional
        Lower bound of the desirable curvature percentage range,
        by default 20.
    max_pct : float, optional
        Upper bound of the desirable curvature percentage range,
        by default 60.

    Returns
    -------
    score : float
        Mean indicator score in [0, 1]. NaN if no valid segments.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (T, 3).")

    T = positions.shape[0]
    frame_times = np.arange(T) / float(fps)
    beat_frames = np.searchsorted(frame_times, beat_times).astype(int)

    values = []
    for i in range(len(beat_frames) - 1):
        start = beat_frames[i]
        end = beat_frames[i + 1]
        start = max(0, min(start, T - 2))
        end = max(start + 1, min(end, T))

        segment = positions[start:end]
        curvature = calculate_curvature_percentage(segment)
        if np.isnan(curvature):
            continue
        if min_pct <= curvature <= max_pct:
            values.append(1.0)
        else:
            values.append(0.0)

    if not values:
        return float("nan")
    return float(np.mean(values))


def s_curve_scores_head_and_hand(
    head_positions: np.ndarray,
    hand_positions: np.ndarray,
    beat_times: np.ndarray,
    fps: float,
    min_pct: float = 20.0,
    max_pct: float = 60.0,
) -> Tuple[float, float]:
    """
    Convenience wrapper to compute S-curve scores for head and hand.

    Parameters
    ----------
    head_positions : np.ndarray, shape (T, 3)
        3D positions of the head joint.
    hand_positions : np.ndarray, shape (T, 3)
        3D positions of a hand joint (e.g., right hand).
    beat_times : np.ndarray
        Beat times in seconds.
    fps : float
        Motion frame rate.
    min_pct, max_pct : float
        Desirable curvature percentage range.

    Returns
    -------
    (head_score, hand_score) : Tuple[float, float]
        S-curve scores for head and hand, respectively.
    """
    head_score = s_curve_score_from_positions(
        head_positions, beat_times, fps, min_pct=min_pct, max_pct=max_pct
    )
    hand_score = s_curve_score_from_positions(
        hand_positions, beat_times, fps, min_pct=min_pct, max_pct=max_pct
    )
    return head_score, hand_score


def s_curve_scores_from_audio_and_positions(
    audio_path: str,
    head_positions: np.ndarray,
    hand_positions: np.ndarray,
    fps: float,
    fps_madmom: int = 100,
    min_pct: float = 20.0,
    max_pct: float = 60.0,
) -> Tuple[float, float]:
    """
    High-level helper that extracts beat times from audio and then
    computes S-curve scores for head and hand.
    """
    beat_times = tempo_utils.get_beat_times(audio_path, fps=fps_madmom)
    return s_curve_scores_head_and_hand(
        head_positions=head_positions,
        hand_positions=hand_positions,
        beat_times=beat_times,
        fps=fps,
        min_pct=min_pct,
        max_pct=max_pct,
    )


