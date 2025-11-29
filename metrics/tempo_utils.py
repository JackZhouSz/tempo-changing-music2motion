#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tempo and beat utilities for the JoruriPuppet metrics and audio features.

All functions here are thin, explicit wrappers around Madmom and NumPy,
kept small on purpose so that users can easily adapt them to their own
pipelines while staying consistent with the paper and appendix.
"""

from __future__ import annotations

from typing import Tuple

import madmom
import numpy as np


def get_beat_times(audio_path: str, fps: int = 100) -> np.ndarray:
    """
    Extract beat times (in seconds) from an audio file using Madmom.

    Parameters
    ----------
    audio_path : str
        Path to an audio file (e.g., WAV).
    fps : int, optional
        Frame rate for the DBN beat tracker, by default 100.

    Returns
    -------
    beat_times : np.ndarray, shape (n_beats,)
        Beat times in seconds.
    """
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=fps)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    beat_times = proc(act)
    return beat_times


def tempo_and_diff_from_beats(
    beat_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous tempo (BPM) and its first-order difference ΔBPM
    from beat times, as described in the paper and appendix.

    Let Δt_k be the duration (in seconds) between beat k and beat k+1.

    BPM_k      = 60 / Δt_k
    ΔBPM_k     = BPM_k - BPM_{k-1}

    Parameters
    ----------
    beat_times : np.ndarray, shape (n_beats,)
        Beat times in seconds.

    Returns
    -------
    bpm : np.ndarray, shape (n_beats-1,)
        Instantaneous BPM for each beat interval.
    bpm_diff : np.ndarray, shape (n_beats-2,)
        First-order difference of BPM (ΔBPM).
    """
    if beat_times.ndim != 1 or beat_times.size < 3:
        raise ValueError("beat_times must be 1D with at least 3 entries.")

    intervals = np.diff(beat_times)  # seconds between beats
    if np.any(intervals <= 0):
        raise ValueError("beat_times must be strictly increasing.")

    bpm = 60.0 / intervals
    bpm_diff = np.diff(bpm)
    return bpm, bpm_diff


def tempo_and_diff_interpolated(
    beat_times: np.ndarray,
    target_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous BPM and ΔBPM and linearly interpolate them
    onto a regular frame grid of length `target_length`.

    This is useful when you want per-frame tempo features that align
    with either audio frames or motion frames.

    Parameters
    ----------
    beat_times : np.ndarray, shape (n_beats,)
        Beat times in seconds.
    target_length : int
        Number of frames to interpolate onto.

    Returns
    -------
    bpm_interp : np.ndarray, shape (target_length,)
        Interpolated BPM sequence.
    bpm_diff_interp : np.ndarray, shape (target_length,)
        Interpolated ΔBPM sequence.
    """
    bpm, bpm_diff = tempo_and_diff_from_beats(beat_times)

    # Time stamps for BPM at midpoints of beat intervals
    bpm_times = 0.5 * (beat_times[:-1] + beat_times[1:])
    # Time stamps for ΔBPM at midpoints of BPM intervals
    bpm_diff_times = 0.5 * (bpm_times[:-1] + bpm_times[1:])

    if target_length <= 1:
        raise ValueError("target_length must be >= 2.")

    # Construct a synthetic time axis; caller is responsible for mapping
    # it to audio or motion time. For many use cases, beat_times[-1] is
    # close to the audio duration.
    audio_duration = beat_times[-1]
    new_times = np.linspace(0.0, audio_duration, target_length)

    bpm_interp = np.interp(new_times, bpm_times, bpm)
    bpm_diff_interp = np.interp(new_times, bpm_diff_times, bpm_diff)
    return bpm_interp, bpm_diff_interp


