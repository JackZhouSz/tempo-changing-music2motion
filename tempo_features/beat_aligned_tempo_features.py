#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beat-aligned (motion-time) tempo-changing features.

This module is a cleaned-up and generalized version of the logic used in
`bvhFeatureExportAllmotorica-tempoNewAlignT.py`, focusing only on the
audio side and the alignment to motion frames.

Given:
- an audio file,
- the motion frame rate,
- the desired number of motion frames,

it produces:
- chroma (first dimension),
- spectral flux,
- beat activation (one-hot),
- instantaneous BPM and ΔBPM,

all aligned to the motion frame grid.
"""

from __future__ import annotations

from typing import Dict

import librosa
import numpy as np

from metrics import tempo_utils


def extract_beat_aligned_tempo_features(
    audio_path: str,
    motion_fps: float,
    n_frames: int,
) -> Dict[str, np.ndarray]:
    """
    Extract beat-aligned tempo-changing features aligned to motion frames.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    motion_fps : float
        Motion frame rate (frames per second).
    n_frames : int
        Number of motion frames to align to.

    Returns
    -------
    features : Dict[str, np.ndarray]
        A dictionary with keys:
        - 'chroma_0'      : shape (n_frames,)
        - 'spectral_flux' : shape (n_frames,)
        - 'beat_activation' : shape (n_frames,)
        - 'tempo'         : shape (n_frames,)
        - 'tempo_diff'    : shape (n_frames,)
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")

    # Load audio at native sampling rate
    y, sr = librosa.load(audio_path, sr=None)

    # Choose hop_length so that audio frames roughly match motion frames
    hop_length = int(sr / motion_fps)

    # Chroma and spectral flux on audio-time grid
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).T
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Beat times from Madmom
    beat_times = tempo_utils.get_beat_times(audio_path, fps=100)

    # Map beat times to motion frames
    beat_positions = np.round(beat_times * motion_fps).astype(int)
    beat_activation = np.zeros(n_frames, dtype=np.float32)
    valid_idx = beat_positions[beat_positions < n_frames]
    beat_activation[valid_idx] = 1.0

    # Tempo-changing features interpolated to n_frames
    bpm_interp, bpm_diff_interp = tempo_utils.tempo_and_diff_interpolated(
        beat_times, target_length=n_frames
    )

    # Align audio-time features to n_frames via simple indexing / interpolation
    T_audio = spectral_flux.shape[0]
    if T_audio == 0:
        raise RuntimeError("Failed to extract audio features (spectral flux length is 0).")

    idx = np.linspace(0, T_audio - 1, n_frames)

    chroma0 = np.interp(idx, np.arange(T_audio), chroma[:T_audio, 0])
    spectral_flux_aligned = np.interp(idx, np.arange(T_audio), spectral_flux)

    return {
        "chroma_0": chroma0.astype(np.float32),
        "spectral_flux": spectral_flux_aligned.astype(np.float32),
        "beat_activation": beat_activation.astype(np.float32),
        "tempo": bpm_interp.astype(np.float32),
        "tempo_diff": bpm_diff_interp.astype(np.float32),
    }


