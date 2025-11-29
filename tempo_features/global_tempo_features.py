#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global (audio-time) tempo-changing features.

This module is a cleaned-up version of `extract_musicfea37_tempoChanges.py`,
generalized into reusable functions.

Given a music file, it extracts:

- Onset envelope (1D),
- MFCC (20 dims),
- Chroma (12 dims),
- Onset peak one-hot,
- Beat position one-hot,
- Instantaneous BPM and ΔBPM (2 dims, from Madmom beat times),

and concatenates them into a (T, 37) feature matrix.
"""

from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np

from metrics import tempo_utils


def extract_global_tempo_features(
    audio_path: str,
    fps: int = 30,
    hop_length: int = 512,
    n_mfcc: int = 20,
    n_chroma: int = 12,
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (35 + 2)-dimensional music features with global tempo-changing
    information interpolated over the audio frame grid.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    fps : int, optional
        Virtual frame rate, by default 30. Used together with hop_length
        to set the analysis sampling rate: sr = fps * hop_length.
    hop_length : int, optional
        Hop length for STFT-based features, by default 512.
    n_mfcc : int, optional
        Number of MFCC coefficients, by default 20.
    n_chroma : int, optional
        Number of chroma bins, by default 12.
    target_length : int, optional
        If given, the output will be interpolated (or truncated) to
        exactly `target_length` frames.

    Returns
    -------
    features : np.ndarray, shape (T, 35 + 2)
        Concatenated features:
        [envelope, MFCC(20), Chroma(12), onset_onehot, beat_onehot, BPM, ΔBPM].
    bpm_interp : np.ndarray, shape (T,)
        Interpolated instantaneous BPM.
    bpm_diff_interp : np.ndarray, shape (T,)
        Interpolated ΔBPM.
    """
    sr = fps * hop_length

    # Basic features
    y, _ = librosa.load(audio_path, sr=sr)

    envelope = librosa.onset.onset_strength(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    chroma = librosa.feature.chroma_cens(
        y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma
    ).T

    # Onset peaks
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=sr, hop_length=hop_length
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)
    peak_onehot[peak_idxs] = 1.0

    # Beat positions (Librosa)
    start_bpm = librosa.beat.tempo(y=y, sr=sr)[0]
    tempo_lib, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=sr,
        hop_length=hop_length,
        start_bpm=start_bpm,
        tightness=100,
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    # Tempo-changing features from Madmom
    beat_times = tempo_utils.get_beat_times(audio_path, fps=100)
    bpm_interp, bpm_diff_interp = tempo_utils.tempo_and_diff_interpolated(
        beat_times, target_length=envelope.shape[0]
    )

    # Stack to (T, 35 + 2)
    features = np.concatenate(
        [
            envelope[:, None],
            mfcc,
            chroma,
            peak_onehot[:, None],
            beat_onehot[:, None],
            bpm_interp[:, None],
            bpm_diff_interp[:, None],
        ],
        axis=-1,
    )

    if target_length is not None and target_length > 0:
        # Simple temporal resampling to target_length
        T = features.shape[0]
        idx = np.linspace(0, T - 1, target_length)
        features = np.stack(
            [np.interp(idx, np.arange(T), features[:, d]) for d in range(features.shape[1])],
            axis=-1,
        )
        bpm_interp = np.interp(idx, np.arange(T), bpm_interp)
        bpm_diff_interp = np.interp(idx, np.arange(T), bpm_diff_interp)

    return features.astype(np.float32), bpm_interp.astype(np.float32), bpm_diff_interp.astype(
        np.float32
    )


