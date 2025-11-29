"""
I/O utilities for reading motion files (BVH, TRC, NPZ) for metrics evaluation.
"""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Reuse the existing BVH reader from dataset/python
from dataset.python import bvh as bvh_utils


def read_bvh(path: str) -> Tuple[np.ndarray, float, Dict[str, int]]:
    """
    Read a BVH file and return motion data, frame time, and a joint map.

    Parameters
    ----------
    path : str
        Path to the BVH file.

    Returns
    -------
    data : np.ndarray
        Motion data matrix (T, D).
    frame_time : float
        Frame time in seconds.
    joint_map : Dict[str, int]
        A simplified mapping from joint name to channel index start.
        (Note: The minimal bvh parser in this repo doesn't return a full hierarchy
         object, so we might only get raw data. For accurate joint positions,
         use TRC or a full FK engine.)
    """
    data, fs_str, header = bvh_utils.bvhreader(path)
    try:
        frame_time = float(fs_str)
    except ValueError:
        frame_time = float(fs_str.strip())
    
    # Parse header to find joint names and their channel offsets
    # This is a naive parser that assumes 3/6 channels per JOINT/ROOT
    joint_map = {}
    channel_count = 0
    
    # Simple parsing of hierarchy to guess channel mapping if needed
    # (Skipping complex hierarchy parsing for now as metrics use positions mostly from TRC)
    
    return data, frame_time, {}


def read_trc(path: str) -> Tuple[Dict[str, np.ndarray], float, int]:
    """
    Read a TRC file and return a dictionary of joint positions.

    Parameters
    ----------
    path : str
        Path to the TRC file.

    Returns
    -------
    joints_dict : Dict[str, np.ndarray]
        Dictionary where keys are marker names (e.g. 'Head') and values
        are (T, 3) numpy arrays of positions.
    frame_rate : float
        Data rate (FPS) from the file header.
    num_frames : int
        Number of frames.
    """
    with open(path, 'r') as f:
        header_lines = [next(f) for _ in range(5)]

    # Line 3 contains metadata (DataRate, CameraRate, NumFrames, etc.)
    meta_info = header_lines[2].strip().split('\t')
    try:
        data_rate = float(meta_info[0])
        num_frames = int(meta_info[2])
    except (ValueError, IndexError):
        # Fallback if tab splitting fails or format slightly differs
        # Try splitting by whitespace
        meta_info = header_lines[2].strip().split()
        data_rate = float(meta_info[0])
        num_frames = int(meta_info[2])

    # Line 4 contains marker names
    # Format: Frame# Time reference ... Marker1 ... Marker2 ...
    # Markers usually have 3 columns (X, Y, Z) but the name only appears once
    header_row = header_lines[3].strip().split('\t')
    
    # Find marker names and their starting column indices
    # Usually, columns 0 and 1 are Frame# and Time. Markers start around col 2.
    # Valid marker names are non-empty strings that are not "Frame#", "Time", etc.
    marker_names = []
    col_indices = []
    
    # Skip first two columns (Frame#, Time)
    current_col = 2
    for item in header_row[2:]:
        if item and item.strip():
            marker_names.append(item.strip())
            col_indices.append(current_col)
            # Each marker typically takes 3 columns (X, Y, Z)
            current_col += 3
        else:
            # Empty columns between markers? Or X/Y/Z subheaders?
            # In standard TRC, the header row has empty tabs for Y and Z columns of a marker
            current_col += 1
            
    # Load data starting from line 6 (0-indexed line 5 is subheader X1 Y1 Z1...)
    # Pandas is robust for this
    df = pd.read_csv(path, sep='\t', skiprows=5, header=None)
    
    # If pandas read extra empty columns at the end, drop them
    df = df.dropna(axis=1, how='all')
    
    data = df.values
    joints_dict = {}
    
    # Re-scan col_indices based on marker_names and the logic that each has 3 columns
    # We assume the markers appear in order in the file corresponding to the names we found
    
    # Better approach: Identify columns by skipping Frame# and Time
    # Frame# is col 0, Time is col 1
    # Markers start at col 2.
    
    # Let's manually map names to columns assuming 3 cols per marker
    # The list 'marker_names' should correspond to blocks of 3 columns
    
    current_data_col = 2
    for name in marker_names:
        # Extract 3 columns
        if current_data_col + 3 <= data.shape[1]:
            joints_dict[name] = data[:, current_data_col : current_data_col + 3]
            current_data_col += 3
        else:
            break
            
    return joints_dict, data_rate, num_frames


def read_smpl_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read an SMPL NPZ file (trans, poses).

    Parameters
    ----------
    path : str
        Path to .npz file.

    Returns
    -------
    trans : np.ndarray
        Root translation (T, 3).
    poses : np.ndarray
        Joint rotations (axis-angle) (T, 24, 3).
    """
    data = np.load(path)
    return data['trans'], data['poses']

