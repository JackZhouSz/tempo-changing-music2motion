#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight BVH reader / writer utilities, adapted from the internal
tools used for JoruriPuppet data processing.

This module is intentionally minimal and only supports the subset of
BVH features needed in this project:

- reading motion data as a NumPy array,
- preserving the original header lines,
- writing motion data back with the original hierarchy.

The API is compatible with the original scripts:

- bvhreader(path) -> (data, frame_time, header_lines)
- bvhoutput(data, frame_time, name_without_ext, header_lines)
- errc / errb for fixing large Euler angle jumps.
"""

import numpy as np


def bvhreader(path):
    """Read a BVH file.

    Parameters
    ----------
    path : str
        Path to a `.bvh` file.

    Returns
    -------
    data : np.ndarray
        Motion data of shape (T, D), where T is the number of frames.
    fs : str
        Frame time string as it appears in the BVH file (e.g. '0.0333333').
    header_lines : list of str
        Lines from the start of the file up to (and including) 'MOTION' and
        'Frames' / 'Frame Time' lines, which can be reused when writing out.
    """
    with open(path) as f:
        lines = f.readlines()

    # Find the "MOTION" line that separates hierarchy and data
    data_start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            data_start_idx = i
            break

    if data_start_idx is None:
        raise ValueError(f"No 'MOTION' section found in BVH file: {path}")

    # Frame time is usually on the second line after 'MOTION'
    fs_line = lines[data_start_idx + 2]
    fs = fs_line[12:].strip()  # strip "Frame Time:"

    # Motion data lines start after the frame header
    motion_lines = lines[data_start_idx + 3 :]
    # Use a temporary text buffer via numpy
    from io import StringIO

    buf = StringIO("".join(motion_lines))
    data = np.loadtxt(buf)

    header_lines = lines[: data_start_idx + 1]
    return data, fs, header_lines


def bvhoutput(data, fs, name, header_lines):
    """Write motion data back to a BVH file.

    Parameters
    ----------
    data : np.ndarray
        Motion data of shape (T, D).
    fs : str or float
        Frame time. Will be written after the 'Frame Time:' label.
    name : str
        Output path **without** the `.bvh` extension.
    header_lines : list of str
        Header lines as returned by `bvhreader`.
    """
    path_w = name + ".bvh"

    # First write only the numeric data to a temporary file
    tmp_path = path_w
    np.savetxt(tmp_path, data, delimiter=" ", fmt="%.4e")

    with open(tmp_path) as f:
        motion_lines = f.readlines()

    n_frames = data.shape[0]
    frame_header = [
        f"Frames: {n_frames}\n",
        f"Frame Time: {fs}\n",
    ]

    all_lines = header_lines + frame_header + motion_lines

    with open(path_w, mode="w") as f:
        f.writelines(all_lines)


def errc(data, start, end):
    """Fix large discontinuities in Euler angle channels.

    This function is kept for backward compatibility with the original
    scripts. It unwraps angles by adding / subtracting 360 degrees when
    a jump larger than 350 degrees is observed between consecutive frames.
    """
    n = data.shape[0]
    for j in range(start, end):
        for i in range(n - 1):
            if data[i, j] - data[i + 1, j] < -350:
                data[i + 1 :, j] = data[i + 1 :, j] - 360
            if data[i, j] - data[i + 1, j] > 350:
                data[i + 1 :, j] = data[i + 1 :, j] + 360
    return data


def errb(data, start, end):
    """Clamp Euler angle channels into [-180, 180] with wrapping."""
    n = data.shape[0]
    for j in range(start, end):
        for i in range(n):
            if data[i, j] < -180:
                data[i:, j] = data[i:, j] + 360
            if data[i, j] > 180:
                data[i:, j] = data[i:, j] - 360
    return data


