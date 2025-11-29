#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to extract tempo-changing music features.
Demonstrates:
1. Pure Madmom beat tracking (Tempo & Delta-Tempo only) - Robust mode.
2. Full Feature Extraction (Librosa + Madmom) - Optional, requires working libsndfile.
"""

import os
import sys
import numpy as np

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, repo_root)

from metrics import tempo_utils
# We import the full modules inside the functions to allow partial failure if librosa/soundfile is broken

def run_features_extraction(example_data_root):
    print(f"Extracting features for data in: {example_data_root}")
    
    wav_path = os.path.join(example_data_root, "wav", "clip_001.wav")
    bvh_path = os.path.join(example_data_root, "bvhSMPL", "clip_001Re.bvh")
    
    if not os.path.exists(wav_path):
        print("Error: WAV file not found.")
        return

    # --- Part 1: Pure Madmom (Tempo/Beats only) ---
    print("\n--- [Part 1] Madmom-only Tempo Extraction (No Librosa) ---")
    try:
        print(f"Processing {wav_path} with Madmom...")
        # This uses tempo_utils which wraps Madmom directly
        beat_times = tempo_utils.get_beat_times(wav_path, fps=100)
        print(f"Found {len(beat_times)} beats.")
        
        bpm, bpm_diff = tempo_utils.tempo_and_diff_from_beats(beat_times)
        print(f"Average BPM: {np.mean(bpm):.2f}")
        print(f"Average Delta-BPM: {np.mean(bpm_diff):.4f}")
        
        # Demonstrate interpolation to arbitrary length (e.g. 1000 frames)
        target_len = 1000
        bpm_interp, bpm_diff_interp = tempo_utils.tempo_and_diff_interpolated(beat_times, target_len)
        print(f"Interpolated to {target_len} frames: BPM shape {bpm_interp.shape}")
        
        print("-> Madmom tempo extraction successful.")
        
    except Exception as e:
        print(f"Madmom extraction failed: {e}")
        print("Note: Ensure madmom is installed and ffmpeg/audioread is working.")

    # --- Part 2: Full Features (Librosa + Madmom) ---
    print("\n--- [Part 2] Full Audio Features (Librosa + Madmom) ---")
    print("Note: This requires a working 'libsndfile' installation.")
    
    try:
        # Import here to avoid crash at script start if librosa is broken
        from tempo_features import global_tempo_features
        from tempo_features import beat_aligned_tempo_features
        from metrics import io_utils 

        print("Attempting to extract 37-dim features...")
        feats, bpm, bpm_diff = global_tempo_features.extract_global_tempo_features(
            audio_path=wav_path,
            fps=30,
            hop_length=512
        )
        
        print(f"Feature Matrix Shape: {feats.shape} (Time, 37)")
        print(f" -> Success! Full features extracted.")
        
        # Beat-aligned mode
        if os.path.exists(bvh_path):
            print("\nAttempting beat-aligned features...")
            bvh_data, frame_time, _ = io_utils.read_bvh(bvh_path)
            n_frames = bvh_data.shape[0]
            motion_fps = 1.0 / frame_time if frame_time > 0 else 120.0
            
            aligned_feats = beat_aligned_tempo_features.extract_beat_aligned_tempo_features(
                audio_path=wav_path,
                motion_fps=motion_fps,
                n_frames=n_frames
            )
            print("Aligned Features Dictionary keys:", aligned_feats.keys())
            
    except OSError as e:
        if "libsndfile" in str(e) or "sndfile" in str(e):
            print(f"Skipping Librosa part due to missing soundfile library: {e}")
            print("This is common on some Windows environments. The Madmom part (Part 1) is sufficient for tempo metrics.")
        else:
            print(f"Failed with OSError: {e}")
    except Exception as e:
        print(f"Failed to extract full features: {e}")

if __name__ == "__main__":
    # Default example data path
    # Updated to use the internal 'exampleData' inside the repo
    default_data = os.path.abspath(os.path.join(repo_root, "exampleData"))
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = default_data
        
    run_features_extraction(data_path)
