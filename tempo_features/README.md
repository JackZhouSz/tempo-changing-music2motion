# Tempo-Changing Music Features

This folder contains tools to extract tempo-changing features (BPM and ΔBPM) from audio.

See the main [README](../README.md#tempo‑changing-music-features-global-and-frame‑aligned) for details.

## Contents

- `global_tempo_features.py` : Extracts features interpolated to audio frames (e.g., for AIST++/FineDance).
- `beat_aligned_tempo_features.py` : Extracts features aligned to motion frames (e.g., for Motorica/JoruriPuppet).
- `examples/` : Example script `extract_features_example.py` demonstrating extraction.
