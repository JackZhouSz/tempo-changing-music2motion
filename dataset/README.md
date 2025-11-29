# Dataset Tools

This folder contains Python scripts and MotionBuilder scripts for processing the JoruriPuppet dataset.

See the main [README](../README.md#dataset-pipeline-from-puppet-bvh-to-smpl-bvh--npz--trc) for the full workflow and usage instructions.

## Contents

- `python/` : Python utilities for BVH preprocessing (T-pose insertion) and SMPL NPZ conversion.
- `motionbuilder/` : Python scripts to run inside Autodesk MotionBuilder for retargeting.
- `examples/` : Scripts verifying the pipeline steps (e.g., `dataset_pipeline_example.py`).
