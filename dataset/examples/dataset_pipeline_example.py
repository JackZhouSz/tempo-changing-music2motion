#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script to verify the dataset processing pipeline (Python parts).

This script demonstrates:
1. Step 1: Adding T-pose to raw BVH clips.
2. Step 3: Converting SMPL BVH (assumed to be output by MotionBuilder) to SMPL NPZ.

Note: Step 2 (MotionBuilder) requires running inside Autodesk MotionBuilder,
so it is skipped here. We use the provided example data 'bvhSMPL' as the
hypothetical output of Step 2.
"""

import os
import sys
import shutil
import argparse

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, repo_root)

from dataset.python import add_tpose_and_rename_clips
from dataset.python import smpl_bvh_to_smpl_npz

def run_pipeline_example():
    print(f"Running full dataset pipeline verification.")
    
    # Define directories based on the plan:
    # Source: exampleData/bvh/clip_001.bvh (The raw reference data)
    # Working Dir: exampleData/bvh_to_smpl_example/ (The new area for conversion)
    
    example_data_dir = os.path.join(repo_root, "exampleData")
    src_bvh_file = os.path.join(example_data_dir, "bvh", "clip_001.bvh")
    
    if not os.path.exists(src_bvh_file):
        print(f"Error: Source file not found: {src_bvh_file}")
        return

    # The root for our conversion example
    work_dir = os.path.join(example_data_dir, "bvh_to_smpl_example")
    
    # Clean previous run if exists
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir)
    
    # Subdirectories
    dir_step1_raw = os.path.join(work_dir, "bvh_raw")
    dir_step1_ready = os.path.join(work_dir, "bvhForC")
    dir_step2_output = os.path.join(dir_step1_ready, "output") # Where MB outputs
    dir_step3_smpl = os.path.join(work_dir, "bvhSMPL")
    dir_step3_npz = os.path.join(work_dir, "npz")
    
    print(f"Working directory: {work_dir}")

    # --- Step 1: Prepare BVH (Add T-pose) ---
    print("\n[Step 1] Preparing raw BVH clips (add T-pose)...")
    
    # Copy the raw file to a 'bvh_raw' folder to simulate a dataset folder
    os.makedirs(dir_step1_raw, exist_ok=True)
    shutil.copy(src_bvh_file, os.path.join(dir_step1_raw, "clip_001.bvh"))
    
    # Run the preparation logic
    # This reads from 'bvh_raw', renames (optional), adds T-pose, writes to 'bvhForC'
    bvh_paths = add_tpose_and_rename_clips.copy_and_rename(dir_step1_raw, dir_step1_raw) # We just reuse the dir for renaming
    # Note: copy_and_rename outputs to dst_raw_dir. We can just use it in place or specific dir.
    # Let's just call prepend_tpose_frame directly on the file we copied.
    
    add_tpose_and_rename_clips.prepend_tpose_frame([os.path.join(dir_step1_raw, "clip_001.bvh")], dir_step1_ready)
    
    print(f" -> Output generated in {dir_step1_ready}")
    print(f" -> Ready for MotionBuilder.")

    # --- Step 2: MotionBuilder (Simulation) ---
    print("\n[Step 2] MotionBuilder retargeting...")
    print(" -> INSTRUCTIONS: Open MotionBuilder, run 'dataset/motionbuilder/PuppetToSmpl.py'.")
    print(" -> It will read from 'exampleData/bvh_to_smpl_example/bvhForC' and output to 'output' subdir.")
    
    # Since we are verifying the python script, we mock the MB output using pre-calculated data
    print(" -> [Simulation] Copying pre-calculated SMPL BVH to simulate MB output...")
    os.makedirs(dir_step2_output, exist_ok=True)
    
    # Use the provided sample result 'clip_001Re.bvh' as the mock output
    sample_result_bvh = os.path.join(example_data_dir, "bvhSMPL", "clip_001Re.bvh")
    if os.path.exists(sample_result_bvh):
        shutil.copy(sample_result_bvh, os.path.join(dir_step2_output, "clip_001Re.bvh"))
        print(" -> Mock output created.")
    else:
        print(" -> Warning: Pre-calculated sample 'clip_001Re.bvh' not found. Step 3 will fail unless you run MB manually.")

    # --- Step 3: SMPL BVH -> SMPL NPZ ---
    print("\n[Step 3] Converting SMPL BVH to NPZ...")
    
    if not os.path.exists(os.path.join(dir_step2_output, "clip_001Re.bvh")):
        print(" -> Skipping Step 3 (No input files).")
        return

    # We need a reference SMPL T-pose BVH for the header.
    # We copied 'smpl-T.bvh' to dataset/motionbuilder, let's use that.
    ref_bvh_path = os.path.join(repo_root, "dataset", "motionbuilder", "smpl-T.bvh")
    
    if not os.path.exists(ref_bvh_path):
        # Fallback: use the output file itself if it serves as a valid structure reference
        ref_bvh_path = os.path.join(dir_step2_output, "clip_001Re.bvh")
    
    smpl_bvh_to_smpl_npz.build_smpl_bvh(
        root=work_dir,
        smpl_t_bvh_path=ref_bvh_path,
        input_bvh_dir="bvhForC/output", # Relative to root=work_dir
        out_bvh_smpl_dir="bvhSMPL",     # Relative to root=work_dir
    )
    
    smpl_bvh_to_smpl_npz.build_smpl_npz(
        root=work_dir,
        in_bvh_smpl_dir="bvhSMPL",
        out_npz_dir="npz"
    )
    
    print(f" -> SMPL NPZ generated in {dir_step3_npz}")
    print(f" -> Files: {os.listdir(dir_step3_npz) if os.path.exists(dir_step3_npz) else 'None'}")
    print("\nFull pipeline verification completed!")

if __name__ == "__main__":
    run_pipeline_test()

