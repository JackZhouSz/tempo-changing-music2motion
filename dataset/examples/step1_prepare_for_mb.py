#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 of the Dataset Pipeline: Prepare BVH files for MotionBuilder.

This script:
1. Reads raw BVH files from 'exampleData/bvh'.
2. Inserts a T-pose frame at the beginning of each clip.
3. Saves the ready-to-process files into 'exampleData/bvh_to_smpl_example/bvhForC'.

After running this, you should open Autodesk MotionBuilder and run
'dataset/motionbuilder/PuppetToSmpl.py'.
"""

import os
import sys
import shutil

# Add repository root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, repo_root)

from dataset.python import add_tpose_and_rename_clips

def run_step1_prepare():
    print(f"Running Step 1: Prepare BVH for MotionBuilder")
    
    # Source raw data
    example_data_dir = os.path.join(repo_root, "exampleData")
    src_bvh_file = os.path.join(example_data_dir, "bvh", "clip_001.bvh")
    
    if not os.path.exists(src_bvh_file):
        print(f"Error: Source file not found: {src_bvh_file}")
        return

    # Output directory for this example workflow
    work_dir = os.path.join(example_data_dir, "bvh_to_smpl_example")
    
    # Clean up previous Step 1 output (but try to keep Step 2 output if it exists?)
    # For a clean start, we might want to clear bvhForC
    dir_step1_ready = os.path.join(work_dir, "bvhForC")
    
    if os.path.exists(dir_step1_ready):
        print(f"Cleaning up existing directory: {dir_step1_ready}")
        shutil.rmtree(dir_step1_ready)
    os.makedirs(dir_step1_ready)
    
    print(f"Working directory: {work_dir}")

    # --- Execution ---
    print(" -> Reading raw BVH and inserting T-pose...")
    
    # We treat src_bvh_file as a single file input. 
    # The utility function expects a list of paths.
    add_tpose_and_rename_clips.prepend_tpose_frame([src_bvh_file], dir_step1_ready)
    
    print(f" -> Output generated in: {dir_step1_ready}")
    print(f" -> Files: {os.listdir(dir_step1_ready)}")
    print("\n[Next Step]")
    print("1. Open Autodesk MotionBuilder.")
    print("2. Run the script: dataset/motionbuilder/PuppetToSmpl.py")
    print("3. It will process these files and output to 'output' subdirectory.")
    print("4. After MB finishes, run 'dataset/examples/step3_finalize_npz.py'.")

if __name__ == "__main__":
    run_step1_prepare()

