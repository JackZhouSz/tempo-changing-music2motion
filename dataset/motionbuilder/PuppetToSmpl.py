#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MotionBuilder script for retargeting puppet BVH clips to an SMPL skeleton.

This is a cleaned version of the internal script used for JoruriPuppet,
showing how to:

1. Import a prepared SMPL T-pose FBX file (with a Character already defined),
2. Import puppet BVH clips (with a prepended T-pose frame),
3. Create a new Character for each BVH,
4. Retarget motion from the puppet skeleton to the SMPL skeleton,
5. Plot (bake) the animation to the SMPL skeleton,
6. Export the resulting SMPL BVH clips.

Usage
-----
- Open this script in Autodesk MotionBuilder.
- Edit `FOLDER_PATH` and `FBX_T_POSE_PATH` below to match your setup:
  - `FOLDER_PATH` should point to the directory containing the `bvhForC`
    BVH clips prepared by `add_tpose_and_rename_clips.py`.
  - `FBX_T_POSE_PATH` should point to the SMPL T-pose FBX file that already
    has a Character definition.
- Run the script inside MotionBuilder.

The script will:
- import the SMPL T-pose FBX for each clip,
- import the corresponding BVH,
- create and characterize a new Character,
- retarget and bake the motion,
- export the result as a BVH file with suffix `Re.bvh` into a subfolder
  called `output` under `FOLDER_PATH`.
"""

import os

import pyfbsdk
from pyfbsdk import (
    FBApplication,
    FBCharacter,
    FBCharacterInputType,
    FBCharacterPlotWhere,
    FBModelSkeleton,
    FBPlotOptions,
    FBSystem,
)
import xml.etree.ElementTree as etree


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
# Using relative paths to make the script portable.
# Assumes this script is located at: repo/dataset/motionbuilder/PuppetToSmpl.py
#
# Directories:
#   Input  (Example Data): repo/exampleData/bvh_to_smpl_example/bvhForC
#   Input  (SMPL FBX):     repo/dataset/motionbuilder/smpl-male-T-pose.fbx
#   Output (Result):       repo/exampleData/bvh_to_smpl_example/bvhSMPL

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Define paths relative to the repository root
FOLDER_PATH = os.path.join(REPO_ROOT, "exampleData", "bvh_to_smpl_example", "bvhForC")
FBX_T_POSE_PATH = os.path.join(SCRIPT_DIR, "smpl-male-T-pose.fbx")
OUTPUT_DIR_NAME = "output"  # MotionBuilder script outputs here inside FOLDER_PATH first

# Verify paths exist
print(f"DEBUG: Repo Root determined as: {REPO_ROOT}")
print(f"DEBUG: Looking for Input BVH at: {FOLDER_PATH}")
print(f"DEBUG: Looking for FBX T-Pose at: {FBX_T_POSE_PATH}")

if not os.path.exists(FOLDER_PATH):
    msg = f"[Error] Input folder not found:\n{FOLDER_PATH}\n\nPlease run the Python Step 1 script first."
    print(msg)
    pyfbsdk.FBMessageBox("Script Error", msg, "OK")
    # We cannot use sys.exit() in MB as it kills MB, just stop main logic
    FOLDER_PATH = None 

if not os.path.exists(FBX_T_POSE_PATH):
    msg = f"[Error] SMPL FBX not found:\n{FBX_T_POSE_PATH}\n\nCheck if the file exists in dataset/motionbuilder."
    print(msg)
    pyfbsdk.FBMessageBox("Script Error", msg, "OK")
    FBX_T_POSE_PATH = None


def convert_skeleton_definition_xml_to_dict(xml_file_name):
    """
    Convert a Skeleton Definition XML file to a Python dictionary.

    The XML file (e.g., 'HIK.xml') should define the mapping from
    MotionBuilder character slots to joint names.
    """
    xml_file_path = os.path.join(os.path.expanduser("~"), xml_file_name)
    parsed_xml_file = etree.parse(xml_file_path)
    skel_def_dict = {}
    for line in parsed_xml_file.iter("item"):
        joint_name = line.attrib.get("value")
        if joint_name:
            slot_name = line.attrib.get("key")
            skel_def_dict[slot_name] = joint_name
    return skel_def_dict


def characterize_character(character_name, num):
    """
    Create and characterize a new Character from an imported BVH skeleton.
    """
    new_character = FBCharacter(character_name + str(num))
    char_slot_name_joint_name_dict = convert_skeleton_definition_xml_to_dict(
        "HIK.xml"
    )
    for slot_name, joint_name in char_slot_name_joint_name_dict.items():
        mapping_slot = new_character.PropertyList.Find(slot_name + "Link")
        joint_obj = pyfbsdk.FBFindModelByLabelName(
            f"BVH {num}:{joint_name}"
        )
        if joint_obj:
            mapping_slot.append(joint_obj)

    characterized = new_character.SetCharacterizeOn(True)
    if not characterized:
        print(new_character.GetCharacterizeError())
    else:
        FBApplication().CurrentCharacter = new_character
    return new_character


def plot_character():
    """Bake the current Character's animation onto its skeleton."""
    l_character = FBApplication().CurrentCharacter

    plot_options = FBPlotOptions()
    plot_options.ConstantKeyReducerKeepOneKey = False
    plot_options.PlotAllTakes = False
    plot_options.PlotOnFrame = True
    plot_options.PlotPeriod = pyfbsdk.FBTime(0, 0, 0, 1)
    plot_options.PlotTranslationOnRootOnly = False
    plot_options.PreciseTimeDiscontinuities = False
    plot_options.RotationFilterToApply = (
        pyfbsdk.FBRotationFilter.kFBRotationFilterUnroll
    )
    plot_options.UseConstantKeyReducer = False

    l_character.PlotAnimation(
        FBCharacterPlotWhere.kFBCharacterPlotOnSkeleton,
        plot_options,
    )


def activate_character(character_name):
    """Set the character with name `character_name` as the active Character."""
    for l_char in FBSystem().Scene.Characters:
        if l_char.Name == character_name:
            l_char.InputCharacter = FBApplication().CurrentCharacter
            l_char.InputType = FBCharacterInputType.kFBCharacterInputCharacter
            l_char.ActiveInput = True
            FBApplication().CurrentCharacter = l_char


def select_descendants(skeleton):
    """Recursively select a skeleton joint and all its descendants."""
    skeleton.Selected = True
    for child in skeleton.Children:
        if isinstance(child, FBModelSkeleton):
            select_descendants(child)


def import_bvh(file_path):
    """Import a BVH file into the current scene."""
    return FBApplication().FileImport(file_path)


def export_bvh(file_path):
    """Export the current scene to a BVH file."""
    return FBApplication().FileExport(file_path)


def retarget(num):
    """Characterize, retarget, bake, and select the SMPL skeleton."""
    _ = characterize_character("Character", num)
    activate_character("Character")
    plot_character()

    all_skeletons = FBSystem().Scene.ModelSkeletons
    for skeleton in all_skeletons:
        if skeleton.Name == "Pelvis":
            select_descendants(skeleton)
            break


def main():
    if FOLDER_PATH is None or FBX_T_POSE_PATH is None:
        print("Script execution aborted due to path errors.")
        return

    folder_path = FOLDER_PATH
    fbx_file_path = FBX_T_POSE_PATH

    print(f"DEBUG: Checking directory contents: {folder_path}")
    try:
        all_files = os.listdir(folder_path)
        print(f"DEBUG: All files found: {all_files}")
    except Exception as e:
        print(f"DEBUG: Failed to list directory: {e}")
        return

    output_dir = os.path.join(folder_path, OUTPUT_DIR_NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        print(f"Using existing directory: {output_dir}")

    bvh_files = [f for f in all_files if f.lower().endswith(".bvh")]
    
    if not bvh_files:
        msg = f"No .bvh files found in:\n{folder_path}\n(See console for file list)"
        print(msg)
        pyfbsdk.FBMessageBox("Warning", msg, "OK")
        return

    print(f"Found {len(bvh_files)} BVH files to process: {bvh_files}")
    
    success_count = 0
    for file in bvh_files:
        try:
            app = FBApplication()
            # Force a new scene to avoid clutter
            app.FileNew()
            
            print(f"DEBUG: Opening FBX: {fbx_file_path}")
            # Open FBX T-pose (Character)
            status = app.FileOpen(fbx_file_path)
            if not status:
                print(f"Failed to open FBX: {fbx_file_path}")
                # Try creating a message box to alert user
                pyfbsdk.FBMessageBox("Error", f"Could not open FBX:\n{fbx_file_path}", "OK")
                return # Critical error, stop script

            print(f"Processing BVH: {file}")
            bvh_file_path = os.path.join(folder_path, file)
            
            # Import BVH
            print(f"DEBUG: Importing BVH: {bvh_file_path}")
            if app.FileImport(bvh_file_path):
                print("BVH file imported successfully.")
            else:
                print(f"Failed to import BVH: {bvh_file_path}")
                continue

            # Do the work
            retarget(1)

            out_bvh_path = os.path.join(
                output_dir, file[:-4] + "Re.bvh"
            )
            if export_bvh(out_bvh_path):
                print(f"Exported: {out_bvh_path}")
                success_count += 1
            else:
                print(f"Failed to export: {out_bvh_path}")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            import traceback
            traceback.print_exc()

    pyfbsdk.FBMessageBox("Process Completed", f"Successfully processed {success_count}/{len(bvh_files)} files.\nCheck output folder.", "OK")


# In MotionBuilder, scripts are often executed via `exec` and `__name__` is not
# set to '__main__', so we call main() unconditionally.
main()
