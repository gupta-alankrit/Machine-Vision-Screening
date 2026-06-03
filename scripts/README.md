# Scripts pipeline -- how to reproduce `models/workspace_B_best.pt`

This document describes the order in which the scripts in this folder
were used to produce the current Workspace-B model, what each script
consumes and produces, and the command-line invocation used to run it.

For the top-level repository overview and folder descriptions, see
[`../README.md`](../README.md).

## Environment

The whole pipeline runs from a single conda environment. Create it once
from the repo root using [`../requirements.txt`](../requirements.txt) --
see the **Environment setup** section of the [top-level
README](../README.md) for the exact commands. The env name `mvs` is used
throughout the examples below; substitute your own if you named it
differently.

Two pieces of setup live outside `requirements.txt` because they aren't
pip-installable:

- **SAM-2** (script 02) -- installed from GitHub into the same env.
- **ROS 2 Jazzy** (script 03 only when the input is a `.db3` file) --
  system-level install, sourced at run time via
  `source /opt/ros/jazzy/setup.bash`.

Both are covered in the top-level README. `.bag` (ROS 1) inputs to
script 03 do NOT need ROS 2 -- they go through `pyrealsense2` which is
in `requirements.txt`.

## Pipeline overview

Scripts 01-03 are independent data-preparation stages and can be run in
any order or in parallel -- they produce three separate intermediate
artifacts that all feed into stage 4. The numbering reflects how often
you'll need to re-run them when adding new components: script 01
processes one-time workspace recordings, while scripts 02 and 03 are
per-component setup-time tasks that get re-run as the component set
grows.

| Stage | Script | Purpose                                                                |
|-------|--------|------------------------------------------------------------------------|
| 1     | 01     | Extract frames from the hard-negative bag / db3 recordings for CVAT    |
| 1a    | --     | (Manual) hand-label the extracted frames in CVAT                        |
| 2     | 02     | Build the SAM-2 mask bank from the Nikon turntable clips               |
| 3     | 03     | Build the static YOLO dataset + merged CVAT export from the 22 setups  |
| 4     | 04     | Merge stage-1 + stage-3 CVAT exports into a train/val/test split       |
| 5     | 05     | Copy-paste-augment the train split using the mask bank                 |
| 6     | 06     | Fine-tune YOLOv8-seg from `sim_baseline_best.pt`                        |

Scripts 07, 08, and 09 are auxiliary -- they are not part of the
Workspace-B training flow. See "Auxiliary scripts" at the bottom of
this document.

---

## Stage 1 -- `01_process_workspace_videos.py`

Extract PNG frames from each hard-negative recording at a fixed time
stride (default 2 s) so they can be hand-labeled in CVAT. The script
auto-detects the format by extension: `.bag` -> ROS 1 + pyrealsense2,
`.db3` -> ROS 2 + rclpy.

This is the one-time workspace-recording step. Once the platform is
captured under its real deployment lighting, you don't typically re-run
this when you add new component variants -- only when the workspace
itself changes (camera move, new lighting, new background).

| Direction | Path |
|-----------|------|
| In        | `data/workspace_B/hard_negatives/raw_videos/{video_1.bag, video_2.db3}` (obtain out of band, see manifest) |
| Out       | `<out_dir>/<stem>/need_cvat/<stem>_NNNN_t<sec>s.png` |

```bash
conda activate mvs

# For the .bag input (pyrealsense2 path):
python3 scripts/01_process_workspace_videos.py \
    --bags    data/workspace_B/hard_negatives/raw_videos/video_1.bag \
    --out_dir data/workspace_B/hard_negatives/extracted_frames

# For the .db3 input -- source ROS 2 first (system-level install required):
source /opt/ros/jazzy/setup.bash
python3 scripts/01_process_workspace_videos.py \
    --bags    data/workspace_B/hard_negatives/raw_videos/video_2.db3 \
    --out_dir data/workspace_B/hard_negatives/extracted_frames
```

## Stage 1a (manual) -- hand-label in CVAT

Upload the contents of `<out_dir>/<stem>/need_cvat/` to CVAT, label each
`screw / nut / gear` (rare in these clips) and `platform` (drawn on
every frame), then export as **COCO 1.0** and place the unzipped export
at:

```
data/workspace_B/hard_negatives/extracted_frames/<stem>/cvat_export/
```

The repo already ships these `cvat_export/` folders for both videos --
if you are not re-extracting frames, this step is already done.

---

## Stage 2 -- `02_build_sam2_mask_bank.py`

Build a SAM-2 mask bank by interactively tracking each isolated
component through its turntable clip. Each accepted clip produces a
sequence of `(RGB frame, binary mask)` pairs that the copy-paste
augmentation step uses as a paste source.

Re-run this stage when adding new component variants -- each new
variant needs at least one turntable clip and one pass through SAM-2 to
extend the mask bank.

| Direction | Path |
|-----------|------|
| In        | `data/workspace_B/individual_components/raw_videos/<component>_<take>.MOV` (40 files; obtain out of band, see manifest) |
| Out       | `<mask_bank_dir>/<component>/<clip_stem>/frame_NNNNNN.png` + `frame_NNNNNN_mask.png` + `manifest.json` |

Interactive: the script shows the first frame; click once on the
component to give SAM-2 a positive point prompt, then accept or redo
four sample masks. Repeat for each clip.

```bash
conda activate mvs
python3 scripts/02_build_sam2_mask_bank.py \
    --videos_dir       data/workspace_B/individual_components/raw_videos \
    --out_dir          <choose a path outside the repo for the mask bank> \
    --sam2_checkpoint  <path to sam2.1_hiera_large.pt> \
    --sam2_config      configs/sam2.1/sam2.1_hiera_l.yaml \
    --frame_stride     2
```

---

## Stage 3 -- `03_static_dataset_from_cvat.py`

Combine the two CVAT exports of the 22 static setups into

1. a YOLO-seg dataset at `<out_dir>/` with a train / val / test split
   assigned by setup id (a setup's `_Color` and `_empty` images stay
   together; no setup is split across two splits), and
2. a merged CVAT export at `<out_dir>/../cvat_export_combined/`, used as
   input by stage 4.

Per-export split counts in the example below: 10/3/3 from `cvat_export_1`
and 4/1/1 from `cvat_export_2`.

Re-run this stage when adding new (with, empty) setup pairs for new
component variants.

| Direction | Path |
|-----------|------|
| In        | `data/workspace_B/one_component_image_empty_scene_pair/cvat_export_1/` + `cvat_export_2/` |
| Out       | `<out_dir>/{images,labels,paste_regions}/{train,val,test}/...` + `<out_dir>/../cvat_export_combined/` |

```bash
python3 scripts/03_static_dataset_from_cvat.py \
    --cvat_export_dirs \
        data/workspace_B/one_component_image_empty_scene_pair/cvat_export_1 \
        data/workspace_B/one_component_image_empty_scene_pair/cvat_export_2 \
    --val_setups  3 1 \
    --test_setups 3 1 \
    --out_dir <pick a path for the static YOLO dataset>
```

---

## Stage 4 -- `04_build_split_dataset.py`

Merge

- the combined CVAT export from stage 3,
- the per-video CVAT exports from stage 1a, and
- the train/val/test setup assignments from stage 3

into a single train/val/test YOLO-seg dataset. Static-setup splits are
preserved exactly from stage 3; hard-negative video frames are shuffled
into a 70/15/15 split within each video.

| Direction | Path |
|-----------|------|
| In        | `<stage 3 out>/../cvat_export_combined/`, `<stage 3 out>/`, `data/workspace_B/hard_negatives/extracted_frames/<stem>/cvat_export/` (both stems) |
| Out       | `<out_dir>/{images,labels,paste_regions}/{train,val,test}/...` + `dataset.yaml` + `source_index.json` |

```bash
conda activate mvs
python3 scripts/04_build_split_dataset.py \
    --static_cvat_dir    <stage 3 out>/../cvat_export_combined \
    --static_dataset_dir <stage 3 out> \
    --video_cvat_dirs \
        data/workspace_B/hard_negatives/extracted_frames/20260526_221125/cvat_export \
        data/workspace_B/hard_negatives/extracted_frames/20260527_210106/cvat_export \
    --out_dir <pick a path for split_recorded>
```

---

## Stage 5 -- `05_copy_paste_augment.py`

Paste SAM-2-masked component crops from the stage-2 mask bank onto the
train images of stage 4. Per-image binary `paste_regions/` masks (built
in stages 3 and 4 from the CVAT platform polygons) constrain pastes to
the labeled platform area. Adds 2D rotation, perspective warp, color
match (Reinhard, Lab space), and noise match (Laplacian) for realism.
Val and test images are copied through unchanged -- no augmentation.

| Direction | Path |
|-----------|------|
| In        | `<stage 2 out>` + `<stage 4 out>` |
| Out       | `<out_dataset>/{images,labels}/{train,val,test}/...` + `dataset.yaml` + `augmentation_log.json` |

```bash
python3 scripts/05_copy_paste_augment.py \
    --mask_bank      <stage 2 out> \
    --source_dataset <stage 4 out> \
    --out_dataset    <pick a path for split_augmented> \
    --augs_per_image 15 \
    --paste_count_min 1 \
    --paste_count_max 6
```

---

## Stage 6 -- `06_finetune_sim_model.ipynb`

Two-stage fine-tune of `models/sim_baseline_best.pt`:

- **Stage 1**: freeze the backbone (`freeze=10`), train head only. Goal:
  adapt the segmentation head to real-image statistics without
  disturbing low-level features.
- **Stage 2**: unfreeze all layers, train at a lower LR. Goal: refine
  the whole network on the real domain.

Augmentation is applied online during training in two layers:
1. YOLO native args (`hsv_h, hsv_s, hsv_v, degrees, translate, scale, fliplr, perspective`).
2. An Albumentations monkey-patch that replaces the default pipeline.

| Direction | Path |
|-----------|------|
| In        | `<stage 5 out>/dataset.yaml`, `models/sim_baseline_best.pt` |
| Out       | A training run under the configured `project=` path. The final model is `<run>/real_ft_stage2/weights/best.pt` -- the same file shipped here as `models/workspace_B_best.pt`. |

Edit the path constants in section 1 of the notebook to point at your
local stage-5 dataset and `models/sim_baseline_best.pt`, then run the
notebook top to bottom.

---

## Auxiliary scripts (not in the Workspace-B training flow)

These three scripts are present in the repo but are **not** required to
reproduce `workspace_B_best.pt`. They are kept here for completeness
because script 09 imports 07 and 08 via `importlib`, and you may want
to run 09 to build a held-out eval dataset for a different workspace.

- **`07_filter_and_split.py`** -- filter a CVAT COCO export and assign
  train / val / test by image. Older real-image flow.
- **`08_augment_and_yolo_seg.py`** -- materialize a split COCO into the
  YOLO-seg directory layout. Older real-image flow.
- **`09_prepare_workspace_B_dataset.py`** -- build a test-only YOLO-seg
  eval dataset from any single CVAT export. Useful for evaluating an
  existing model on a fresh held-out workspace without going through
  stages 4 and 5. Imports 07 + 08 via `importlib`.
