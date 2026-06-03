# Machine-Vision-Screening

YOLOv8-seg pipeline for detecting screws, nuts, and gears on a
manufacturing platform. The current working model -- trained on
Workspace-B data -- is at `models/workspace_B_best.pt`. This repository
contains the scripts, raw input data, and trained weights required to
reproduce that model end-to-end.

## Repository layout

```
Machine-Vision-Screening/
├── LICENSE
├── README.md                       (this file)
├── requirements.txt                pip dependencies for the single conda env
├── scripts/                        all data-prep, augmentation, and training scripts
│   └── README.md                   pipeline order + per-script usage
├── data/
│   └── workspace_B/                raw inputs captured for the Workspace-B model
│       ├── one_component_image_empty_scene_pair/
│       │   ├── cvat_export_1/      16 static setups (CVAT-labeled, COCO 1.0)
│       │   └── cvat_export_2/      6  additional sideways-orientation setups
│       ├── hard_negatives/
│       │   ├── extracted_frames/   frames + hand-labels for each hard-negative video
│       │   │   ├── 20260526_221125/cvat_export/
│       │   │   └── 20260527_210106/cvat_export/
│       │   └── raw_videos/         video_1.bag + video_2.db3   (NOT in repo, see MANIFEST.md)
│       └── individual_components/
│           └── raw_videos/         40 Nikon turntable .MOV clips (NOT in repo, see MANIFEST.md)
└── models/
    ├── sim_baseline_best.pt        Isaac-Sim-trained checkpoint; starting point for fine-tuning
    └── workspace_B_best.pt         current working model (fine-tuned for Workspace B)
```

## Folder details

### `scripts/`

Self-contained pipeline scripts. The digit prefix on each filename
indicates its rough place in the pipeline. Run order, environment
prerequisites, and command examples are in
[`scripts/README.md`](scripts/README.md).

### `data/workspace_B/one_component_image_empty_scene_pair/`

22 paired RealSense captures of the workspace platform. Each "setup" is
two images: one with a single component placed on the platform and one
empty (the empty image acts as a hard negative for that setup's
background).

- `cvat_export_1/` -- the original 16 setups (7 screw variants, 7 nut
  variants, 2 gear variants).
- `cvat_export_2/` -- 6 additional setups capturing sideways
  orientations that the model previously mis-classified (2 each of
  screw, nut, gear).

Each `cvat_export_*/` is a CVAT COCO 1.0 export:
`annotations/instances_default.json` + `images/default/*.png`.
Categories: `screw, nut, gear, platform`. The `platform` polygons are
not training targets -- they define the allowed paste region for the
copy-paste augmentation step (script 05).

### `data/workspace_B/hard_negatives/`

Two RealSense recordings of the workspace platform with no components
placed on it, used as in-distribution hard negatives during training.

- `raw_videos/` -- `video_1.bag` and `video_2.db3` (21 GB total).
  **Not in repo.** Hashes, source-location info, and instructions to
  obtain are in `raw_videos/MANIFEST.md`.
- `extracted_frames/<stem>/cvat_export/` -- frames sampled from each
  video by script 01, then hand-labeled in CVAT and exported as
  COCO 1.0. These exports are sufficient to rebuild the training
  dataset without needing the raw videos.

### `data/workspace_B/individual_components/`

Source data for the SAM-2 mask bank used by the copy-paste augmentation
step (script 05).

- `raw_videos/` -- 40 Nikon D5300 turntable clips (~1.4 GB total). Each
  clip shows one component (`<id>_<take>.MOV`) rotating against a plain
  white wall so SAM-2 can extract clean per-frame instance masks.
  **Not in repo.** Camera specs, hashes, and instructions to obtain are
  in `raw_videos/MANIFEST.md`.

### `models/`

- `sim_baseline_best.pt` -- YOLOv8n-seg trained on Isaac-Sim synthetic
  images. Stage 1 of the fine-tune starts from this checkpoint.
- `workspace_B_best.pt` -- the fine-tuned Workspace-B model produced by
  `scripts/06_finetune_sim_model.ipynb` (stage 2). This is the current
  deployment model.

## Out-of-band data

The two `raw_videos/` folders are placeholders -- the actual video files
are too large for GitHub and live on the capture machine. To reproduce
the pipeline end-to-end (i.e. re-run scripts 01 and 02), read the
`MANIFEST.md` inside each `raw_videos/` folder; it lists the files,
their SHA256 hashes, the exact path the files must be placed at inside
the repo, and how to obtain them.

The raw videos are only required if you intend to re-run script 02
(mask bank) or script 01 (frame extraction). For any other workflow
(retraining from existing hand-labels, inference, etc.) the in-repo
`cvat_export/` folders and the existing model checkpoint are
sufficient.

## Environment setup

The whole pipeline runs from a single conda environment. The pip
packages it needs are listed in [`requirements.txt`](requirements.txt).

### 1. Create the env and install dependencies

For GPU training (recommended for stage 6 -- training on CPU is unusably
slow), install the CUDA-matched PyTorch wheel first, then everything
else:

```bash
conda create -n mvs python=3.10 -y
conda activate mvs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For a CPU-only setup (fine for stages 1-5 and inference; not for stage 6
training) just skip the `--index-url` line:

```bash
conda create -n mvs python=3.10 -y
conda activate mvs
pip install -r requirements.txt
```

You can name the env anything you like; `mvs` is just the name used
throughout the docs.

### 2. SAM-2 (only needed if you re-run script 02)

SAM-2 is not on PyPI. Install it from GitHub into the same env, then
download the checkpoint:

```bash
pip install "git+https://github.com/facebookresearch/segment-anything-2.git"
# Place the checkpoint anywhere on disk; pass its path to script 02 via
# --sam2_checkpoint. Example download location:
#   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 3. ROS 2 (only needed if you re-run script 03 on a `.db3` recording)

`rclpy` is not pip-installable. Install ROS 2 Jazzy at the system level
(see the official ROS 2 install docs), then source it before running
script 03 on a `.db3` input:

```bash
source /opt/ros/jazzy/setup.bash
python3 scripts/03_process_workspace_videos.py --bags <...>.db3 ...
```

`.bag` (ROS 1) inputs do NOT need ROS 2 -- they go through
`pyrealsense2` which is in `requirements.txt`.

## Quick start

- Inference only: load `models/workspace_B_best.pt` with Ultralytics:
  `from ultralytics import YOLO; m = YOLO("models/workspace_B_best.pt")`.
- Reproduce the training: follow [`scripts/README.md`](scripts/README.md).

## License

See [`LICENSE`](LICENSE).
