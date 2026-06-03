# Hard-negative raw videos -- MANIFEST (files not in repo)

These are RealSense D455 recordings of the **Workspace B** scene used as
hard-negative training material (frames extracted via `01_process_workspace_videos.py`).

The files themselves are too large for GitHub and are tracked **out of band**.

## Where to save after obtaining

After receiving the videos from the repo owner (see "How to obtain"
below), place them at this exact path inside your local clone of the
repo so the scripts in `scripts/` can find them without code changes:

```
<repo_root>/data/workspace_B/hard_negatives/raw_videos/
├── video_1.bag
└── video_2.db3
```

## Source location on capture machine

`/home/agupta3129/machine_vision/RealSenseRecordings/workspace_B/fine-tuning-images/hard_negatives/raw_videos/`

## Files

| Filename     | Size (bytes)   | Size (GB) | SHA256                                                              |
|--------------|----------------|-----------|---------------------------------------------------------------------|
| video_1.bag  | 9,601,510,992  | 9.0       | a439db447f1da85e200efa8a07bc6095ba2058c8fde20704ead6299b81ec7bf3    |
| video_2.db3  | 11,975,696,384 | 12.0      | 1b8b6fa3673897a3c783fe9cb472fb7bda4faf7f4769c2351df1963b938fbdae    |

## Format

- `.bag`  -- RealSense ROS1 bag (color stream from D455).
- `.db3`  -- RealSense ROS2 SQLite bag (color stream from D455).

## Notes

- The frames extracted from these videos (and their CVAT hand-labels) ARE in
  the repo under `data/workspace_B/hard_negatives/extracted_frames/<video_stem>/cvat_export/`.
- The CVAT hand-labels are sufficient to rebuild the dataset; the raw videos
  are only required if you want to **re-extract** frames at different sample
  rates / from different time windows.
- Categories present in these clips: mostly `platform`, with a small number
  of incidental `screw`/`nut` annotations:
    - `video_1`: 73 platform, 3 nut
    - `video_2`: 88 platform, 1 screw
- The existing `extracted_frames/` subfolders still use the original
  capture-timestamp stems — these map to the renamed videos as:
    - `extracted_frames/20260526_221125/` &harr; `video_1.bag`
    - `extracted_frames/20260527_210106/` &harr; `video_2.db3`

## How to obtain

Contact the repo owner -- these files live on the capture machine and are
shared via external storage (USB / NAS / cloud), not through git.
