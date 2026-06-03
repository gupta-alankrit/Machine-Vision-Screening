#!/usr/bin/env python3
"""
STEP 02: Build a SAM-2 mask bank from isolated-component videos.

For each video file in --videos_dir named <component_id>_<reseat_idx>.<ext>
(e.g. screw_a_01.mp4, nut_b_02.bag, gear_a_03.mp4):

  1. Extract frames to a temp dir
  2. Show first frame; user clicks the component once (positive point prompt)
  3. Run SAM-2 video propagation -> one mask per frame
  4. Show 4 random sample frames with mask overlay; user accepts or redoes
  5. Save (frame RGB, binary mask) pairs to
     <out_dir>/<component_id>/<video_stem>/frame_NNNNNN.png + frame_NNNNNN_mask.png

Output layout:
    <out_dir>/
      manifest.json              # one row per accepted video (relative paths only)
      screw_a/
        screw_a_01/
          frame_000000.png       # original RGB frame
          frame_000000_mask.png  # binary mask (0 / 255)
          frame_000001.png
          frame_000001_mask.png
          ...
        screw_a_02/
          ...
      nut_b/
        ...

Pre-reqs:
  - The single project conda env (see README.md "Environment setup") plus
    SAM-2 installed into it from GitHub:
        pip install "git+https://github.com/facebookresearch/segment-anything-2.git"
  - SAM-2.1 checkpoint (`sam2.1_hiera_large.pt`) downloaded locally.
  - Display available (script uses cv2.imshow for click + review). For SSH,
    use `ssh -X` or run locally.

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):
  >> conda activate mvs
  >> python3 scripts/02_build_sam2_mask_bank.py \
       --videos_dir       data/workspace_B/individual_components/raw_videos \
       --out_dir          <choose a path outside the repo for the mask bank> \
       --sam2_checkpoint  <absolute or repo-relative path to sam2.1_hiera_large.pt> \
       --sam2_config      configs/sam2.1/sam2.1_hiera_l.yaml \
       --frame_stride     2
"""

import argparse
import json
import os
import random
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# Repo root resolved from this script's location (<repo>/scripts/<name>.py).
REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(p):
    """Expand ~ and resolve. Absolute paths are returned as-is (after
    expanduser). Relative paths are resolved against the repo root (NOT the
    current working directory), so the script works regardless of where it is
    invoked from."""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


try:
    import torch
except ImportError as e:
    raise SystemExit("PyTorch not installed in this env.") from e

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    raise SystemExit(
        "SAM-2 not installed. Install with:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything-2.git\n"
        "Then download a SAM-2.1 checkpoint from "
        "https://github.com/facebookresearch/segment-anything-2#download-checkpoints"
    ) from e


# ---------------------------------------------------------------------------
# Video -> JPEG frames (SAM-2 reads from a directory of JPEGs)
# ---------------------------------------------------------------------------

def extract_video_to_jpegs(video_path: Path, out_dir: Path, frame_stride: int) -> int:
    """Extract video frames to JPEGs named 00000.jpg, 00001.jpg, ... in out_dir.

    Returns the number of frames written. Frames are sampled every `frame_stride`.
    Handles both regular video files (.mp4/.mov/.avi/.mkv) and RealSense .bag files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if video_path.suffix.lower() == ".bag":
        return _extract_bag(video_path, out_dir, frame_stride)
    return _extract_cv2(video_path, out_dir, frame_stride)


def _extract_cv2(video_path: Path, out_dir: Path, frame_stride: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    n_saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_stride == 0:
            cv2.imwrite(str(out_dir / f"{n_saved:05d}.jpg"), frame)
            n_saved += 1
        frame_idx += 1
    cap.release()
    return n_saved


def _extract_bag(bag_path: Path, out_dir: Path, frame_stride: int) -> int:
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise SystemExit(
            f"pyrealsense2 not installed; can't extract .bag file {bag_path}.\n"
            "Install with: pip install pyrealsense2"
        ) from e

    pipeline = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag_path), repeat_playback=False)
    profile = pipeline.start(cfg)
    profile.get_device().as_playback().set_real_time(False)

    n_saved = 0
    frame_idx = 0
    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break  # end of bag
            color = frames.get_color_frame()
            if not color:
                continue
            img = np.asanyarray(color.get_data())
            # RealSense delivers RGB; cv2.imwrite wants BGR.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if frame_idx % frame_stride == 0:
                cv2.imwrite(str(out_dir / f"{n_saved:05d}.jpg"), img)
                n_saved += 1
            frame_idx += 1
    finally:
        pipeline.stop()
    return n_saved


# ---------------------------------------------------------------------------
# Resolution pre-check (cheap; run before SAM-2 to fail fast on bad inputs)
# ---------------------------------------------------------------------------

def get_video_resolution(video_path: Path):
    """Return (width, height) of the video, or None if it can't be opened.

    Uses cv2.VideoCapture which handles .MOV/.mp4/.avi/.mkv. No frame decoding —
    just reads the stream header.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h) if (w > 0 and h > 0) else None
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Interactive UI helpers
# ---------------------------------------------------------------------------

def click_on_first_frame(frame_path: Path):
    """Show the first frame and capture a single positive click on the component.

    Returns (x, y) of the latest click. Returns None if user pressed 'q' to skip.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Could not read frame: {frame_path}")

    win = "Click on the component, then Enter to confirm  (q = skip this video)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, img)

    clicks = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            disp = img.copy()
            cv2.circle(disp, (x, y), 8, (0, 0, 255), -1)
            cv2.circle(disp, (x, y), 14, (0, 255, 0), 2)
            cv2.imshow(win, disp)

    cv2.setMouseCallback(win, on_click)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            cv2.destroyWindow(win)
            return None
        if k in (13, 10):  # Enter / Return
            if clicks:
                cv2.destroyWindow(win)
                return clicks[-1]
            print("  [INFO] Click on the component first, then press Enter.")


def review_sample_masks(out_dir: Path, frame_indices, n_samples: int = 4):
    """Show n_samples random frame+mask overlays for visual QC.

    Returns:
        True  -> user accepted (Enter / any key while iterating)
        False -> user wants to redo (r)
        None  -> user wants to skip this video entirely (q)
    """
    sample_indices = random.sample(list(frame_indices), min(n_samples, len(frame_indices)))
    win = "Sample masks   (Enter = next/accept, r = redo prompt, q = skip video)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        for fi in sample_indices:
            img = cv2.imread(str(out_dir / f"frame_{fi:06d}.png"))
            mask = cv2.imread(str(out_dir / f"frame_{fi:06d}_mask.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = img.copy()
            overlay[mask > 0] = (0, 255, 0)
            out = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
            area_frac = float((mask > 0).sum()) / mask.size
            label = f"frame {fi}   mask area: {area_frac * 100:.2f}% of image"
            cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(win, out)
            k = cv2.waitKey(0) & 0xFF
            if k == ord("r"):
                return False
            if k == ord("q"):
                return None
        return True
    finally:
        cv2.destroyWindow(win)


def parse_component_id(video_stem: str):
    """'screw_a_03' -> ('screw_a', '03'). Falls back to (stem, '00') if no _NN suffix."""
    parts = video_stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return video_stem, "00"


# ---------------------------------------------------------------------------
# Per-video SAM-2 propagation
# ---------------------------------------------------------------------------

def run_sam2_on_video(predictor, jpeg_dir: Path, click_xy, device: str):
    """Click prompt at frame 0, propagate, return {frame_idx: binary_mask_uint8}."""
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(jpeg_dir))
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=np.array([click_xy], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),   # 1 = positive
        )
        masks = {}
        for out_frame_idx, _out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            # out_mask_logits shape: (n_objects, 1, H, W) — we only have one object.
            m = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            masks[out_frame_idx] = m
    return masks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True,
                    help="Dir of isolated-component videos. Filenames must follow "
                         "<component_id>_<reseat_idx>.<ext>, e.g. screw_a_01.mp4")
    ap.add_argument("--out_dir", required=True,
                    help="Output mask bank directory.")
    ap.add_argument("--sam2_checkpoint", required=True,
                    help="Absolute path to a SAM-2.1 .pt checkpoint.")
    ap.add_argument("--sam2_config", required=True,
                    help="SAM-2 config name (e.g. configs/sam2.1/sam2.1_hiera_l.yaml).")
    ap.add_argument("--frame_stride", type=int, default=8,
                    help="Save every Nth frame (1 = all, 2 = half, etc.). Default 8.")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                    help="Compute device.")
    ap.add_argument("--skip_review", action="store_true",
                    help="Auto-accept all masks (skip the visual QC step). NOT recommended.")
    ap.add_argument("--min_mask_area_frac", type=float, default=0.001,
                    help="Discard frames whose mask covers less than this fraction "
                         "of the image (default 0.1%% -- catches lost-tracking frames).")
    ap.add_argument("--min_width", type=int, default=1280,
                    help="Minimum required video width in pixels (default: 1280, "
                         "matches D455 deployment resolution). Videos below this are skipped.")
    ap.add_argument("--min_height", type=int, default=720,
                    help="Minimum required video height in pixels (default: 720, "
                         "matches D455 deployment resolution). Videos below this are skipped.")
    args = ap.parse_args()

    videos_dir = _resolve_path(args.videos_dir)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover videos -----------------------------------------------------
    exts = {".mp4", ".mov", ".avi", ".mkv", ".bag"}
    videos = sorted([p for p in videos_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in exts])
    if not videos:
        raise SystemExit(f"No videos found in {videos_dir}")

    print(f"[INFO] Found {len(videos)} videos in {videos_dir}:")
    for v in videos:
        print(f"  - {v.name}")

    # ---- Pre-flight: check each video meets the minimum resolution -----------
    # Higher recording resolution than deployment is the recommended setup
    # (cleaner masks, easier downsampling at copy-paste time). Skip videos
    # below the deployment resolution to avoid wasting SAM-2 inference on them.
    print(f"\n[INFO] Checking video resolutions (required: >= {args.min_width}x{args.min_height})")
    kept_videos = []
    for v in videos:
        res = get_video_resolution(v)
        if res is None:
            print(f"  [SKIP] {v.name}: could not read resolution (corrupt or unreadable)")
            continue
        w, h = res
        if w >= args.min_width and h >= args.min_height:
            print(f"  [OK]   {v.name}: {w}x{h}")
            kept_videos.append((v, (w, h)))
        else:
            print(f"  [SKIP] {v.name}: {w}x{h} below required {args.min_width}x{args.min_height}")

    if not kept_videos:
        raise SystemExit(
            f"No videos meet the minimum resolution {args.min_width}x{args.min_height}. "
            f"Re-record at >= that resolution, or lower --min_width / --min_height if intentional."
        )
    videos = [v for v, _res in kept_videos]
    video_resolutions = dict(kept_videos)   # for stashing into the manifest later
    print(f"[OK] {len(videos)} video(s) pass the resolution check.")

    # ---- Load SAM-2 ----------------------------------------------------------
    sam2_ckpt = _resolve_path(args.sam2_checkpoint)
    assert sam2_ckpt.exists(), f"Missing SAM-2 checkpoint: {sam2_ckpt}"
    print(f"\n[INFO] Loading SAM-2  config={args.sam2_config}  ckpt={sam2_ckpt}")
    predictor = build_sam2_video_predictor(args.sam2_config, str(sam2_ckpt), device=args.device)
    print("[OK] SAM-2 loaded")

    # ---- Manifest scaffolding -----------------------------------------------
    # All paths inside the manifest are RELATIVE to out_dir; the original
    # absolutes are saved once in `_absolute_at_write_time` for audit.
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_paths_note": "Paths under 'videos' are RELATIVE to this file's directory (out_dir).",
        "_absolute_at_write_time": {
            "out_dir":         str(out_dir),
            "videos_dir":      str(videos_dir),
            "sam2_checkpoint": str(sam2_ckpt),
        },
        "sam2_config":     args.sam2_config,
        "frame_stride":    args.frame_stride,
        "device":          args.device,
        "min_mask_area_frac": args.min_mask_area_frac,
        "videos": [],
    }

    # ---- Per-video loop ------------------------------------------------------
    for v in videos:
        component_id, reseat_idx = parse_component_id(v.stem)
        print(f"\n{'=' * 70}\nProcessing: {v.name}")
        print(f"  component_id={component_id}  reseat_idx={reseat_idx}\n{'=' * 70}")

        out_vid_dir = out_dir / component_id / v.stem
        if out_vid_dir.exists() and any(out_vid_dir.iterdir()):
            print(f"  [SKIP] {out_vid_dir} already exists and is non-empty.")
            print(f"         Delete it manually to re-process this video.")
            continue
        out_vid_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            jpeg_dir = Path(td) / "frames"
            print(f"  Extracting frames (stride={args.frame_stride}) -> tmp")
            n_extracted = extract_video_to_jpegs(v, jpeg_dir, args.frame_stride)
            print(f"  Extracted {n_extracted} frames")
            if n_extracted == 0:
                print("  [WARN] No frames extracted; skipping.")
                shutil.rmtree(out_vid_dir, ignore_errors=True)
                continue

            jpeg_paths = sorted(jpeg_dir.glob("*.jpg"))

            accepted = False
            attempts = 0
            last_click = None
            n_saved_final = 0

            while not accepted:
                attempts += 1

                click_xy = click_on_first_frame(jpeg_paths[0])
                if click_xy is None:
                    print("  [SKIP] User chose to skip this video.")
                    shutil.rmtree(out_vid_dir, ignore_errors=True)
                    break
                last_click = click_xy
                print(f"  Attempt {attempts}: click at ({click_xy[0]}, {click_xy[1]})")

                print("  Running SAM-2 propagation...")
                try:
                    masks = run_sam2_on_video(predictor, jpeg_dir, click_xy, args.device)
                except Exception as e:
                    print(f"  [ERROR] SAM-2 propagation failed: {e}")
                    print("  Try a different click position, or 'q' to skip this video.")
                    # Clear and retry
                    for f in out_vid_dir.iterdir():
                        f.unlink()
                    continue

                # ---- Save frames + masks --------------------------------------
                print(f"  Saving {len(masks)} frame+mask pairs to {out_vid_dir}")
                saved_indices = []
                n_dropped_small = 0
                img_area = None
                for fi in sorted(masks.keys()):
                    src_frame = jpeg_paths[fi]
                    img = cv2.imread(str(src_frame))
                    if img is None:
                        continue
                    mask = masks[fi]
                    if mask.shape != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                    img_area = mask.size
                    if (mask > 0).sum() / img_area < args.min_mask_area_frac:
                        n_dropped_small += 1
                        continue
                    cv2.imwrite(str(out_vid_dir / f"frame_{fi:06d}.png"), img)
                    cv2.imwrite(str(out_vid_dir / f"frame_{fi:06d}_mask.png"), mask)
                    saved_indices.append(fi)

                print(f"  Saved {len(saved_indices)} (dropped {n_dropped_small} "
                      f"frames below min_mask_area_frac={args.min_mask_area_frac})")

                if not saved_indices:
                    print("  [WARN] No frames passed min_mask_area filter — "
                          "the click probably missed the component.")
                    for f in out_vid_dir.iterdir():
                        f.unlink()
                    continue

                # ---- Visual QC -----------------------------------------------
                if args.skip_review:
                    accepted = True
                    n_saved_final = len(saved_indices)
                else:
                    print("\n  [Review] Showing 4 random sample masks...")
                    verdict = review_sample_masks(out_vid_dir, saved_indices, n_samples=4)
                    if verdict is None:
                        print("  [SKIP] User skipped this video after review.")
                        shutil.rmtree(out_vid_dir, ignore_errors=True)
                        break
                    if verdict:
                        accepted = True
                        n_saved_final = len(saved_indices)
                        print(f"  [OK] Accepted ({n_saved_final} masks).")
                    else:
                        print("  [REDO] User wants to re-prompt with a different click.")
                        for f in out_vid_dir.iterdir():
                            f.unlink()
                        # loop continues — back to click

            # ---- Record in manifest -------------------------------------------
            if accepted:
                manifest["videos"].append({
                    "video_filename":      v.name,                                   # relative to videos_dir
                    "component_id":        component_id,
                    "reseat_idx":          reseat_idx,
                    "output_dir":          str(out_vid_dir.relative_to(out_dir)),    # relative to out_dir
                    "video_resolution_wh": list(video_resolutions[v]),               # from pre-flight check
                    "n_frames_extracted":  n_extracted,
                    "n_masks_saved":       n_saved_final,
                    "first_frame_click":   list(last_click),
                    "attempts_to_accept":  attempts,
                })

    # ---- Write manifest ------------------------------------------------------
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[OK] Mask bank built at: {out_dir}")
    print(f"[OK] Manifest:           {out_dir / 'manifest.json'}")

    # Per-component summary
    print("\nPer-component summary:")
    by_comp = defaultdict(lambda: {"n_videos": 0, "n_masks": 0})
    for v in manifest["videos"]:
        by_comp[v["component_id"]]["n_videos"] += 1
        by_comp[v["component_id"]]["n_masks"]  += v["n_masks_saved"]
    if not by_comp:
        print("  (no videos accepted)")
    for comp, stats in sorted(by_comp.items()):
        print(f"  {comp:<14s}  videos={stats['n_videos']:<3d}  total_masks={stats['n_masks']}")


if __name__ == "__main__":
    main()
