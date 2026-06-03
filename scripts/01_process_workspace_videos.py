#!/usr/bin/env python3
"""
STEP 01: Extract frames from workspace recordings (.bag OR .db3) for hand-labeling in CVAT.

Auto-detects the file format by extension:
  - .bag  ->  ROS 1 bag (older realsense-viewer / rs-record output); uses pyrealsense2
  - .db3  ->  ROS 2 bag (newer realsense-viewer default); uses sqlite3 + rclpy

For each file:
  - Skip the last --skip_end_sec seconds (avoid end-of-bag artifacts).
  - Sample frames at --frame_stride_sec intervals.
  - Save every sampled frame as a PNG into <out_dir>/<file_stem>/need_cvat/ (one
    subfolder per recording, named after the raw video file), filename
    `<file_stem>_<idx>_t<sec>s.png`.

The user does ALL labeling (component polygons + platform polygon) in CVAT afterwards.

Pre-reqs:
  - The single project conda env (see README.md "Environment setup") provides
    pyrealsense2 for the `.bag` path.
  - For `.db3` inputs you additionally need ROS 2 sourced for rclpy +
    sensor_msgs (system-level install, not pip):
        source /opt/ros/jazzy/setup.bash
    Sourcing ROS 2 alongside the conda env works -- the conda Python is still
    used; ROS 2 just adds its modules to PYTHONPATH.

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):

For .bag file:
  >> conda activate mvs
  >> python3 scripts/01_process_workspace_videos.py \
       --bags    data/workspace_B/hard_negatives/raw_videos/video_1.bag \
       --out_dir data/workspace_B/hard_negatives/extracted_frames

For .db3 file (still inside the same conda env, just source ROS 2 too):
  >> conda activate mvs
  >> source /opt/ros/jazzy/setup.bash
  >> python3 scripts/01_process_workspace_videos.py \
       --bags    data/workspace_B/hard_negatives/raw_videos/video_2.db3 \
       --out_dir data/workspace_B/hard_negatives/extracted_frames

Output layout:
  <out_dir>/
    <stem_A>/
      need_cvat/
        <stem_A>_0000_t0.00s.png
        <stem_A>_0001_t2.00s.png
        ...
    <stem_B>/
      need_cvat/
        <stem_B>_0000_t0.00s.png
        ...
"""

import argparse
import sqlite3
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


# =============================================================================
# .bag handling (ROS 1, via pyrealsense2)
# =============================================================================

def extract_bag(
    bag_path: Path,
    need_cvat_dir: Path,
    frame_stride_sec: float,
    skip_end_sec: float,
    start_skip_sec: float,
):
    """Extract color frames from a ROS 1 .bag via pyrealsense2."""
    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise SystemExit(
            "pyrealsense2 required to read .bag files.\n"
            "  Install:  pip install pyrealsense2"
        ) from e

    pipeline = rs.pipeline()
    cfg = rs.config()
    rs.config.enable_device_from_file(cfg, str(bag_path), repeat_playback=False)
    profile = pipeline.start(cfg)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    total_dur_sec = float(playback.get_duration().total_seconds())
    sample_end_sec = max(0.0, total_dur_sec - skip_end_sec)
    print(f"  duration: {total_dur_sec:.2f}s  "
          f"sample range: [{start_skip_sec:.2f}, {sample_end_sec:.2f}]s  "
          f"(stride {frame_stride_sec}s)")
    if sample_end_sec <= start_skip_sec:
        pipeline.stop()
        raise SystemExit(
            f"Bag too short: duration={total_dur_sec:.2f}s, "
            f"start_skip={start_skip_sec}s, skip_end={skip_end_sec}s leaves no frames."
        )

    next_sample_target = start_skip_sec
    saved_idx = 0
    first_frame_ts_ms = None
    manifest = []

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                break   # end of bag

            color = frames.get_color_frame()
            if not color:
                continue

            ts_ms = float(color.get_timestamp())
            if first_frame_ts_ms is None:
                first_frame_ts_ms = ts_ms
            t_sec = (ts_ms - first_frame_ts_ms) / 1000.0

            if t_sec >= sample_end_sec:
                continue
            if t_sec < start_skip_sec:
                continue
            if t_sec < next_sample_target:
                continue

            img_rgb = np.asanyarray(color.get_data())
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            fname = f"{bag_path.stem}_{saved_idx:04d}_t{t_sec:.2f}s.png"
            cv2.imwrite(str(need_cvat_dir / fname), img_bgr)
            manifest.append({
                "source":   bag_path.name,
                "filename": fname,
                "time_sec": round(t_sec, 4),
                "idx":      saved_idx,
            })
            saved_idx += 1
            next_sample_target += frame_stride_sec
    finally:
        pipeline.stop()

    return {
        "source_name":        bag_path.name,
        "format":             "bag",
        "total_duration_sec": total_dur_sec,
        "n_frames_saved":     saved_idx,
        "manifest":           manifest,
    }


# =============================================================================
# .db3 handling (ROS 2, via sqlite3 + rclpy deserialization)
# =============================================================================

def _ros_image_to_cv2(msg):
    """Convert a sensor_msgs/msg/Image to OpenCV BGR (no cv_bridge dependency)."""
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "rgb8":
        return cv2.cvtColor(data.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
    if enc == "bgr8":
        return data.reshape((h, w, 3))
    if enc == "rgba8":
        return cv2.cvtColor(data.reshape((h, w, 4)), cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(data.reshape((h, w, 4)), cv2.COLOR_BGRA2BGR)
    if enc == "mono8":
        return data.reshape((h, w))
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def _find_color_image_topic(cursor):
    """Pick the color-image topic from a realsense-viewer .db3 rosbag2 SQLite DB."""
    cursor.execute("SELECT id, name, type FROM topics;")
    topics = cursor.fetchall()

    # Prefer the realsense-viewer-default RGB topic
    for topic_id, name, typ in topics:
        if typ == "sensor_msgs/msg/Image" and "Color_0/image/data" in name:
            return topic_id, name
    # Fallback: any image topic with 'color' in the name
    for topic_id, name, typ in topics:
        if typ == "sensor_msgs/msg/Image" and "color" in name.lower() and "image" in name.lower():
            return topic_id, name
    # Final fallback: first image topic
    for topic_id, name, typ in topics:
        if typ == "sensor_msgs/msg/Image":
            return topic_id, name
    raise RuntimeError("No sensor_msgs/msg/Image topic found in this .db3 file.")


def extract_db3(
    db3_path: Path,
    need_cvat_dir: Path,
    frame_stride_sec: float,
    skip_end_sec: float,
    start_skip_sec: float,
):
    """Extract color frames from a ROS 2 .db3 via direct SQLite + rclpy deserialization."""
    try:
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
    except ImportError as e:
        raise SystemExit(
            "rclpy + sensor_msgs not importable -- required for .db3 files.\n"
            "  Source ROS 2 first:  source /opt/ros/jazzy/setup.bash"
        ) from e

    conn = sqlite3.connect(str(db3_path))
    cur = conn.cursor()

    topic_id, topic_name = _find_color_image_topic(cur)
    print(f"  topic: {topic_name}")

    cur.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id = ?;",
        (topic_id,),
    )
    first_ts, last_ts = cur.fetchone()
    if first_ts is None:
        conn.close()
        raise SystemExit(f"No messages on topic {topic_name} in {db3_path.name}")

    total_dur_sec = (last_ts - first_ts) / 1e9
    sample_end_sec = max(0.0, total_dur_sec - skip_end_sec)
    print(f"  duration: {total_dur_sec:.2f}s  "
          f"sample range: [{start_skip_sec:.2f}, {sample_end_sec:.2f}]s  "
          f"(stride {frame_stride_sec}s)")
    if sample_end_sec <= start_skip_sec:
        conn.close()
        raise SystemExit(
            f".db3 too short: duration={total_dur_sec:.2f}s, "
            f"start_skip={start_skip_sec}s, skip_end={skip_end_sec}s leaves no frames."
        )

    sample_end_ts   = first_ts + int(sample_end_sec   * 1e9)
    start_skip_ts   = first_ts + int(start_skip_sec   * 1e9)
    stride_ns       = int(frame_stride_sec * 1e9)
    next_sample_ts  = start_skip_ts

    saved_idx = 0
    manifest = []

    # Stream rows -- don't fetchall() (could be GBs).
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC;",
        (topic_id,),
    )
    for ts, blob in cur:
        if ts < next_sample_ts:
            continue
        if ts >= sample_end_ts:
            break
        try:
            msg = deserialize_message(blob, Image)
            img_bgr = _ros_image_to_cv2(msg)
        except Exception as e:
            print(f"  [WARN] decode failed at t={(ts - first_ts)/1e9:.2f}s: {e}")
            continue

        t_sec = (ts - first_ts) / 1e9
        fname = f"{db3_path.stem}_{saved_idx:04d}_t{t_sec:.2f}s.png"
        ok = cv2.imwrite(str(need_cvat_dir / fname), img_bgr)
        if not ok:
            print(f"  [WARN] cv2.imwrite failed: {fname}")
            continue
        manifest.append({
            "source":   db3_path.name,
            "filename": fname,
            "time_sec": round(t_sec, 4),
            "idx":      saved_idx,
        })
        saved_idx += 1
        next_sample_ts += stride_ns

    conn.close()
    return {
        "source_name":        db3_path.name,
        "format":             "db3",
        "total_duration_sec": total_dur_sec,
        "n_frames_saved":     saved_idx,
        "manifest":           manifest,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True,
                    help="One or more .bag or .db3 recordings. Format is auto-detected.")
    ap.add_argument("--out_dir", required=True,
                    help="Output dir. Frames go to <out_dir>/<recording_stem>/need_cvat/ "
                         "(one subfolder per recording, named after the raw video file).")
    ap.add_argument("--frame_stride_sec", type=float, default=2.0,
                    help="Sampling interval in seconds. Default 2.0 (~0.5 fps).")
    ap.add_argument("--skip_end_sec", type=float, default=2.0,
                    help="Skip the last N seconds of each recording. Default 2.0.")
    ap.add_argument("--start_skip_sec", type=float, default=0.0,
                    help="Optionally skip the first N seconds. Default 0.0.")
    args = ap.parse_args()

    inputs = [_resolve_path(b) for b in args.bags]
    for p in inputs:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")
        if p.suffix.lower() not in (".bag", ".db3"):
            raise SystemExit(f"Unsupported extension '{p.suffix}' on {p.name}. "
                             f"Expected .bag or .db3.")

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for p in inputs:
        ext = p.suffix.lower()
        # Per-recording output: <out_dir>/<video_stem>/need_cvat/
        need_cvat_dir = out_dir / p.stem / "need_cvat"
        need_cvat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 70}\nProcessing: {p.name}  [{ext[1:]}]  ->  {p.stem}/need_cvat/\n{'=' * 70}")
        if ext == ".bag":
            stats = extract_bag(p, need_cvat_dir,
                                args.frame_stride_sec, args.skip_end_sec, args.start_skip_sec)
        else:    # .db3
            stats = extract_db3(p, need_cvat_dir,
                                args.frame_stride_sec, args.skip_end_sec, args.start_skip_sec)
        print(f"  saved {stats['n_frames_saved']} frames -> {need_cvat_dir}")
        total += stats["n_frames_saved"]

    print(f"\n[OK] Total {total} frames written under {out_dir}")
    print(f"     (one subfolder per recording, named after the raw video file)")
    print(f"\nNext: upload each recording's subfolder to CVAT and hand-label each frame")
    print(f"      (screw / nut / gear polygons where applicable + a platform polygon).")


if __name__ == "__main__":
    main()
