#!/usr/bin/env python3
"""
STEP 04: Build the merged train/val/test dataset at <out_dir>/ from:
  - the static CVAT export (16 (with, empty) pairs, 4 labels: screw/nut/gear/platform)
  - one or more video CVAT exports (mostly hard negatives + a few component frames)
  - the existing static_dataset/ — used ONLY to preserve the train/val/test
    assignments that script 03 chose for the static images.

For every image:
  - Components (screw/nut/gear) -> YOLO-seg label file. Empty file if no component.
  - Platform polygons -> binary mask PNG under paste_regions/.
  - Image is copied into images/<split>/.

Source-group tagging (written to source_index.json) lets the augmentation step
(script 05) pick the right per-source color/noise reference for each background:
  - static_setup_<N>   :  uses <N>_Color.png's component(s) as the reference
  - video_<bag_stem>   :  uses that video's component-bearing frames as the reference

Split policy:
  Static  ->  preserved from static_dataset_dir/{train,val,test}/ (script 03).
  Video   ->  random 70/15/15 (configurable) per video, stratified.

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):
  >> conda activate mvs
  >> python3 scripts/04_build_split_dataset.py \
       --static_cvat_dir    <stage 3 out>/../cvat_export_combined \
       --static_dataset_dir <stage 3 out> \
       --video_cvat_dirs \
           data/workspace_B/hard_negatives/extracted_frames/20260526_221125/cvat_export \
           data/workspace_B/hard_negatives/extracted_frames/20260527_210106/cvat_export \
       --out_dir <choose a path for split_recorded>

Output layout:
  <out_dir>/
    images/{train,val,test}/<stem>.png
    labels/{train,val,test}/<stem>.txt        # YOLO-seg, screw/nut/gear only
    paste_regions/<stem>.png                  # union of platform polygons, per image (binary 0/255)
    dataset.yaml                              # relative paths; 3 component classes only
    source_index.json                         # stem -> source_group + per-group component-stem list
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import shutil


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


# Canonical YOLO class order — must match training scripts elsewhere in the project.
YOLO_CLASS_NAMES = ["screw", "nut", "gear"]
PLATFORM_LABEL = "platform"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_source_image(export_dir: Path, file_name: str):
    """Locate an image file inside a CVAT export, regardless of subset folder."""
    candidates = [
        export_dir / "images" / file_name,
        export_dir / "images" / "default" / file_name,
        export_dir / "images" / "Test" / file_name,
        export_dir / file_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = list(export_dir.rglob(file_name))
    return hits[0] if hits else None


def static_setup_id(file_name: str) -> str:
    """'3_Color.png' -> '3' (setup id, shared by `<N>_Color` and `<N>_empty`)."""
    m = re.match(r"^(\d+)_", Path(file_name).stem)
    return m.group(1) if m else Path(file_name).stem


def read_static_split_assignments(static_dataset_dir: Path) -> dict:
    """Return {setup_id: 'train' | 'val' | 'test'} from the existing
    static_dataset/images/{train,val,test}/. The 'test' subdir is optional
    (older static_dataset/ builds didn't have it)."""
    assignments = {}
    for split in ("train", "val", "test"):
        split_dir = static_dataset_dir / "images" / split
        if not split_dir.exists():
            continue
        for img_path in split_dir.iterdir():
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                assignments[static_setup_id(img_path.name)] = split
    return assignments


def load_coco(export_dir: Path):
    """Load the COCO JSON inside a CVAT export."""
    json_paths = sorted((export_dir / "annotations").glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No annotation JSON in {export_dir / 'annotations'}")
    return json_paths[0], json.loads(json_paths[0].read_text())


def coco_category_map(coco, required=YOLO_CLASS_NAMES + [PLATFORM_LABEL]):
    """Return (yolo_id_for_cat_id, platform_cat_id) given a COCO categories list.
    Validates that all required labels are present."""
    name_to_cat = {c["name"]: c["id"] for c in coco["categories"]}
    missing = [n for n in required if n not in name_to_cat]
    if missing:
        raise SystemExit(
            f"Missing labels in CVAT export: {missing}. Found: {list(name_to_cat.keys())}"
        )
    yolo_id_for_cat_id = {name_to_cat[n]: i for i, n in enumerate(YOLO_CLASS_NAMES)}
    platform_cat_id   = name_to_cat[PLATFORM_LABEL]
    return yolo_id_for_cat_id, platform_cat_id


def write_yolo_label(out_path: Path, ann_list, yolo_id_for_cat_id, platform_cat_id, W, H):
    """Write the YOLO-seg .txt for one image. Returns (yolo_lines, platform_polys_px)."""
    yolo_lines = []
    platform_polys_px = []
    for ann in ann_list:
        seg = ann.get("segmentation", [])
        if not seg or not isinstance(seg, list) or not seg[0]:
            continue
        poly_flat = seg[0]
        cat_id = ann["category_id"]
        if cat_id == platform_cat_id:
            pts = np.array(
                [[poly_flat[i], poly_flat[i + 1]] for i in range(0, len(poly_flat), 2)],
                dtype=np.int32,
            )
            if len(pts) >= 3:
                platform_polys_px.append(pts)
        elif cat_id in yolo_id_for_cat_id:
            yolo_id = yolo_id_for_cat_id[cat_id]
            norm = []
            for i in range(0, len(poly_flat), 2):
                norm.append(poly_flat[i]     / W)
                norm.append(poly_flat[i + 1] / H)
            yolo_lines.append(f"{yolo_id} " + " ".join(f"{v:.6f}" for v in norm))
    out_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
    return yolo_lines, platform_polys_px


# ---------------------------------------------------------------------------
# Per-export processing (writes images/labels/paste_regions; updates trackers)
# ---------------------------------------------------------------------------

def process_export(
    export_dir: Path,
    out_dir: Path,
    split_for_image,                         # callable: file_name -> 'train'/'val'/'test'
    source_group_for_image,                  # callable: file_name -> source_group string
    stem_to_source: dict,                    # output: populated
    source_to_component_stems: defaultdict,  # output: populated
    counts: dict,                            # output: populated {split: {"pos": n, "neg": n}}
):
    """Process all images in a CVAT export. Writes to out_dir/{images,labels,paste_regions}/."""
    coco_path, coco = load_coco(export_dir)
    print(f"  [INFO] reading {coco_path.relative_to(export_dir.parent)} "
          f"({len(coco['images'])} images, {len(coco['annotations'])} anns)")

    yolo_id_for_cat_id, platform_cat_id = coco_category_map(coco)

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    n_missing_src = 0
    for img in coco["images"]:
        file_name = img["file_name"]
        W, H = int(img["width"]), int(img["height"])
        stem = Path(file_name).stem
        split = split_for_image(file_name)
        source_group = source_group_for_image(file_name)
        stem_to_source[stem] = source_group

        src_path = find_source_image(export_dir, file_name)
        if src_path is None:
            print(f"    [WARN] source image not found: {file_name}")
            n_missing_src += 1
            continue

        # Copy image
        shutil.copy2(src_path, out_dir / "images" / split / file_name)

        # Write YOLO label + collect platform polygons
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"
        yolo_lines, platform_polys_px = write_yolo_label(
            lbl_path, anns_by_img.get(img["id"], []),
            yolo_id_for_cat_id, platform_cat_id, W, H,
        )

        if yolo_lines:
            counts[split]["pos"] += 1
            source_to_component_stems[source_group].append(stem)
        else:
            counts[split]["neg"] += 1

        if platform_polys_px:
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, platform_polys_px, 255)
            cv2.imwrite(str(out_dir / "paste_regions" / f"{stem}.png"), mask)

    if n_missing_src:
        print(f"    [WARN] {n_missing_src} image(s) referenced in COCO but not found")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_cvat_dir", required=True, type=Path,
                    help="A single CVAT COCO export folder containing ALL static (with, empty) "
                         "image annotations. If you labeled across multiple CVAT tasks, use "
                         "script 03's --combined_export_out to produce a merged export and "
                         "point this at that folder.")
    ap.add_argument("--static_dataset_dir", required=True, type=Path,
                    help="Existing static_dataset/ (output of script 03); read for train/val split.")
    ap.add_argument("--video_cvat_dirs", nargs="+", required=True, type=Path,
                    help="One or more video CVAT export folders. The source-group name for "
                         "each video is derived from the parent dir name (e.g., "
                         "<video_stem>/cvat_export -> source_group = 'video_<video_stem>').")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Output dataset dir (created). Final layout: images/{train,val,test}/, "
                         "labels/..., paste_regions/, dataset.yaml, source_index.json.")
    ap.add_argument("--val_ratio",  type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    static_cvat = _resolve_path(args.static_cvat_dir)
    static_ds   = _resolve_path(args.static_dataset_dir)
    video_cvats = [_resolve_path(p) for p in args.video_cvat_dirs]
    out_dir     = _resolve_path(args.out_dir)
    val_ratio   = args.val_ratio
    test_ratio  = args.test_ratio
    train_ratio = 1.0 - val_ratio - test_ratio
    if not (0.0 < train_ratio < 1.0):
        raise SystemExit(f"--val_ratio + --test_ratio must be in (0, 1). Got val={val_ratio}, test={test_ratio}.")

    print(f"[INFO] Output dataset: {out_dir}")
    print(f"[INFO] Static CVAT:    {static_cvat}")
    print(f"[INFO] Static dataset: {static_ds}  (used only for preserving train/val/test assignments)")
    print(f"[INFO] Video CVATs:    {len(video_cvats)}")
    for v in video_cvats:
        print(f"         - {v}")
    print(f"[INFO] Split: train={train_ratio:.2f}  val={val_ratio:.2f}  test={test_ratio:.2f}  "
          f"seed={args.seed}")

    # ---- Setup output dirs --------------------------------------------------
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out_dir {out_dir} already exists and is non-empty. "
                         f"Delete it first to rebuild from scratch.")
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "paste_regions").mkdir(parents=True, exist_ok=True)

    # ---- Trackers (populated by process_export) ----------------------------
    stem_to_source           = {}
    source_to_component_stems = defaultdict(list)
    counts                   = {s: {"pos": 0, "neg": 0} for s in ("train", "val", "test")}

    # ---- Static: preserve existing train/val from static_dataset/ ----------
    print(f"\n{'=' * 70}\nStatic export (preserve existing train/val from {static_ds.name})\n{'=' * 70}")
    static_split_by_setup = read_static_split_assignments(static_ds)
    if not static_split_by_setup:
        raise SystemExit(f"No images found under {static_ds / 'images'}. "
                         f"Run script 03 first to build static_dataset/.")
    n_static_train = sum(1 for v in static_split_by_setup.values() if v == 'train')
    n_static_val   = sum(1 for v in static_split_by_setup.values() if v == 'val')
    n_static_test  = sum(1 for v in static_split_by_setup.values() if v == 'test')
    print(f"  [INFO] Static split lookup: "
          f"{n_static_train} train, {n_static_val} val, {n_static_test} test setups.")

    def static_split_for(file_name):
        sid = static_setup_id(file_name)
        if sid not in static_split_by_setup:
            raise SystemExit(f"Static image {file_name} has setup id {sid}, "
                             f"not found in static_dataset's existing splits.")
        return static_split_by_setup[sid]

    def static_source_for(file_name):
        return f"static_setup_{static_setup_id(file_name)}"

    process_export(
        static_cvat, out_dir,
        split_for_image=static_split_for,
        source_group_for_image=static_source_for,
        stem_to_source=stem_to_source,
        source_to_component_stems=source_to_component_stems,
        counts=counts,
    )

    # ---- Each video: random 70/15/15 stratified within the video -----------
    rng = random.Random(args.seed)
    for vcvat in video_cvats:
        # Source group is the parent directory name (e.g. <bag_stem>/cvat_export)
        video_stem = vcvat.parent.name
        source_group = f"video_{video_stem}"
        print(f"\n{'=' * 70}\nVideo export: {video_stem}\n{'=' * 70}")

        # First pass: gather all image file_names in this export
        _, coco = load_coco(vcvat)
        file_names = sorted(img["file_name"] for img in coco["images"])
        rng.shuffle(file_names)

        n = len(file_names)
        n_val  = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        n_train = n - n_val - n_test

        val_set   = set(file_names[:n_val])
        test_set  = set(file_names[n_val:n_val + n_test])
        train_set = set(file_names[n_val + n_test:])
        print(f"  [INFO] {n} frames split: "
              f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

        def video_split_for(fn, _val=val_set, _test=test_set):
            if fn in _val:  return "val"
            if fn in _test: return "test"
            return "train"

        def video_source_for(fn, _sg=source_group):
            return _sg

        process_export(
            vcvat, out_dir,
            split_for_image=video_split_for,
            source_group_for_image=video_source_for,
            stem_to_source=stem_to_source,
            source_to_component_stems=source_to_component_stems,
            counts=counts,
        )

    # ---- Write source_index.json -------------------------------------------
    source_index = {
        "_paths_note": "stems are filename stems (no extension). All images live under "
                       "images/{train,val,test}/<stem>.png in this dataset's directory.",
        "stem_to_source": dict(sorted(stem_to_source.items())),
        "source_to_component_stems": {
            grp: sorted(stems) for grp, stems in sorted(source_to_component_stems.items())
        },
    }
    (out_dir / "source_index.json").write_text(json.dumps(source_index, indent=2))

    # ---- Write dataset.yaml (relative paths, 3 component classes) ----------
    yaml_text = (
        "# All paths are RELATIVE to the directory containing this file.\n"
        f"# Absolute root at write time: {out_dir.resolve()}\n"
        "# Built by 04_build_split_dataset.py from the static + video CVAT exports.\n"
        "# 'platform' polygons are NOT training targets -- only stored as paste_regions/<stem>.png\n"
        "# masks for the copy-paste augmentation step (script 05).\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        "names:\n"
        "  0: screw\n"
        "  1: nut\n"
        "  2: gear\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text)

    # ---- Summary -----------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"[OK] Built: {out_dir}")
    print("=" * 70)
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"  {split:5s}: {c['pos']} positives + {c['neg']} hard negatives "
              f"= {c['pos'] + c['neg']} images")
    n_paste = sum(1 for _ in (out_dir / "paste_regions").glob("*.png"))
    print(f"  paste_regions: {n_paste} masks written")
    print(f"\n  dataset.yaml:      {out_dir / 'dataset.yaml'}")
    print(f"  source_index.json: {out_dir / 'source_index.json'}")
    print(f"\n  Source-group summary (number of component-bearing stems per group):")
    for grp, stems in source_index["source_to_component_stems"].items():
        print(f"    {grp:32s}  {len(stems)} component frame(s)")


if __name__ == "__main__":
    main()
