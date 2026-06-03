#!/usr/bin/env python3
'''
STEP 08: Convert the cleaned + split COCO from step 07 into a YOLO-seg dataset
        physically split into train / val / test folders.

NO augmentation is performed here. All augmentation (rotation, HSV, brightness,
contrast, gamma, Gaussian noise, translate, scale, fliplr, mosaic) is applied
ONLINE during fine-tuning by YOLO and Albumentations. Online augmentation gives
the model a fresh random variant of every image every epoch — over enough
epochs that yields much more uniqueness than a fixed offline-augmented dataset.

Inputs (from step 07):
    --in_dir <path>   contains coco_clean.json, images/, splits/{train,val,test}.txt

Outputs in --out_dir:
    dataset.yaml
    images/{train,val,test}/<file_name>.png
    labels/{train,val,test}/<file_name>.txt

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):
    >> python3 scripts/08_augment_and_yolo_seg.py \
        --in_dir  <output dir from script 07> \
        --out_dir <choose an output path>
'''

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_utils


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


def load_manifest(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def ann_to_polys(ann, h, w):
    """Convert a COCO annotation's segmentation to a list of polygon point
    arrays in pixel coordinates. Handles polygon-list and RLE (compressed and
    uncompressed) formats."""
    seg = ann.get("segmentation", None)
    if seg is None:
        return []

    if isinstance(seg, list) and (len(seg) == 0 or not isinstance(seg[0], dict)):
        polys = []
        for poly in seg:
            if len(poly) < 6:
                continue
            polys.append(np.array(poly, dtype=np.float32).reshape(-1, 2))
        return polys

    rles = []
    if isinstance(seg, dict):
        rles = [seg]
    elif isinstance(seg, list) and len(seg) and isinstance(seg[0], dict):
        rles = seg
    else:
        return []

    polys = []
    for rle in rles:
        r = rle
        if isinstance(r.get("counts", None), list):
            r = mask_utils.frPyObjects(r, h, w)
            if isinstance(r, list):
                r = r[0]
        elif isinstance(r.get("counts", None), str):
            r = dict(r)
            r["counts"] = r["counts"].encode("ascii")
        mask = mask_utils.decode(r)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) < 3:
                continue
            polys.append(c.reshape(-1, 2).astype(np.float32))
    return polys


def largest_polygon(ann, h, w):
    """Return the largest polygon (by area) for a COCO annotation, or None."""
    polys = ann_to_polys(ann, h, w)
    best, best_area = None, -1.0
    for pts in polys:
        if len(pts) < 3:
            continue
        a = abs(cv2.contourArea(pts.astype(np.float32)))
        if a > best_area:
            best_area = a
            best = pts
    return best if (best is not None and best_area >= 1.0) else None


def polygon_to_yolo_line(class_id, pts, w, h):
    if pts is None or len(pts) < 3:
        return None
    ptsn = pts.astype(np.float32).copy()
    ptsn[:, 0] = np.clip(ptsn[:, 0] / w, 0.0, 1.0)
    ptsn[:, 1] = np.clip(ptsn[:, 1] / h, 0.0, 1.0)
    flat = ptsn.reshape(-1)
    if len(flat) < 6:
        return None
    return f"{class_id} " + " ".join(f"{x:.6f}" for x in flat.tolist())


def process_split(coco, in_images_dir, out_images, out_labels, fnames):
    name_to_img = {im["file_name"]: im for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for fname in tqdm(fnames, desc=f"{out_images.parent.name}/{out_images.name}"):
        im = name_to_img.get(fname)
        if im is None:
            continue
        h, w = int(im["height"]), int(im["width"])
        src = in_images_dir / fname
        if not src.exists():
            continue
        shutil.copy2(src, out_images / fname)

        lines = []
        for ann in anns_by_img.get(im["id"], []):
            best = largest_polygon(ann, h, w)
            if best is None:
                continue
            line = polygon_to_yolo_line(int(ann["category_id"]) - 1, best, w, h)
            if line:
                lines.append(line)
        (out_labels / f"{Path(fname).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True,
                    help="Directory produced by 07_filter_and_split.py")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_dir = _resolve_path(args.in_dir)
    out_dir = _resolve_path(args.out_dir)

    coco = json.loads((in_dir / "coco_clean.json").read_text())
    in_images_dir = in_dir / "images"
    train_files = load_manifest(in_dir / "splits" / "train.txt")
    val_files = load_manifest(in_dir / "splits" / "val.txt")
    test_files = load_manifest(in_dir / "splits" / "test.txt")

    print(f"[INFO] Splits: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    print(f"[INFO] Categories: {[(c['id'], c['name']) for c in coco['categories']]}")

    n_train = process_split(coco, in_images_dir,
                            out_dir / "images" / "train", out_dir / "labels" / "train",
                            train_files)
    n_val = process_split(coco, in_images_dir,
                          out_dir / "images" / "val", out_dir / "labels" / "val",
                          val_files)
    n_test = process_split(coco, in_images_dir,
                           out_dir / "images" / "test", out_dir / "labels" / "test",
                           test_files)
    print(f"[OK] Wrote train={n_train}, val={n_val}, test={n_test}")

    names = [c["name"] for c in sorted(coco["categories"], key=lambda x: x["id"])]
    yaml = f"path: {out_dir.resolve()}\n"
    yaml += "train: images/train\n"
    yaml += "val: images/val\n"
    yaml += "test: images/test\n"
    yaml += "names:\n"
    for i, n in enumerate(names):
        yaml += f"  {i}: {n}\n"
    (out_dir / "dataset.yaml").write_text(yaml)
    print(f"[OK] Wrote: {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
