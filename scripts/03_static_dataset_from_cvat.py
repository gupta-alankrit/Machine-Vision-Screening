#!/usr/bin/env python3
"""
STEP 03: Build a YOLO-seg dataset from the CVAT COCO export of the 32 static
images (16 (with, empty) pairs).

The CVAT export has FOUR label classes:
  screw, nut, gear, platform

This script:
  1. Splits the 32 images into train/val by setup (so X_Color and X_empty stay
     together -- pairs are not separated across splits).
  2. For each image:
       - Writes the YOLO-seg label file with ONLY screw/nut/gear lines.
         Platform polygons are NEVER written to labels/ (they're not training
         targets — they only define where copy-paste is allowed).
       - If the image has no component polygons, writes an EMPTY .txt
         (training treats it as a hard negative).
       - If the image has any platform polygons, rasterizes the union of them
         to a binary mask under paste_regions/<stem>.png.
  3. Writes dataset.yaml with relative paths and only the three component
     classes in the `names` list (no platform).

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):
  >> python3 scripts/03_static_dataset_from_cvat.py \
       --cvat_export_dirs \
           data/workspace_B/one_component_image_empty_scene_pair/cvat_export_1 \
           data/workspace_B/one_component_image_empty_scene_pair/cvat_export_2 \
       --val_setups  3 1 \
       --test_setups 3 1 \
       --out_dir <choose a path for the static YOLO dataset>

  -> writes the YOLO static_dataset at --out_dir AND a merged CVAT export at
     <out_dir>/../cvat_export_combined/  (point script 04's --static_cvat_dir
     at that combined folder).

Output layout:
  <out_dir>/
    images/train/<setup>_Color.png       # with-component (positive sample)
    images/train/<setup>_empty.png       # empty (hard negative)
    images/val/...
    images/test/...
    labels/train/<setup>_Color.txt       # YOLO-seg, component classes only
    labels/train/<setup>_empty.txt       # empty file (hard negative)
    labels/val/...
    labels/test/...
    paste_regions/<setup>_Color.png      # binary mask (0/255), union of platform polygons
    paste_regions/<setup>_empty.png
    dataset.yaml                         # relative paths, 3 component classes only
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
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


# Canonical YOLO class order. Must match training scripts elsewhere in the project.
YOLO_CLASS_NAMES = ["screw", "nut", "gear"]
PLATFORM_LABEL = "platform"


def find_source_image(export_dir: Path, file_name: str):
    """Locate an image file inside the CVAT export, regardless of subset folder."""
    candidates = [
        export_dir / "images" / file_name,
        export_dir / "images" / "default" / file_name,
        export_dir / "images" / "Test" / file_name,
        export_dir / file_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: search recursively
    hits = list(export_dir.rglob(file_name))
    return hits[0] if hits else None


def setup_key(file_name: str) -> str:
    """'10_Color.png' -> '10'. Used to keep (with, empty) pairs in the same split."""
    stem = Path(file_name).stem
    m = re.match(r"^(\d+)_", stem)
    return m.group(1) if m else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvat_export_dirs", required=True, nargs="+",
                    help="One or more unzipped CVAT COCO export folders (each contains "
                         "annotations/ + images/). Useful when the static images were labeled "
                         "across multiple CVAT tasks. Setup-id collisions across exports "
                         "(e.g. both have '10_Color.png') are rejected — re-number first.")
    ap.add_argument("--out_dir", required=True,
                    help="Output YOLO-seg dataset directory (will be created).")
    ap.add_argument("--val_setups",  type=int, nargs="+", default=[3],
                    help="Number of setups to hold out for val PER EXPORT. Either a single "
                         "int (applied to every export) or one int per --cvat_export_dirs "
                         "entry. Example: --cvat_export_dirs E1 E2 --val_setups 3 1 -> "
                         "3 val from E1, 1 val from E2.")
    ap.add_argument("--test_setups", type=int, nargs="+", default=[3],
                    help="Number of setups to hold out for test PER EXPORT. Same semantics "
                         "as --val_setups. Per-export stratification ensures BOTH canonical "
                         "and sideways setups (if you exported them in separate tasks) appear "
                         "in val and test, instead of all landing in train by luck of shuffle.")
    ap.add_argument("--combined_export_out", default=None,
                    help="Path to write the combined CVAT export folder (renumbered ids, "
                         "merged COCO json, copied images). Default: a sibling of --out_dir "
                         "named 'cvat_export_combined'. Pass this combined folder as "
                         "--static_cvat_dir to script 04.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    export_dirs = [_resolve_path(p) for p in args.cvat_export_dirs]
    out_dir     = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + validate every export. Category ids may differ per export
    # (CVAT assigns them per-task), so we keep one yolo/platform map per export.
    exports = []   # list of (export_dir, coco, yolo_id_for_cat_id, platform_cat_id, anns_by_img)
    for ed in export_dirs:
        json_paths = sorted((ed / "annotations").glob("*.json"))
        if not json_paths:
            raise SystemExit(f"No JSON files under {ed / 'annotations'}")
        coco_path = json_paths[0]
        print(f"[INFO] Reading: {coco_path}")
        coco = json.loads(coco_path.read_text())

        name_to_cat_id = {c["name"]: c["id"] for c in coco["categories"]}
        missing = [n for n in YOLO_CLASS_NAMES + [PLATFORM_LABEL] if n not in name_to_cat_id]
        if missing:
            raise SystemExit(
                f"Missing required label(s) in {ed}: {missing}. "
                f"Found: {list(name_to_cat_id.keys())}"
            )
        yolo_id_for_cat_id = {name_to_cat_id[n]: i for i, n in enumerate(YOLO_CLASS_NAMES)}
        platform_cat_id   = name_to_cat_id[PLATFORM_LABEL]
        print(f"  [INFO] Category mapping: "
              f"{ {c['name']: c['id'] for c in coco['categories']} }")

        anns_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)

        exports.append((ed, coco, yolo_id_for_cat_id, platform_cat_id, anns_by_img))

    # ---- Split by setup so (with, empty) pairs stay together. Setup id is the
    # numeric filename prefix ('10' from '10_Color.png') — must be unique across
    # all exports, else we wouldn't know which export owns the setup.
    images_by_setup = defaultdict(list)   # setup_id -> list of (img dict, export_idx)
    for idx, (_, coco, _, _, _) in enumerate(exports):
        for img in coco["images"]:
            images_by_setup[setup_key(img["file_name"])].append((img, idx))

    # Reject collisions explicitly so the user knows to re-number.
    for sk, entries in images_by_setup.items():
        exp_indices = {ei for _, ei in entries}
        if len(exp_indices) > 1:
            colliding = sorted({export_dirs[ei].name for ei in exp_indices})
            raise SystemExit(
                f"Setup id '{sk}' appears in multiple exports: {colliding}. "
                f"Re-number the filenames in one of them and re-export."
            )

    # ---- Per-export stratified split. Each export gets its own val/test count
    # so canonical and sideways exports both contribute setups to val and test.
    n_exports = len(exports)
    def _expand_per_export(values, name):
        if len(values) == 1:
            return [values[0]] * n_exports
        if len(values) != n_exports:
            raise SystemExit(
                f"--{name} expects either 1 int (applied to all exports) or "
                f"exactly {n_exports} ints (one per --cvat_export_dirs entry). "
                f"Got {len(values)}: {values}."
            )
        return list(values)

    val_per_export  = _expand_per_export(args.val_setups,  "val_setups")
    test_per_export = _expand_per_export(args.test_setups, "test_setups")

    # Bucket setup ids by which export owns them (each setup id maps to exactly
    # one export thanks to the earlier collision check).
    setups_by_export = defaultdict(list)
    for sk, entries in images_by_setup.items():
        owning_idx = entries[0][1]
        setups_by_export[owning_idx].append(sk)

    random.seed(args.seed)
    val_setup_set, test_setup_set, train_setup_set = set(), set(), set()
    for ei in range(n_exports):
        sids = sorted(setups_by_export.get(ei, []))
        random.shuffle(sids)
        n_total = len(sids)
        n_val  = max(0, min(val_per_export[ei],  n_total))
        n_test = max(0, min(test_per_export[ei], n_total - n_val))
        n_train = n_total - n_val - n_test
        if n_total > 0 and n_train < 1:
            print(f"  [WARN] export '{export_dirs[ei].name}': val+test consume all setups, "
                  f"no train. Got val={n_val}, test={n_test}, total={n_total}.")
        val_setup_set.update(sids[:n_val])
        test_setup_set.update(sids[n_val:n_val + n_test])
        train_setup_set.update(sids[n_val + n_test:])
        print(f"  [INFO] export '{export_dirs[ei].name}': "
              f"{n_train} train, {n_val} val, {n_test} test setups -> "
              f"val={sorted(sids[:n_val])} "
              f"test={sorted(sids[n_val:n_val + n_test])}")

    if (len(train_setup_set) + len(val_setup_set) + len(test_setup_set)) < 1:
        raise SystemExit("No setups found across the provided exports.")
    print(f"[INFO] Combined split (seed={args.seed}): "
          f"{len(train_setup_set)} train, {len(val_setup_set)} val, {len(test_setup_set)} test setups")

    # ---- Make output dirs --------------------------------------------------
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "paste_regions").mkdir(parents=True, exist_ok=True)

    # ---- Process every image ----------------------------------------------
    n_pos   = {"train": 0, "val": 0, "test": 0}
    n_neg   = {"train": 0, "val": 0, "test": 0}
    n_paste = 0
    n_missing_src = 0

    # Flatten the per-setup mapping back to a per-image iteration order, keeping
    # the export index attached so we use the right source folder + cat maps.
    all_image_entries = [
        (img, export_idx)
        for entries in images_by_setup.values()
        for (img, export_idx) in entries
    ]

    for img, export_idx in all_image_entries:
        export_dir, _, yolo_id_for_cat_id, platform_cat_id, anns_by_img = exports[export_idx]
        file_name = img["file_name"]
        W, H = int(img["width"]), int(img["height"])
        setup = setup_key(file_name)
        if setup in val_setup_set:
            split = "val"
        elif setup in test_setup_set:
            split = "test"
        else:
            split = "train"

        # Locate source image in the owning export
        src_path = find_source_image(export_dir, file_name)
        if src_path is None:
            print(f"  [WARN] source image not found: {file_name}  (export: {export_dir.name})")
            n_missing_src += 1
            continue

        # Copy image to images/<split>/
        dst_img = out_dir / "images" / split / file_name
        shutil.copy2(src_path, dst_img)

        # Walk annotations: components -> YOLO labels; platform -> mask
        anns = anns_by_img.get(img["id"], [])
        yolo_lines = []
        platform_polys_px = []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg or not isinstance(seg, list) or not seg[0]:
                continue
            poly_flat = seg[0]                       # COCO: [x1, y1, x2, y2, ...]
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
                # Normalize to image dims
                norm = []
                for i in range(0, len(poly_flat), 2):
                    norm.append(poly_flat[i]     / W)
                    norm.append(poly_flat[i + 1] / H)
                yolo_lines.append(f"{yolo_id} " + " ".join(f"{v:.6f}" for v in norm))
            # else: unknown category — silently skip

        # Write YOLO label file (empty file == hard negative)
        dst_lbl = out_dir / "labels" / split / (Path(file_name).stem + ".txt")
        dst_lbl.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        if yolo_lines:
            n_pos[split] += 1
        else:
            n_neg[split] += 1

        # Write paste region mask
        if platform_polys_px:
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(mask, platform_polys_px, 255)
            mask_path = out_dir / "paste_regions" / (Path(file_name).stem + ".png")
            cv2.imwrite(str(mask_path), mask)
            n_paste += 1

    # ---- Write the combined CVAT export folder ------------------------------
    # Downstream scripts (04) expect a SINGLE CVAT export with one annotations/
    # JSON and one images/ folder. We merge all input exports into this layout,
    # renumbering image/annotation ids to avoid collisions. Categories are
    # taken from the first export and remapped on each annotation to match.
    combined_out = (_resolve_path(args.combined_export_out)
                    if args.combined_export_out
                    else out_dir.parent / "cvat_export_combined")
    if combined_out.exists():
        shutil.rmtree(combined_out)
    (combined_out / "annotations").mkdir(parents=True)
    (combined_out / "images" / "default").mkdir(parents=True)

    # Take categories from export 0 as the canonical schema.
    canonical_categories = exports[0][1]["categories"]
    name_to_canonical_id = {c["name"]: c["id"] for c in canonical_categories}

    merged_images = []
    merged_annotations = []
    next_img_id = 1
    next_ann_id = 1
    for ei, (ed, coco, _, _, _) in enumerate(exports):
        # Per-export remap: this export's cat ids -> canonical cat ids
        local_to_canonical = {}
        for c in coco["categories"]:
            if c["name"] not in name_to_canonical_id:
                raise SystemExit(f"Export {ed.name} has category '{c['name']}' "
                                 f"not present in the first export's schema.")
            local_to_canonical[c["id"]] = name_to_canonical_id[c["name"]]

        local_img_id_remap = {}
        for img in coco["images"]:
            file_name = img["file_name"]
            src_path = find_source_image(ed, file_name)
            if src_path is None:
                # Already warned above during the YOLO pass; skip here too.
                continue
            new_img = dict(img)
            new_img["id"] = next_img_id
            local_img_id_remap[img["id"]] = next_img_id
            merged_images.append(new_img)
            shutil.copy2(src_path, combined_out / "images" / "default" / file_name)
            next_img_id += 1

        for ann in coco["annotations"]:
            if ann["image_id"] not in local_img_id_remap:
                continue
            new_ann = dict(ann)
            new_ann["id"]          = next_ann_id
            new_ann["image_id"]    = local_img_id_remap[ann["image_id"]]
            new_ann["category_id"] = local_to_canonical[ann["category_id"]]
            merged_annotations.append(new_ann)
            next_ann_id += 1

    merged_coco = {
        "licenses":    exports[0][1].get("licenses",    []),
        "info":        exports[0][1].get("info",        {}),
        "categories":  canonical_categories,
        "images":      merged_images,
        "annotations": merged_annotations,
    }
    (combined_out / "annotations" / "instances_default.json").write_text(
        json.dumps(merged_coco, indent=2)
    )
    print(f"[INFO] Wrote combined CVAT export: {combined_out}  "
          f"({len(merged_images)} images, {len(merged_annotations)} annotations)")

    # ---- Write dataset.yaml (relative paths, three component classes only) -
    abs_root = out_dir.resolve()
    yaml_text = (
        "# All paths are RELATIVE to the directory containing this file.\n"
        f"# Absolute root at write time: {abs_root}\n"
        "# Built by 03_static_dataset_from_cvat.py from the (with, empty) pair captures.\n"
        "# 'platform' polygons from CVAT are NOT a training class -- they only feed the\n"
        "# paste_regions/ masks used by script 05's copy-paste augmentation.\n"
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
    print("\n" + "=" * 60)
    print(f"[OK] Built: {out_dir}")
    print("=" * 60)
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: {n_pos[sp]} positives + {n_neg[sp]} hard negatives "
              f"= {n_pos[sp] + n_neg[sp]} images")
    print(f"  paste_regions written for {n_paste} images")
    if n_missing_src:
        print(f"  [WARN] {n_missing_src} image(s) referenced in COCO json but not found on disk")
    print(f"\n  dataset.yaml:  {out_dir / 'dataset.yaml'}")
    print(f"  paste_regions: {out_dir / 'paste_regions/'}")


if __name__ == "__main__":
    main()
