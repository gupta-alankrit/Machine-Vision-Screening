#!/usr/bin/env python3
"""
STEP 09: Build a frozen YOLO-seg EVALUATION dataset (test-only) from a CVAT
COCO 1.0 export. Designed for held-out workspaces (e.g. Workspace B) where
no train/val split is needed — every image goes into the test split.

Reuses the heavy lifting from 07_filter_and_split.py + 08_augment_and_yolo_seg.py
via importlib (filenames start with a digit so a normal `import` doesnʼt work).

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):
    >> python3 scripts/09_prepare_workspace_B_dataset.py \
        --cvat_export_dir <path to an unzipped CVAT COCO 1.0 export> \
        --out_dir         <choose an output path>

Output layout (rooted at --out_dir):
    images/test/<frame>.png    # all hand-labeled frames (flat)
    labels/test/<frame>.txt    # YOLO-seg polygon labels (flat)
    coco_clean.json            # filtered COCO (audit trail)
    splits/test.txt            # filename manifest for the test split
    dataset.yaml               # relative paths only; only `test:` key set

Use with Ultralytics:
    model.val(data='<out_dir>/dataset.yaml', split='test', ...)
"""

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Reuse step 07 and step 08 via importlib (filenames begin with a digit).
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent   # <repo>/scripts/<name>.py -> <repo>


def _resolve_path(p):
    """Expand ~ and resolve. Absolute paths are returned as-is (after
    expanduser). Relative paths are resolved against the repo root (NOT the
    current working directory), so the script works regardless of where it is
    invoked from."""
    p = Path(p).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def _load_sibling(name: str, filename: str):
    """Load a sibling .py module whose filename starts with a digit."""
    spec = importlib.util.spec_from_file_location(name, str(HERE / filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_step07 = _load_sibling("_step07", "07_filter_and_split.py")
_step08 = _load_sibling("_step08", "08_augment_and_yolo_seg.py")


def main():
    ap = argparse.ArgumentParser(
        description="Build a test-only YOLO-seg eval dataset from a CVAT COCO export."
    )
    ap.add_argument("--cvat_export_dir", required=True, help="Unzipped CVAT export folder (contains annotations/ + images/).")
    ap.add_argument("--out_dir", required=True, help="Output directory for the test-only YOLO-seg dataset.")
    ap.add_argument("--keep_labels", nargs="+", default=["screw", "nut", "gear"],
                    help="Category names to keep, in order -> YOLO indices 0..K-1. Must match the order used during training.")
    ap.add_argument("--drop_labels", nargs="+", default=["ignore", "object"],
                    help="Drop every image containing any annotation of these category names.")
    args = ap.parse_args()

    export_dir = _resolve_path(args.cvat_export_dir)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Locate + clean the COCO using step 07's helpers ---------------
    coco_path, coco = _step07.load_coco_any(list(export_dir.rglob("*.json")))
    print(f"[OK] Source COCO: {coco_path}")
    print(f"[INFO] Source: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} anns, {len(coco['categories'])} categories")

    clean = _step07.filter_and_canonicalize(coco, args.keep_labels, args.drop_labels)
    print(f"[OK] After filter+canonicalize: {len(clean['images'])} images, "
          f"{len(clean['annotations'])} anns")
    print(f"[OK] Categories (kept order -> YOLO ID): "
          f"{[(c['id'], c['name']) for c in clean['categories']]}")

    # ---- 2) Build the test-split filename manifest (no train/val) ---------
    test_files = sorted([im["file_name"] for im in clean["images"]])
    print(f"[OK] {len(test_files)} files assigned to test split (eval-only — no train/val).")
    print(f"[INFO] Per-class instance count: "
          f"{_step07.per_class_counts(clean, test_files)}")

    # ---- 3) Stage all images flat in a temp dir, then convert via 08 ------
    # 08's process_split() expects all images to live in one flat directory.
    # CVAT often exports them under images/Test/ or similar; we flatten here
    # using 07's find_image_file() (which rglobs as a fallback).
    with tempfile.TemporaryDirectory() as td:
        staging = Path(td) / "images"
        staging.mkdir(parents=True, exist_ok=True)

        missing = 0
        for im in clean["images"]:
            src = _step07.find_image_file(export_dir, im["file_name"])
            if src is None:
                missing += 1
                continue
            shutil.copy2(src, staging / im["file_name"])
        if missing:
            print(f"[WARN] Could not locate {missing} image file(s) in the export.")

        # Step 08: COCO -> YOLO-seg, for the test split only.
        n_test = _step08.process_split(
            coco=clean,
            in_images_dir=staging,
            out_images=out_dir / "images" / "test",
            out_labels=out_dir / "labels" / "test",
            fnames=test_files,
        )
        print(f"[OK] Wrote {n_test} YOLO-seg test samples -> "
              f"{out_dir / 'images' / 'test'} + {out_dir / 'labels' / 'test'}")

    # ---- 4) Write dataset.yaml (relative paths only) ----------------------
    # Ultralytics' check_det_dataset() requires both `train:` and `val:` keys to
    # exist in every dataset YAML — even when only model.val(split="test") is
    # called. For an eval-only dataset, alias all three keys to the same
    # images/test folder and document loudly that this is eval-only.
    abs_root = out_dir.resolve()
    names = [c["name"] for c in sorted(clean["categories"], key=lambda x: x["id"])]
    yaml_text = (
        "# All paths are RELATIVE to the directory containing this file.\n"
        f"# Absolute root at write time: {abs_root}\n"
        "#\n"
        "# EVAL-ONLY DATASET (Workspace B / held-out). DO NOT TRAIN ON THIS YAML.\n"
        "# Ultralytics requires both `train:` and `val:` keys to exist in any dataset YAML\n"
        "# (even when only calling model.val(split='test')), so they are aliased here to\n"
        "# the same images/test folder. Always pass split='test' when consuming this file:\n"
        "#     model.val(data=<this file>, split='test', ...)\n"
        "train: images/test\n"
        "val:   images/test\n"
        "test:  images/test\n"
        "names:\n"
    )
    for i, n in enumerate(names):
        yaml_text += f"  {i}: {n}\n"
    (out_dir / "dataset.yaml").write_text(yaml_text)
    print(f"[OK] Wrote: {out_dir / 'dataset.yaml'}")

    # ---- 5) Preserve audit trail (cleaned COCO + filename manifest) -------
    (out_dir / "splits").mkdir(exist_ok=True)
    (out_dir / "coco_clean.json").write_text(json.dumps(clean))
    (out_dir / "splits" / "test.txt").write_text("\n".join(test_files) + "\n")
    print(f"[OK] Audit trail: {out_dir / 'coco_clean.json'} + {out_dir / 'splits' / 'test.txt'}")

    print(f"\n[OK] Eval dataset ready at: {out_dir}")
    print(f"     Validate with:")
    print(f"       model.val(data='{out_dir / 'dataset.yaml'}', split='test', ...)")


if __name__ == "__main__":
    main()
