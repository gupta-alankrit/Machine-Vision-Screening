#!/usr/bin/env python3
'''
STEP 07: Filter a CVAT COCO export to only the manually-labeled images and
        deterministically split them into train / val / test sets.

Pipeline:
    1) Find the COCO annotations json in --export_dir.
    2) Keep only images that have >=1 annotation in --keep_labels AND no
       annotation in --drop_labels.
    3) Re-number COCO categories to 1..K in the order given by --keep_labels
       so YOLO indices end up matching the sim baseline ([screw, nut, gear]).
    4) Shuffle (seeded) and split the kept image FILENAMES into train/val/test
       by --val_ratio and --test_ratio. Splits are recorded as filename
       manifests in <out_dir>/splits/ — images themselves are kept together in
       one images/ folder; the physical split happens later in step 08.

Output layout in --out_dir:
    coco_clean.json          # filtered + canonical-ordered COCO
    images/<file_name>.png   # every kept image, all together
    splits/
        train.txt            # one file_name per line
        val.txt
        test.txt

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):
    >> python3 scripts/07_filter_and_split.py \
        --export_dir   <path to an unzipped CVAT COCO 1.0 export> \
        --out_dir      <choose an output path> \
        --keep_labels  screw nut gear \
        --drop_labels  ignore object

Notes on splits:
  * The split is PERCENTAGE-based, not based on absolute image counts — so it
    automatically rescales when the number of kept images changes. Defaults are
    --train_ratio 0.70, --val_ratio 0.15, --test_ratio 0.15 (must sum to 1.0).
    Examples of what these defaults produce:
        n =  50 kept images -> train 34, val  8, test  8
        n = 101 kept images -> train 71, val 15, test 15
        n = 200 kept images -> train 140, val 30, test 30
    n_val and n_test are computed via round(n * ratio); n_train absorbs the
    rounding remainder so train+val+test == n exactly.
  * Splits are deterministic given --seed. The exact filenames are written to
    splits/{train,val,test}.txt so the split itself is reproducible even if
    someone later changes the seed.
'''

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path


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


def load_coco_any(json_paths):
    for p in json_paths:
        try:
            d = json.loads(Path(p).read_text())
            if "images" in d and "annotations" in d and "categories" in d:
                return Path(p), d
        except Exception:
            pass
    raise RuntimeError("Could not find a valid COCO JSON in export.")


def find_image_file(export_dir: Path, file_name: str):
    candidates = [
        export_dir / file_name,
        export_dir / "images" / file_name,
        export_dir / "data" / file_name,
        export_dir / "images" / "default" / file_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = list(export_dir.rglob(file_name))
    return hits[0] if hits else None


def filter_and_canonicalize(coco, keep_labels, drop_labels):
    """Drop every image that contains any annotation in `drop_labels`, keep
    every remaining image that contains at least one annotation in
    `keep_labels`, and re-id categories to 1..K in keep_labels order."""
    name_to_catid = {c["name"]: c["id"] for c in coco["categories"]}
    keep_catids = {name_to_catid[n] for n in keep_labels if n in name_to_catid}
    drop_catids = {name_to_catid[n] for n in drop_labels if n in name_to_catid}

    missing_keep = [n for n in keep_labels if n not in name_to_catid]
    if missing_keep:
        print(f"[WARN] --keep_labels missing from source COCO: {missing_keep}")
    missing_drop = [n for n in drop_labels if n not in name_to_catid]
    if missing_drop:
        print(f"[WARN] --drop_labels missing from source COCO: {missing_drop}")

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    keep_imgs = []
    for img in coco["images"]:
        anns = anns_by_image.get(img["id"], [])
        has_keep = any(a.get("category_id") in keep_catids for a in anns)
        has_drop = bool(drop_catids) and any(a.get("category_id") in drop_catids for a in anns)
        if has_keep and not has_drop:
            keep_imgs.append(img)
    keep_ids = {im["id"] for im in keep_imgs}

    keep_anns = [
        a for a in coco["annotations"]
        if a["image_id"] in keep_ids and a.get("category_id") in keep_catids
    ]

    # canonical category remap
    new_cats = [
        {"id": i + 1, "name": n}
        for i, n in enumerate(keep_labels) if n in name_to_catid
    ]
    old_to_new_catid = {name_to_catid[c["name"]]: c["id"] for c in new_cats}
    for a in keep_anns:
        a["category_id"] = old_to_new_catid[a["category_id"]]

    # reindex images and annotations
    old_to_new_imgid = {}
    new_images = []
    for new_id, im in enumerate(keep_imgs, start=1):
        old_to_new_imgid[im["id"]] = new_id
        ni = dict(im); ni["id"] = new_id
        new_images.append(ni)
    new_anns = []
    for new_aid, a in enumerate(keep_anns, start=1):
        na = dict(a); na["id"] = new_aid
        na["image_id"] = old_to_new_imgid[a["image_id"]]
        new_anns.append(na)

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": new_cats,
        "images": new_images,
        "annotations": new_anns,
    }


def make_splits(filenames, val_ratio, test_ratio, seed):
    rng = random.Random(seed)
    files = sorted(filenames)
    rng.shuffle(files)
    n = len(files)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Bad split sizes: n={n}, val_ratio={val_ratio}, test_ratio={test_ratio}")
    train = sorted(files[:n_train])
    val = sorted(files[n_train:n_train + n_val])
    test = sorted(files[n_train + n_val:])
    return train, val, test


def per_class_counts(coco, file_subset):
    name_subset = set(file_subset)
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_id_in_subset = {im["id"] for im in coco["images"] if im["file_name"] in name_subset}
    cnt = Counter()
    for a in coco["annotations"]:
        if a["image_id"] in img_id_in_subset:
            cnt[id_to_name.get(a["category_id"], a["category_id"])] += 1
    return dict(cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True,
                    help="Unzipped CVAT export folder (contains annotations/ and images/).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--keep_labels", nargs="+", default=["screw", "nut", "gear"],
                    help="Category names to keep, in the order that becomes YOLO indices 0..K-1.")
    ap.add_argument("--drop_labels", nargs="+", default=["ignore", "object"],
                    help="Drop every image containing at least one annotation of ANY of these category names.")
    ap.add_argument("--train_ratio", type=float, default=0.70,
                    help="Fraction of kept images assigned to train. Must sum to 1 with val/test.")
    ap.add_argument("--val_ratio", type=float, default=0.15,
                    help="Fraction of kept images assigned to val.")
    ap.add_argument("--test_ratio", type=float, default=0.15,
                    help="Fraction of kept images assigned to test.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"--train_ratio + --val_ratio + --test_ratio must sum to 1.0 (got {total:.6f})"
        )

    export_dir = _resolve_path(args.export_dir)
    out_dir = _resolve_path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)

    coco_path, coco = load_coco_any(list(export_dir.rglob("*.json")))
    print(f"[OK] Source COCO: {coco_path}")
    print(f"[INFO] Source: {len(coco['images'])} images, {len(coco['annotations'])} anns, "
          f"{len(coco['categories'])} categories")

    clean = filter_and_canonicalize(coco, args.keep_labels, args.drop_labels)
    print(f"[OK] After filter+canonicalize: {len(clean['images'])} images, "
          f"{len(clean['annotations'])} anns")
    print(f"[OK] Categories: {[(c['id'], c['name']) for c in clean['categories']]}")

    filenames = [im["file_name"] for im in clean["images"]]
    train, val, test = make_splits(filenames, args.val_ratio, args.test_ratio, args.seed)
    print(
        f"[OK] Split (seed={args.seed}, ratios train/val/test = "
        f"{args.train_ratio:.2f}/{args.val_ratio:.2f}/{args.test_ratio:.2f}): "
        f"train={len(train)}, val={len(val)}, test={len(test)}"
    )

    # Per-class counts per split (useful sanity check)
    for split_name, files in [("train", train), ("val", val), ("test", test)]:
        print(f"[INFO] {split_name} per-class instance count: {per_class_counts(clean, files)}")

    missing = 0
    for im in clean["images"]:
        src = find_image_file(export_dir, im["file_name"])
        if src is None:
            missing += 1
            continue
        shutil.copy2(src, out_dir / "images" / im["file_name"])
    if missing:
        print(f"[WARN] Missing image files: {missing}")

    (out_dir / "coco_clean.json").write_text(json.dumps(clean))
    (out_dir / "splits" / "train.txt").write_text("\n".join(train) + "\n")
    (out_dir / "splits" / "val.txt").write_text("\n".join(val) + "\n")
    (out_dir / "splits" / "test.txt").write_text("\n".join(test) + "\n")

    print(f"[OK] Wrote: {out_dir / 'coco_clean.json'}")
    print(f"[OK] Wrote: {out_dir / 'splits' / '{train,val,test}.txt'}")


if __name__ == "__main__":
    main()
