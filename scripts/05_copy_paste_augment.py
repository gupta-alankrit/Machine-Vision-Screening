#!/usr/bin/env python3
"""
STEP 05: SAM-2 copy-paste augmentation for the fine-tuning dataset.

Takes:
  - a SAM-2 mask bank (output of script 02)
  - the existing YOLO-seg fine-tuning dataset

Produces a new augmented dataset where each labeled training image gets K
augmented variants with additional components pasted on top. Originals, val,
test, and hard-negative training images are copied through unchanged.

Per-class paste sizes are auto-calibrated from the source dataset's existing
labels (median pixel area per class), with configurable jitter at paste time.

Run (paths below are repo-relative; the script resolves them against the
repo root so it works regardless of the current working directory):
  >> python3 scripts/05_copy_paste_augment.py \
       --mask_bank       <stage 2 out> \
       --source_dataset  <stage 4 out> \
       --out_dataset     <choose a path for split_augmented> \
       --augs_per_image  15 \
       --paste_count_min 1 \
       --paste_count_max 6

Output layout:
  <out_dataset>/
    images/train/<stem>.png            # all original train images (copied)
    images/train/<stem>_aug0.png       # augmented variant 0
    images/train/<stem>_aug1.png       # augmented variant 1
    ...
    images/val/  <stem>.png            # val copied unchanged
    images/test/ <stem>.png            # test copied unchanged
    labels/{train,val,test}/<stem>.txt
    dataset.yaml                       # relative paths only
    augmentation_log.json              # per-augmented-image record (paths relative)
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


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


# ----------------------------------------------------------------------------
# Helpers: YOLO label I/O + polygon math
# ----------------------------------------------------------------------------

def read_yolo_seg(path: Path):
    """Return list of (class_id, polygon_norm) tuples. Empty list if no labels."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        poly = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
        if len(poly) >= 3:
            out.append((cls, poly))
    return out


def write_yolo_seg(path: Path, instances):
    """instances = list of (class_id, polygon_norm). Writes one line per instance."""
    lines = []
    for cls, poly in instances:
        flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
        lines.append(f"{cls} {flat}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def poly_norm_to_px(poly_norm, h, w):
    return np.array([[int(x * w), int(y * h)] for x, y in poly_norm], dtype=np.int32)


def poly_px_to_norm(poly_px, h, w):
    return [(float(x) / w, float(y) / h) for x, y in poly_px]


def poly_to_mask(poly_px, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [poly_px], 1)
    return m


def mask_to_largest_polygon(mask: np.ndarray):
    """Extract the largest contour as a polygon (Nx2 int32). None if mask is empty."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 6:
        return None
    return biggest.reshape(-1, 2)


def bbox_iou(a, b):
    """a, b = (x0, y0, x1, y1) tuples. Returns IoU."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------------
# Calibration: per-class target paste size from source dataset labels
# ----------------------------------------------------------------------------

def calibrate_paste_sizes(labels_dir: Path, img_w: int, img_h: int):
    """For each class_id, return (median_area_px, p05_area_px, p95_area_px)."""
    per_class = defaultdict(list)
    for lbl in labels_dir.rglob("*.txt"):
        for cls, poly in read_yolo_seg(lbl):
            poly_px = poly_norm_to_px(poly, img_h, img_w)
            mask = poly_to_mask(poly_px, img_h, img_w)
            area = int(mask.sum())
            if area > 0:
                per_class[cls].append(area)
    out = {}
    for cls, areas in per_class.items():
        arr = np.array(areas)
        out[cls] = {
            "n_samples": int(len(arr)),
            "median_area_px": int(np.median(arr)),
            "p05_area_px":    int(np.percentile(arr, 5)),
            "p95_area_px":    int(np.percentile(arr, 95)),
        }
    return out


# ----------------------------------------------------------------------------
# Mask bank loading
# ----------------------------------------------------------------------------

def discover_mask_bank(mask_bank: Path):
    """Return dict: component_id -> list of (rgb_path, mask_path) tuples."""
    by_comp = defaultdict(list)
    for comp_dir in sorted(mask_bank.iterdir()):
        if not comp_dir.is_dir():
            continue
        comp_id = comp_dir.name
        for video_dir in sorted(comp_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            for mask_p in sorted(video_dir.glob("*_mask.png")):
                rgb_p = video_dir / mask_p.name.replace("_mask.png", ".png")
                if rgb_p.exists():
                    by_comp[comp_id].append((rgb_p, mask_p))
    return by_comp


def component_to_class(component_id: str, name_to_id: dict):
    """'screw_a' -> 0 (screw). Returns None if class prefix not in name_to_id."""
    class_name = component_id.split("_")[0]
    return name_to_id.get(class_name)


# ----------------------------------------------------------------------------
# Core: paste one component onto a background
# ----------------------------------------------------------------------------

def extract_component_patch(rgb_path: Path, mask_path: Path):
    """Read RGB+mask, return (cropped_rgb, cropped_mask) tight around the mask bbox."""
    rgb = cv2.imread(str(rgb_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None:
        return None
    if mask.shape != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask > 127).astype(np.uint8)
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return rgb[y0:y1, x0:x1].copy(), mask_bin[y0:y1, x0:x1].copy()


def tight_crop_to_mask(rgb_patch, mask_patch):
    """Re-crop both to the mask's bounding box. Returns (None, None) if mask is empty."""
    ys, xs = np.where(mask_patch > 0)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return rgb_patch[y0:y1, x0:x1].copy(), mask_patch[y0:y1, x0:x1].copy()


def rotate_patch(rgb_patch, mask_patch, angle_deg):
    """Rotate patch + mask in the image plane by angle_deg around the patch center.
    Returns a possibly-larger image whose bounding box contains the rotated content."""
    h, w = mask_patch.shape
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Expand the output canvas so the rotated patch is fully contained.
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2.0
    M[1, 2] += (new_h - h) / 2.0

    rgb_r = cv2.warpAffine(rgb_patch, M, (new_w, new_h),
                           flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask_r = cv2.warpAffine(mask_patch, M, (new_w, new_h),
                            flags=cv2.INTER_NEAREST, borderValue=0)
    return rgb_r, mask_r


def perspective_warp_patch(rgb_patch, mask_patch, strength):
    """Random mild perspective warp. `strength` is the max corner displacement as a
    fraction of the patch dimension (e.g. 0.05 = up to ±5%). Returns same-shape patches."""
    h, w = mask_patch.shape
    dx = w * strength
    dy = h * strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [0 + random.uniform(-dx, dx), 0 + random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), 0 + random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
        [0 + random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    rgb_w = cv2.warpPerspective(rgb_patch, M, (w, h),
                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask_w = cv2.warpPerspective(mask_patch, M, (w, h),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
    return rgb_w, mask_w


def resize_patch(rgb_patch, mask_patch, target_area_px):
    """Resize both so the mask covers ~target_area_px pixels. Keeps aspect ratio."""
    cur_area = int(mask_patch.sum())
    if cur_area == 0:
        return None, None
    scale = float(np.sqrt(target_area_px / cur_area))
    new_w = max(8, int(rgb_patch.shape[1] * scale))
    new_h = max(8, int(rgb_patch.shape[0] * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    rgb_r = cv2.resize(rgb_patch, (new_w, new_h), interpolation=interp)
    mask_r = cv2.resize(mask_patch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return rgb_r, mask_r


def find_paste_location(patch_mask, bg_w, bg_h, occupied_bboxes,
                        paste_region_mask=None, max_iou=0.05, n_tries=30):
    """Return (x0, y0) such that the pasted patch satisfies BOTH:
      - bbox-IoU with each entry in `occupied_bboxes` is <= max_iou, AND
      - if `paste_region_mask` is given, EVERY foreground pixel of `patch_mask`
        (after positioning) lands inside the paste-region mask. Stricter than the
        previous "center inside mask" check — guarantees that no pixel of the
        pasted component extends beyond the platform boundary.
    Returns None if no valid spot found in n_tries.
    """
    patch_h, patch_w = patch_mask.shape
    patch_fg = patch_mask > 0
    for _ in range(n_tries):
        x0 = random.randint(0, max(0, bg_w - patch_w))
        y0 = random.randint(0, max(0, bg_h - patch_h))
        cand = (x0, y0, x0 + patch_w, y0 + patch_h)
        if not all(bbox_iou(cand, b) <= max_iou for b in occupied_bboxes):
            continue
        if paste_region_mask is not None:
            pr_crop = paste_region_mask[y0:y0 + patch_h, x0:x0 + patch_w]
            if pr_crop.shape != patch_mask.shape:
                continue   # patch would clip off bg edge; bounds above should prevent this
            # Every foreground patch pixel must land where paste_region is nonzero.
            if (patch_fg & (pr_crop == 0)).any():
                continue
        return (x0, y0)
    return None


def alpha_paste(bg, patch_rgb, patch_mask, x0, y0, feather_px=5):
    """Paste patch onto bg at (x0, y0) with an INWARD-only feathered alpha. In-place on bg.

    The naive "blur the binary mask" approach spreads alpha SYMMETRICALLY across
    the mask boundary, which (a) leaves a halo of partial-patch color OUTSIDE the
    silhouette (visible as a fringe against the bg) and (b) makes the boundary
    pixel itself ~50% bg. Instead we:
      1. erode the binary mask by feather_px (the "core" stays at alpha=1),
      2. Gaussian-blur the eroded mask to create a smooth ramp from core->edge,
      3. clamp the result to the ORIGINAL mask so alpha is exactly 0 outside.
    Net effect: no patch color leaks past the silhouette, and the in-silhouette
    edge fades smoothly into bg over `feather_px` pixels — eliminating the
    paper-cutout look without producing a halo.
    """
    h, w = patch_mask.shape
    binary = (patch_mask > 0).astype(np.uint8)
    if feather_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * feather_px + 1, 2 * feather_px + 1)
        )
        core = cv2.erode(binary, kernel, iterations=1)
        k = max(1, 2 * feather_px + 1)
        alpha = cv2.GaussianBlur(core.astype(np.float32), (k, k), feather_px)
        # Clamp: zero outside the original mask silhouette (no fringe).
        alpha = alpha * binary.astype(np.float32)
    else:
        alpha = binary.astype(np.float32)
    alpha_3 = np.stack([alpha] * 3, axis=-1)
    roi = bg[y0:y0 + h, x0:x0 + w].astype(np.float32)
    composite = patch_rgb.astype(np.float32) * alpha_3 + roi * (1.0 - alpha_3)
    bg[y0:y0 + h, x0:x0 + w] = np.clip(composite, 0, 255).astype(np.uint8)


def color_match_to_local_bg(patch_rgb, patch_mask, bg, x0, y0):
    """Reinhard-style color transfer: match the patch foreground's per-channel
    mean+std (in Lab color space) to the local background region underneath.
    Reduces 'pasted look' from cross-camera color cast / exposure mismatch.
    Returns a new patch_rgb (background pixels of the patch are left unchanged).
    """
    h, w = patch_mask.shape
    H, W = bg.shape[:2]
    # find_paste_location already ensures the patch fits inside bg, but clamp anyway.
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    bg_region = bg[y0:y1, x0:x1]
    if bg_region.size == 0:
        return patch_rgb

    src_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(bg_region, cv2.COLOR_BGR2LAB).astype(np.float32)

    fg_mask = patch_mask > 0
    if fg_mask.sum() < 10:
        return patch_rgb

    # Source stats: only patch foreground pixels (don't bias by 0-valued bg pixels of patch).
    fg = src_lab[fg_mask]
    src_mean = fg.mean(axis=0)
    src_std  = fg.std(axis=0) + 1e-6

    # Destination stats: entire local bg region (a few existing fg pixels would skew
    # things slightly but in practice this is well behaved).
    dst_pixels = dst_lab.reshape(-1, 3)
    dst_mean = dst_pixels.mean(axis=0)
    dst_std  = dst_pixels.std(axis=0) + 1e-6

    out_lab = src_lab.copy()
    out_lab[fg_mask] = (src_lab[fg_mask] - src_mean) / src_std * dst_std + dst_mean
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def noise_match_to_local_bg(patch_rgb, patch_mask, bg, x0, y0):
    """Add Gaussian noise to the patch foreground so that the patch's
    high-frequency content (Laplacian-std proxy for noise) approaches the local
    background's. Cheaply masks the 'too clean' look of higher-end-camera patches
    pasted into deployment-camera scenes. No-op if patch is already noisier.
    """
    h, w = patch_mask.shape
    H, W = bg.shape[:2]
    x1, y1 = min(W, x0 + w), min(H, y0 + h)
    bg_region = bg[y0:y1, x0:x1]
    if bg_region.size == 0:
        return patch_rgb

    bg_gray    = cv2.cvtColor(bg_region, cv2.COLOR_BGR2GRAY)
    patch_gray = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
    # Laplacian variance is a known proxy for image sharpness/noise. Divide by a
    # heuristic constant to convert it to "perceived noise std" in pixel intensity.
    bg_noise    = float(cv2.Laplacian(bg_gray,    cv2.CV_64F).std()) / 4.0
    patch_noise = float(cv2.Laplacian(patch_gray, cv2.CV_64F).std()) / 4.0

    if bg_noise <= patch_noise:
        return patch_rgb  # patch is already at least as noisy as bg

    delta_std = float(np.sqrt(bg_noise ** 2 - patch_noise ** 2))
    noise = np.random.normal(0.0, delta_std, patch_rgb.shape)
    noisy = np.clip(patch_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Apply noise only on the foreground pixels of the patch; background pixels of
    # the patch are about to be excluded by the mask at paste time anyway.
    fg_3 = np.stack([patch_mask > 0] * 3, axis=-1)
    return np.where(fg_3, noisy, patch_rgb)


def _build_scene_component_mask(bg_shape, scene_component_polys_px):
    """Rasterize all existing-component polygons into a single binary mask of bg."""
    H, W = bg_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly_px in scene_component_polys_px:
        if poly_px is not None and len(poly_px) >= 3:
            cv2.fillPoly(mask, [poly_px], 1)
    return mask


def color_match_to_scene_components(patch_rgb, patch_mask, bg, scene_component_polys_px):
    """Reinhard-style color transfer in Lab. Target = pixels of EXISTING COMPONENTS
    in this bg image (read from the labels), NOT the surrounding background surface.

    Rationale: the goal of color matching for copy-paste is to make the pasted
    patch look like it was photographed under the same lighting/camera as the
    other components in the scene. The local background surface (table, paper,
    etc.) is the WRONG target — it would tint the patch to look like the table.

    Falls back to leaving the patch unchanged if there are no existing components
    in the bg (rather than falling back to local bg, which is the wrong target).
    """
    if not scene_component_polys_px:
        return patch_rgb

    scene_mask = _build_scene_component_mask(bg.shape, scene_component_polys_px)
    if int(scene_mask.sum()) < 50:
        return patch_rgb

    bg_lab    = cv2.cvtColor(bg,        cv2.COLOR_BGR2LAB).astype(np.float32)
    patch_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)

    target_pixels = bg_lab[scene_mask > 0]
    target_mean = target_pixels.mean(axis=0)
    target_std  = target_pixels.std(axis=0) + 1e-6

    fg_mask = patch_mask > 0
    if fg_mask.sum() < 10:
        return patch_rgb
    src_pixels = patch_lab[fg_mask]
    src_mean = src_pixels.mean(axis=0)
    src_std  = src_pixels.std(axis=0) + 1e-6

    out_lab = patch_lab.copy()
    out_lab[fg_mask] = (patch_lab[fg_mask] - src_mean) / src_std * target_std + target_mean
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def noise_match_to_scene_components(patch_rgb, patch_mask, bg, scene_component_polys_px):
    """Add Gaussian noise to the patch foreground so its noise level approaches
    that of the EXISTING COMPONENT pixels in this bg. Same rationale as
    color_match_to_scene_components — match the scene's other components, not
    the workspace surface. No-op if patch is already noisier than the target,
    or if there are no existing components in the bg.
    """
    if not scene_component_polys_px:
        return patch_rgb

    scene_mask = _build_scene_component_mask(bg.shape, scene_component_polys_px)
    if int(scene_mask.sum()) < 50:
        return patch_rgb

    bg_gray    = cv2.cvtColor(bg,        cv2.COLOR_BGR2GRAY)
    patch_gray = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)

    # Estimate noise level from the existing-component pixels only (not the
    # whole bg — the workspace surface has its own noise characteristics that
    # are not representative of the components' high-freq content).
    bg_comp_pixels = bg_gray[scene_mask > 0]
    if bg_comp_pixels.size < 50:
        return patch_rgb
    # Laplacian needs spatial structure; compute it on the full image then
    # sample only the component pixels.
    bg_lap = cv2.Laplacian(bg_gray, cv2.CV_64F)
    bg_noise = float(bg_lap[scene_mask > 0].std()) / 4.0

    patch_lap = cv2.Laplacian(patch_gray, cv2.CV_64F)
    patch_noise = float(patch_lap.std()) / 4.0

    if bg_noise <= patch_noise:
        return patch_rgb

    delta_std = float(np.sqrt(bg_noise ** 2 - patch_noise ** 2))
    noise = np.random.normal(0.0, delta_std, patch_rgb.shape)
    noisy = np.clip(patch_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    fg_3 = np.stack([patch_mask > 0] * 3, axis=-1)
    return np.where(fg_3, noisy, patch_rgb)


# ----------------------------------------------------------------------------
# Source-pool reference (for hard-negatives that have no per-image components)
# ----------------------------------------------------------------------------

def build_source_pool(source_index_path: Path, split_root: Path):
    """Pre-compute per-source-group color (Lab mean/std) + noise (Laplacian std)
    statistics from each group's component-bearing images. Returns:
        (source_pool, stem_to_source)
    where source_pool[group] = {"lab_mean":[3], "lab_std":[3], "lap_noise": float}.

    Used at augmentation time when the background image has no per-image
    components to color-match against -- e.g., hard-negative video frames.
    """
    si = json.loads(source_index_path.read_text())
    stem_to_source = si.get("stem_to_source", {})
    source_to_stems = si.get("source_to_component_stems", {})

    pool = {}
    for grp, stems in source_to_stems.items():
        all_lab_pixels = []
        lap_stds = []
        for stem in stems:
            # Image could be in any of train/val/test
            img_path = None
            lbl_path = None
            for split in ("train", "val", "test"):
                cand = split_root / "images" / split / f"{stem}.png"
                if cand.exists():
                    img_path = cand
                    lbl_path = split_root / "labels" / split / f"{stem}.txt"
                    break
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            instances = read_yolo_seg(lbl_path) if lbl_path.exists() else []
            mask = np.zeros((H, W), dtype=np.uint8)
            for _cls, poly in instances:
                poly_px = poly_norm_to_px(poly, H, W)
                cv2.fillPoly(mask, [poly_px], 1)
            if int(mask.sum()) < 50:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            all_lab_pixels.append(lab[mask > 0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_stds.append(float(lap[mask > 0].std()) / 4.0)

        if not all_lab_pixels:
            continue
        merged = np.concatenate(all_lab_pixels, axis=0)
        pool[grp] = {
            "lab_mean":  merged.mean(axis=0).tolist(),
            "lab_std":  (merged.std(axis=0) + 1e-6).tolist(),
            "lap_noise": float(np.mean(lap_stds)) if lap_stds else 0.0,
            "n_pixels":  int(merged.shape[0]),
            "n_stems":   len(stems),
        }
    return pool, stem_to_source


def color_match_with_pool_fallback(patch_rgb, patch_mask, bg,
                                    scene_component_polys_px, pool_stats):
    """Color-match the patch. Prefers per-image scene components; falls back to
    `pool_stats` (precomputed Lab mean/std for the bg's source group) when the
    bg has no labeled components -- e.g. video hard negatives.
    """
    if scene_component_polys_px:
        return color_match_to_scene_components(
            patch_rgb, patch_mask, bg, scene_component_polys_px
        )
    if pool_stats is None:
        return patch_rgb

    src_lab = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg_mask = patch_mask > 0
    if fg_mask.sum() < 10:
        return patch_rgb
    src_pixels = src_lab[fg_mask]
    src_mean = src_pixels.mean(axis=0)
    src_std  = src_pixels.std(axis=0) + 1e-6
    target_mean = np.array(pool_stats["lab_mean"], dtype=np.float32)
    target_std  = np.array(pool_stats["lab_std"],  dtype=np.float32)

    out_lab = src_lab.copy()
    out_lab[fg_mask] = (src_lab[fg_mask] - src_mean) / src_std * target_std + target_mean
    out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def noise_match_with_pool_fallback(patch_rgb, patch_mask, bg,
                                    scene_component_polys_px, pool_stats):
    """Noise-match the patch. Same fallback strategy as color_match."""
    if scene_component_polys_px:
        return noise_match_to_scene_components(
            patch_rgb, patch_mask, bg, scene_component_polys_px
        )
    if pool_stats is None:
        return patch_rgb

    bg_noise = float(pool_stats["lap_noise"])
    patch_gray = cv2.cvtColor(patch_rgb, cv2.COLOR_BGR2GRAY)
    patch_lap = cv2.Laplacian(patch_gray, cv2.CV_64F)
    patch_noise = float(patch_lap.std()) / 4.0
    if bg_noise <= patch_noise:
        return patch_rgb
    delta_std = float(np.sqrt(bg_noise ** 2 - patch_noise ** 2))
    noise = np.random.normal(0.0, delta_std, patch_rgb.shape)
    noisy = np.clip(patch_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    fg_3 = np.stack([patch_mask > 0] * 3, axis=-1)
    return np.where(fg_3, noisy, patch_rgb)


def poisson_blend(bg, patch_rgb, patch_mask, x0, y0):
    """Gradient-domain (Poisson) blend via cv2.seamlessClone. In-place on bg via
    np.copyto. Falls back to alpha_paste if seamlessClone fails (it raises on
    edge-touching or zero-area masks)."""
    h, w = patch_mask.shape
    center = (x0 + w // 2, y0 + h // 2)            # cv2 wants center in dst coords
    mask_u8 = (patch_mask > 0).astype(np.uint8) * 255

    # seamlessClone is unreliable for tiny masks; fall back to alpha.
    if int(mask_u8.sum()) < 100 * 255:    # < 100 fg pixels
        alpha_paste(bg, patch_rgb, patch_mask, x0, y0, feather_px=3)
        return
    try:
        out = cv2.seamlessClone(patch_rgb, bg, mask_u8, center, cv2.NORMAL_CLONE)
        np.copyto(bg, out)
    except cv2.error:
        # Common cause: mask touches image edge. Fall back to alpha.
        alpha_paste(bg, patch_rgb, patch_mask, x0, y0, feather_px=3)


# ----------------------------------------------------------------------------
# Augment one training image
# ----------------------------------------------------------------------------

def augment_one_image(
    bg_img_path: Path, bg_lbl_path: Path,
    out_img_path: Path, out_lbl_path: Path,
    mask_bank_by_comp: dict, name_to_id: dict,
    target_areas_by_cls: dict,
    n_paste_min: int, n_paste_max: int,
    scale_jitter: float, feather_px: int, hflip_prob: float,
    rotation_deg: float, perspective_strength: float,
    color_match: bool, noise_match: bool, blend_mode: str,
    paste_region_mask=None,        # optional binary mask; constrains paste centers
    pool_stats=None,                # optional source-group color/noise reference (fallback)
):
    """Returns log dict (or None if nothing pasted)."""
    bg = cv2.imread(str(bg_img_path))
    if bg is None:
        return None
    H, W = bg.shape[:2]

    existing = read_yolo_seg(bg_lbl_path)
    occupied = []
    # scene_component_polys_px is the in-pixel-coords list of EXISTING component
    # polygons. Used as the color/noise-match target so pasted patches are
    # adjusted to look like other components in this scene, not like the bg surface.
    scene_component_polys_px = []
    for cls, poly in existing:
        poly_px = poly_norm_to_px(poly, H, W)
        x, y, w_, h_ = cv2.boundingRect(poly_px)
        occupied.append((x, y, x + w_, y + h_))
        scene_component_polys_px.append(poly_px)

    new_instances = list(existing)
    paste_log = []
    n_target = random.randint(n_paste_min, n_paste_max)
    n_pasted = 0

    component_ids = sorted(mask_bank_by_comp.keys())
    random.shuffle(component_ids)
    # Take up to n_target component_ids cyclically (one paste per component_id chosen)
    for attempt_i in range(n_target * 2):   # double the loop budget; collisions can fail
        if n_pasted >= n_target:
            break
        comp_id = component_ids[attempt_i % len(component_ids)]
        cls_id = component_to_class(comp_id, name_to_id)
        if cls_id is None or cls_id not in target_areas_by_cls:
            continue

        rgb_p, mask_p = random.choice(mask_bank_by_comp[comp_id])
        ext = extract_component_patch(rgb_p, mask_p)
        if ext is None:
            continue
        rgb_patch, mask_patch = ext

        # Optional horizontal flip
        if random.random() < hflip_prob:
            rgb_patch = rgb_patch[:, ::-1].copy()
            mask_patch = mask_patch[:, ::-1].copy()

        # Image-plane rotation (the rotary table does NOT cover this — it spins
        # around the world vertical axis, not the camera optical axis).
        if rotation_deg > 0:
            angle = random.uniform(-rotation_deg, rotation_deg)
            rgb_patch, mask_patch = rotate_patch(rgb_patch, mask_patch, angle)

        # Mild perspective warp (the rotary table doesn't cover camera-tilt variation).
        if perspective_strength > 0:
            rgb_patch, mask_patch = perspective_warp_patch(
                rgb_patch, mask_patch, perspective_strength
            )

        # Re-crop tight to the mask -- rotation + perspective leave empty borders
        # that would skew the target-area calculation if not removed first.
        rgb_patch, mask_patch = tight_crop_to_mask(rgb_patch, mask_patch)
        if rgb_patch is None:
            continue

        # Resize with jitter
        target_med = target_areas_by_cls[cls_id]
        jitter = 1.0 + random.uniform(-scale_jitter, scale_jitter)
        target_area_px = max(50, int(target_med * jitter))
        rgb_r, mask_r = resize_patch(rgb_patch, mask_patch, target_area_px)
        if rgb_r is None:
            continue
        if rgb_r.shape[0] >= H or rgb_r.shape[1] >= W:
            continue   # patch larger than background, skip

        # Find collision-free location whose full patch silhouette (not just center)
        # lies inside the paste-region mask, if one was given.
        loc = find_paste_location(mask_r, W, H, occupied,
                                  paste_region_mask=paste_region_mask)
        if loc is None:
            continue
        x0, y0 = loc

        # Color- and noise-match the patch. Preferred target: this bg's EXISTING
        # components. Fallback (e.g. hard negatives that have no components):
        # the source-group's pool_stats (precomputed from same-source labeled frames).
        if color_match:
            rgb_r = color_match_with_pool_fallback(
                rgb_r, mask_r, bg, scene_component_polys_px, pool_stats
            )
        if noise_match:
            rgb_r = noise_match_with_pool_fallback(
                rgb_r, mask_r, bg, scene_component_polys_px, pool_stats
            )

        # Paste. Note: 'poisson' uses cv2.seamlessClone, which re-derives the
        # patch's absolute colors from the LOCAL BG boundary -- meaning Poisson
        # will partially undo the scene-component color matching we just did.
        # If you ran with color_match=True, prefer blend_mode='alpha' to preserve
        # the color match. 'poisson' is mainly useful when color_match is off.
        if blend_mode == "poisson":
            poisson_blend(bg, rgb_r, mask_r, x0, y0)
        else:
            alpha_paste(bg, rgb_r, mask_r, x0, y0, feather_px=feather_px)

        # Build polygon for the pasted instance (in full-bg coordinates)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y0:y0 + mask_r.shape[0], x0:x0 + mask_r.shape[1]] = mask_r
        poly = mask_to_largest_polygon(full_mask)
        if poly is None:
            continue
        poly_norm = poly_px_to_norm(poly, H, W)
        new_instances.append((cls_id, poly_norm))

        x, y, w_, h_ = cv2.boundingRect(poly)
        occupied.append((x, y, x + w_, y + h_))
        paste_log.append({
            "component_id": comp_id,
            "class_id":     cls_id,
            "source_rgb":   rgb_p.name,
            "paste_xy":     [int(x0), int(y0)],
            "paste_wh":     [int(rgb_r.shape[1]), int(rgb_r.shape[0])],
            "target_area_px": target_area_px,
        })
        n_pasted += 1

    if n_pasted == 0:
        return None  # nothing got pasted (rare; high collision)

    cv2.imwrite(str(out_img_path), bg)
    write_yolo_seg(out_lbl_path, new_instances)
    return {"n_pasted": n_pasted, "pastes": paste_log}


# ----------------------------------------------------------------------------
# Main: orchestrate over splits
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_bank", required=True,
                    help="SAM-2 mask bank directory (output of script 02).")
    ap.add_argument("--source_dataset", required=True,
                    help="Existing YOLO-seg dataset (has images/, labels/, dataset.yaml).")
    ap.add_argument("--out_dataset", required=True,
                    help="Output augmented dataset directory.")
    ap.add_argument("--augs_per_image", type=int, default=5,
                    help="Augmented variants per labeled training image.")
    ap.add_argument("--paste_count_min", type=int, default=2)
    ap.add_argument("--paste_count_max", type=int, default=5)
    ap.add_argument("--scale_jitter", type=float, default=0.20,
                    help="+/- relative scale jitter around median target (0.2 = +/-20%%).")
    ap.add_argument("--feather_px", type=int, default=1,
                    help="Edge feathering radius in pixels (alpha blend). Feather is "
                         "applied INWARD only (erode + blur + clamp to mask) so no patch "
                         "color bleeds outside the silhouette. Bump up for softer edges.")
    ap.add_argument("--hflip_prob", type=float, default=0.5,
                    help="Probability of horizontally flipping each pasted patch.")
    ap.add_argument("--rotation_deg", type=float, default=180.0,
                    help="Max +/- rotation in image plane applied to each pasted patch "
                         "(default: full 180, since components have no canonical orientation). "
                         "Set to 0 to disable. Rotary recordings DON'T cover this; the table "
                         "spins around the world vertical axis, not the camera optical axis.")
    ap.add_argument("--perspective_strength", type=float, default=0.05,
                    help="Mild perspective warp applied to each pasted patch. Max corner "
                         "displacement as fraction of patch dim (0.05 = +/-5%%). Set to 0 "
                         "to disable. Rotary recordings don't cover camera-tilt variation.")
    ap.add_argument("--color_match", action=argparse.BooleanOptionalAction, default=True,
                    help="Reinhard color transfer (Lab mean+std) from the patch foreground "
                         "to the EXISTING-COMPONENT pixels of the bg (NOT the surrounding "
                         "surface). Use --no-color_match to disable. No-op if the bg has no "
                         "existing labeled components.")
    ap.add_argument("--noise_match", action=argparse.BooleanOptionalAction, default=True,
                    help="Add Gaussian noise to the patch so its noise level approaches "
                         "the existing-component pixels' noise level (estimated via "
                         "Laplacian std). Use --no-noise_match to disable.")
    ap.add_argument("--blend_mode", choices=("alpha", "poisson"), default="alpha",
                    help="Compositing mode. Default 'alpha' (feathered alpha compositing) "
                         "preserves the patch's matched color. 'poisson' (cv2.seamlessClone) "
                         "re-derives the patch's interior colors from the local bg boundary, "
                         "which partially undoes color matching -- only use 'poisson' when "
                         "running with --no-color_match.")
    ap.add_argument("--paste_region_dir", type=str, default=None,
                    help="Optional. Path to a directory of per-image paste-region masks "
                         "(binary PNGs named <bg_stem>.png). When provided, paste centers "
                         "are constrained to lie inside the mask (prevents off-table pastes). "
                         "Default: auto-detect <source_dataset>/paste_regions/ if it exists.")
    ap.add_argument("--source_index", type=str, default=None,
                    help="Optional. Path to source_index.json from script 04. Enables "
                         "per-source-group color/noise reference fallback for hard-negative "
                         "backgrounds (which have no per-image components to match against). "
                         "Default: auto-detect <source_dataset>/source_index.json if it exists.")
    ap.add_argument("--augment_negatives", action=argparse.BooleanOptionalAction, default=True,
                    help="Also augment hard-negative training images (paste components onto "
                         "empty scenes). Default on. Uses --source_index for the color/noise "
                         "target since hard negatives have no scene components.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    mask_bank = _resolve_path(args.mask_bank)
    src_ds    = _resolve_path(args.source_dataset)
    out_ds    = _resolve_path(args.out_dataset)

    # ---- Load source dataset.yaml for class names ---------------------------
    src_yaml = src_ds / "dataset.yaml"
    assert src_yaml.exists(), f"Missing source dataset.yaml: {src_yaml}"
    cfg = yaml.safe_load(open(src_yaml))
    id_to_name = {int(k): v for k, v in cfg["names"].items()}
    name_to_id = {v: k for k, v in id_to_name.items()}
    print(f"[INFO] Source dataset.yaml classes: {id_to_name}")

    # ---- Discover the mask bank ---------------------------------------------
    bank_by_comp = discover_mask_bank(mask_bank)
    if not bank_by_comp:
        raise SystemExit(f"Mask bank is empty: {mask_bank}")
    print(f"[INFO] Mask bank components ({len(bank_by_comp)}):")
    for comp, pairs in sorted(bank_by_comp.items()):
        cls_id = component_to_class(comp, name_to_id)
        cls_label = id_to_name.get(cls_id, "UNKNOWN") if cls_id is not None else "UNKNOWN"
        print(f"  - {comp:<14s}  class={cls_label:<6s}  {len(pairs)} frame+mask pairs")

    # ---- Auto-calibrate per-class target paste sizes from source training labels
    # Use the first image to fetch (W,H) — assume all training images share resolution.
    train_imgs = sorted([p for p in (src_ds / "images" / "train").rglob("*")
                         if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    assert train_imgs, f"No training images in {src_ds / 'images' / 'train'}"
    probe = cv2.imread(str(train_imgs[0]))
    H_src, W_src = probe.shape[:2]
    print(f"[INFO] Source training image resolution: {W_src}x{H_src}")

    # ---- Load paste-region masks dir + source-index pool (optional) -------
    paste_region_dir = (
        _resolve_path(args.paste_region_dir)
        if args.paste_region_dir
        else (src_ds / "paste_regions" if (src_ds / "paste_regions").exists() else None)
    )
    if paste_region_dir and paste_region_dir.exists():
        n_pr = sum(1 for _ in paste_region_dir.glob("*.png"))
        print(f"[INFO] Paste-region masks: {n_pr} files in {paste_region_dir}")
    else:
        paste_region_dir = None
        print("[INFO] No paste_region_dir given/found; pastes will not be region-constrained.")

    source_index_path = (
        _resolve_path(args.source_index)
        if args.source_index
        else (src_ds / "source_index.json" if (src_ds / "source_index.json").exists() else None)
    )
    source_pool, stem_to_source = {}, {}
    if source_index_path and source_index_path.exists():
        source_pool, stem_to_source = build_source_pool(source_index_path, src_ds)
        print(f"[INFO] Source-pool reference loaded from {source_index_path.name}:")
        for grp, st in sorted(source_pool.items()):
            print(f"  {grp:36s}  n_stems={st['n_stems']:<2d}  n_pixels={st['n_pixels']:>7d}  "
                  f"lap_noise={st['lap_noise']:.2f}")
    else:
        print("[INFO] No source_index.json given/found; color/noise fallback disabled.")

    print("[INFO] Calibrating per-class paste sizes from source labels...")
    target_stats = calibrate_paste_sizes(src_ds / "labels" / "train", W_src, H_src)
    target_areas = {}
    for cls_id, stats in sorted(target_stats.items()):
        target_areas[cls_id] = stats["median_area_px"]
        print(f"  class {cls_id} ({id_to_name.get(cls_id)}): "
              f"median={stats['median_area_px']}, "
              f"p05={stats['p05_area_px']}, p95={stats['p95_area_px']} "
              f"(n={stats['n_samples']} instances)")

    # ---- Set up output dir ---------------------------------------------------
    if out_ds.exists():
        raise SystemExit(f"Output dataset already exists: {out_ds}. Delete it manually first.")
    for split in ("train", "val", "test"):
        (out_ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_ds / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ---- Copy val/test verbatim ---------------------------------------------
    for split in ("val", "test"):
        src_img_dir = src_ds / "images" / split
        src_lbl_dir = src_ds / "labels" / split
        if not src_img_dir.exists():
            continue
        for ip in src_img_dir.iterdir():
            shutil.copy2(ip, out_ds / "images" / split / ip.name)
        for lp in src_lbl_dir.iterdir():
            shutil.copy2(lp, out_ds / "labels" / split / lp.name)
        print(f"[OK] Copied {split} unchanged.")

    # ---- Copy train originals + augment labeled ones ------------------------
    log = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_paths_note": "All paths below are RELATIVE to this file's directory (out_dataset).",
        "_absolute_at_write_time": {
            "mask_bank":      str(mask_bank),
            "source_dataset": str(src_ds),
            "out_dataset":    str(out_ds),
        },
        "args": {k: v for k, v in vars(args).items() if k not in ("mask_bank", "source_dataset", "out_dataset")},
        "target_areas_per_class": {str(k): v for k, v in target_stats.items()},
        "augmented_images": [],
        "summary": {},
    }

    n_train_total = 0
    n_train_with_labels = 0
    n_aug_attempts = 0
    n_aug_succeeded = 0

    for ip in sorted((src_ds / "images" / "train").iterdir()):
        if ip.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        n_train_total += 1

        lp_src = src_ds / "labels" / "train" / f"{ip.stem}.txt"
        # Always copy the original through
        shutil.copy2(ip, out_ds / "images" / "train" / ip.name)
        if lp_src.exists():
            shutil.copy2(lp_src, out_ds / "labels" / "train" / lp_src.name)
        else:
            # Hard negative — copy as empty label so YOLO treats it as background
            (out_ds / "labels" / "train" / f"{ip.stem}.txt").write_text("")

        existing_labels = read_yolo_seg(lp_src)
        if existing_labels:
            n_train_with_labels += 1
        else:
            # Hard negative. Skip augmentation only if user disabled it.
            if not args.augment_negatives:
                continue

        # Look up this image's paste-region mask + source-group color/noise pool
        paste_region_mask = None
        if paste_region_dir is not None:
            pr_path = paste_region_dir / f"{ip.stem}.png"
            if pr_path.exists():
                paste_region_mask = cv2.imread(str(pr_path), cv2.IMREAD_GRAYSCALE)
        bg_source = stem_to_source.get(ip.stem)
        pool_stats = source_pool.get(bg_source) if bg_source else None

        # Augment K variants
        for i in range(args.augs_per_image):
            n_aug_attempts += 1
            aug_stem = f"{ip.stem}_aug{i}"
            aug_ip = out_ds / "images" / "train" / f"{aug_stem}.png"
            aug_lp = out_ds / "labels" / "train" / f"{aug_stem}.txt"
            result = augment_one_image(
                ip, lp_src, aug_ip, aug_lp,
                bank_by_comp, name_to_id, target_areas,
                args.paste_count_min, args.paste_count_max,
                args.scale_jitter, args.feather_px, args.hflip_prob,
                args.rotation_deg, args.perspective_strength,
                args.color_match, args.noise_match, args.blend_mode,
                paste_region_mask=paste_region_mask,
                pool_stats=pool_stats,
            )
            if result is not None:
                n_aug_succeeded += 1
                log["augmented_images"].append({
                    "augmented_image": str((Path("images") / "train" / aug_ip.name)),
                    "augmented_label": str((Path("labels") / "train" / aug_lp.name)),
                    "source_image":    str((Path("images") / "train" / ip.name)),
                    "n_pasted":        result["n_pasted"],
                    "pastes":          result["pastes"],
                })

    log["summary"] = {
        "train_images_total":           n_train_total,
        "train_images_with_labels":     n_train_with_labels,
        "train_images_hard_negatives":  n_train_total - n_train_with_labels,
        "augmentation_attempts":        n_aug_attempts,
        "augmented_images_written":     n_aug_succeeded,
        "final_train_image_count":      n_train_total + n_aug_succeeded,
    }

    # ---- Write dataset.yaml (relative paths, train/val/test all required) --
    abs_root = out_ds.resolve()
    names_yaml = "\n".join(f"  {k}: {v}" for k, v in sorted(id_to_name.items()))
    yaml_text = (
        "# All paths are RELATIVE to the directory containing this file.\n"
        f"# Absolute root at write time: {abs_root}\n"
        "# Augmented training dataset (SAM-2 copy-paste). val/test are unchanged from source.\n"
        "train: images/train\n"
        "val:   images/val\n"
        "test:  images/test\n"
        f"names:\n{names_yaml}\n"
    )
    (out_ds / "dataset.yaml").write_text(yaml_text)

    # ---- Write augmentation_log.json ----------------------------------------
    (out_ds / "augmentation_log.json").write_text(json.dumps(log, indent=2))

    print()
    print("=" * 70)
    print(f"[OK] Augmented dataset built at: {out_ds}")
    print("=" * 70)
    for k, v in log["summary"].items():
        print(f"  {k:<32s}: {v}")
    print(f"\n  dataset.yaml:           {out_ds / 'dataset.yaml'}")
    print(f"  augmentation_log.json:  {out_ds / 'augmentation_log.json'}")


if __name__ == "__main__":
    main()
