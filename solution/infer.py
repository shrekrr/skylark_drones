"""
solution/infer.py -- Skylark GCP Inference Pipeline
=====================================================
Processes an ONNX pose-estimation model over georeferenced drone orthomosaics to
detect Ground Control Point (GCP) marker centres in raster pixel coords and
WGS84 lon/lat.

Design decisions (documented per PRD section 12):
  TILE_SIZE      = 640   px  -- matches model native 640x640 training resolution
                                exactly; larger tiles cause score collapse to <0.1
  TILE_STRIDE    = 480   px  -- 25% overlap (160 px) ensures any marker is fully
                                visible in at least one tile
  CONF_THRESH    = 0.25       -- tunable; empirically markers score 0.85-0.92,
                                noise < 0.10; 0.25 gives wide safety margin
  IOU_THRESH     = 0.45       -- standard YOLO NMS threshold
  DEDUP_RADIUS   = 20    px   -- GCP markers ~50px wide; 20px catches cross-window
                                duplicates from the 160px overlap zone

Scope (per instructor reduction): dev_004 (dev) and test_002 (test) only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window
import pyproj

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
TILE_SIZE       = 640    # Must match model native 640x640 resolution
TILE_STRIDE     = 480    # 25% overlap = 160px; ensures full coverage at edges
CONF_THRESH     = 0.25   # Confidence threshold; real markers score ~0.85-0.92
IOU_THRESH      = 0.45   # NMS IoU threshold (standard YOLO setting)
DEDUP_RADIUS_PX = 20     # Cross-window dedup radius in full-raster pixels
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Letterbox -- exact Ultralytics rounding rule
# ---------------------------------------------------------------------------

def letterbox(
    img_hwc: np.ndarray,
    target: int = 640,
    fill: float = 114.0 / 255.0,
) -> tuple[np.ndarray, float, int, int]:
    """
    Center-pad img_hwc (H x W x 3 float32 [0,1]) to target x target.

    Uses the Ultralytics asymmetric +/-0.1 rounding rule for left/top so the
    inverse transformation is exact.

    Returns:
        padded  -- target x target x 3 float32
        scale   -- resize scale factor
        left    -- left padding in pixels (needed to invert x)
        top_pad -- top  padding in pixels (needed to invert y)
    """
    h, w = img_hwc.shape[:2]
    scale = min(target / w, target / h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    # Bilinear resize -- prefer cv2 (faster), fall back to Pillow
    try:
        import cv2 as _cv2
        resized = _cv2.resize(img_hwc, (new_w, new_h), interpolation=_cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image as _Image
        pil = _Image.fromarray((img_hwc * 255).clip(0, 255).astype(np.uint8))
        pil = pil.resize((new_w, new_h), _Image.BILINEAR)
        resized = np.array(pil, dtype=np.float32) / 255.0

    dw = (target - new_w) / 2
    dh = (target - new_h) / 2
    left    = round(dw - 0.1)
    top_pad = round(dh - 0.1)

    padded = np.full((target, target, 3), fill, dtype=np.float32)
    padded[top_pad: top_pad + new_h, left: left + new_w] = resized
    return padded, scale, left, top_pad


# ---------------------------------------------------------------------------
# NMS helpers
# ---------------------------------------------------------------------------

def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """boxes: (N,4) center_x center_y w h  ->  x1 y1 x2 y2"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _iou_one_vs_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of a single box (4,) against (N,4) boxes."""
    xi1 = np.maximum(box[0], boxes[:, 0])
    yi1 = np.maximum(box[1], boxes[:, 1])
    xi2 = np.minimum(box[2], boxes[:, 2])
    yi2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xi2 - xi1) * np.maximum(0.0, yi2 - yi1)
    a_box   = (box[2] - box[0]) * (box[3] - box[1])
    a_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = a_box + a_boxes - inter
    return inter / np.where(union > 0, union, 1e-9)


def greedy_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = IOU_THRESH,
) -> list[int]:
    """Standard greedy NMS. Returns indices of kept boxes (score-descending)."""
    order = np.argsort(scores)[::-1]
    suppressed = np.zeros(len(order), dtype=bool)
    keep: list[int] = []
    for i, idx in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(int(idx))
        rest = order[i + 1:]
        if len(rest) == 0:
            break
        ious = _iou_one_vs_many(boxes_xyxy[idx], boxes_xyxy[rest])
        suppressed[i + 1:] |= ious >= iou_thresh
    return keep


# ---------------------------------------------------------------------------
# Band / dtype adaptation (F2)
# ---------------------------------------------------------------------------

def _read_window_as_rgb_float(
    ds: rasterio.DatasetReader,
    window: Window,
) -> np.ndarray | None:
    """
    Read a windowed region and return H x W x 3 float32 in [0,1].

    Band selection:
      - If >=3 bands with R/G/B colorinterp tags, select by tag (not position).
      - If <3 bands (or tags missing), replicate available bands to fill 3 channels.

    Normalisation:
      - Integer dtypes: divide by dtype max (e.g. 65535 for uint16, 255 for uint8).
      - Float dtypes: clip to [0,1].

    Nodata:
      - Returns None if every pixel in every selected band is masked (entirely nodata).
      - Partial nodata windows are processed normally.
    """
    # Determine band indices (1-based) to read
    rgb_indices: list[int] = []
    if ds.count >= 3:
        band_map: dict[ColorInterp, int] = {}
        for band_idx in range(1, ds.count + 1):
            ci = ds.colorinterp[band_idx - 1]
            band_map[ci] = band_idx
        r_ci, g_ci, b_ci = ColorInterp.red, ColorInterp.green, ColorInterp.blue
        if r_ci in band_map and g_ci in band_map and b_ci in band_map:
            rgb_indices = [band_map[r_ci], band_map[g_ci], band_map[b_ci]]

    if not rgb_indices:
        # Fallback: first min(3, count) bands; replicate below if <3
        n = min(3, ds.count)
        rgb_indices = list(range(1, n + 1))

    raw = ds.read(rgb_indices, window=window, masked=True)  # (C, H, W) masked

    # Entirely nodata -- safe to skip
    if raw.mask.all():
        return None

    # Dtype max for normalisation
    dtype = np.dtype(ds.dtypes[rgb_indices[0] - 1])
    if np.issubdtype(dtype, np.integer):
        dtype_max = float(np.iinfo(dtype).max)
    else:
        dtype_max = 1.0

    # Fill masked values with 0 (grey padding will cover invalid pixels anyway)
    data = np.ma.filled(raw, 0).astype(np.float32)

    # Normalise to [0,1]
    data = data / dtype_max
    data = data.clip(0.0, 1.0)

    # Replicate to 3 channels if needed
    if len(rgb_indices) == 1:
        data = np.repeat(data, 3, axis=0)
    elif len(rgb_indices) == 2:
        data = np.concatenate([data, data[[0]]], axis=0)

    return data.transpose(1, 2, 0)  # (H, W, 3)


# ---------------------------------------------------------------------------
# Coordinate inversion (F5)
# ---------------------------------------------------------------------------

def _model_to_raster(
    kx: float, ky: float,
    scale: float, left: int, top: int,
    col_off: int, row_off: int,
) -> tuple[float, float]:
    """
    Invert model-input-pixel keypoint (kx, ky) to full-raster pixel (fx, fy).

    Chain:
      model-input pixel
        -> window-local pixel:  wx = (kx - left) / scale
                                wy = (ky - top)  / scale
        -> full-raster pixel:   fx = wx + col_off
                                fy = wy + row_off
    """
    wx = (kx - left) / scale
    wy = (ky - top)  / scale
    fx = wx + col_off
    fy = wy + row_off
    return fx, fy


def _raster_to_geographic(
    fx: float, fy: float,
    transform: rasterio.Affine,
    crs: rasterio.CRS,
) -> tuple[float, float]:
    """
    Convert full-raster continuous pixel (fx, fy) to WGS84 (lon, lat).

    rasterio Affine * (col, row) gives CRS coords. The continuous (col, row)
    convention with upper-left origin matches the (0.5, 0.5) pixel-centre
    convention -- no extra offset needed.
    """
    cx, cy = transform * (fx, fy)

    epsg = crs.to_epsg()
    if epsg == 4326:
        return float(cx), float(cy)

    transformer = pyproj.Transformer.from_crs(
        epsg,
        4326,
        always_xy=True,
    )
    lon, lat = transformer.transform(cx, cy)
    return float(lon), float(lat)


# ---------------------------------------------------------------------------
# Tile iterator (F1, F3)
# ---------------------------------------------------------------------------

def _tile_windows(
    width: int, height: int,
    tile: int = TILE_SIZE,
    stride: int = TILE_STRIDE,
) -> list[tuple[int, int, int, int]]:
    """
    Generate (col_off, row_off, win_w, win_h) covering the full raster.

    Partial edge tiles are included (not dropped), satisfying F1.
    """
    windows = []
    row = 0
    while row < height:
        col = 0
        while col < width:
            win_w = min(tile, width  - col)
            win_h = min(tile, height - row)
            windows.append((col, row, win_w, win_h))
            if col + tile >= width:
                break
            col += stride
        if row + tile >= height:
            break
        row += stride
    return windows


# ---------------------------------------------------------------------------
# Per-scene inference
# ---------------------------------------------------------------------------

def _run_scene(
    scene_id: str,
    raster_path: str,
    session: ort.InferenceSession,
) -> list[dict[str, Any]]:
    """
    Run inference on one scene. Returns list of detection dicts.
    """
    # Each candidate: (full_raster_x, full_raster_y, lon, lat, score)
    all_candidates: list[tuple[float, float, float, float, float]] = []

    with rasterio.open(raster_path) as ds:
        transform = ds.transform
        crs = ds.crs
        width, height = ds.width, ds.height

        windows = _tile_windows(width, height, TILE_SIZE, TILE_STRIDE)
        n_windows = len(windows)
        print(f"  [{scene_id}] {width}x{height} px -- {n_windows} tiles", flush=True)

        for win_i, (col_off, row_off, win_w, win_h) in enumerate(windows):
            if (win_i + 1) % 100 == 0:
                print(f"    tile {win_i + 1}/{n_windows} ...", flush=True)

            window = Window(col_off, row_off, win_w, win_h)

            # Read and convert to RGB float32 [0,1]
            img_hwc = _read_window_as_rgb_float(ds, window)
            if img_hwc is None:
                continue  # entirely nodata

            # Letterbox to 640x640
            padded, scale, left, top_pad = letterbox(img_hwc, target=TILE_SIZE)
            tensor = padded.transpose(2, 0, 1)[np.newaxis]  # (1,3,640,640) NCHW

            # ONNX inference
            raw_out = session.run(None, {"images": tensor})[0]  # (1,9,8400)
            pred = raw_out[0].T  # (8400, 9)

            # Candidate filtering (F4)
            # scores = max of class channels 4:7; no objectness channel exists
            scores_all = pred[:, 4:7].max(axis=1)
            keep_mask = scores_all >= CONF_THRESH
            if keep_mask.sum() == 0:
                continue

            pred_filt   = pred[keep_mask]
            scores_filt = scores_all[keep_mask]
            boxes_xyxy  = _xywh_to_xyxy(pred_filt[:, 0:4])

            # Within-window NMS (F4)
            keep_idx = greedy_nms(boxes_xyxy, scores_filt, IOU_THRESH)

            # Coordinate inversion for surviving candidates (F5)
            for idx in keep_idx:
                kx, ky = float(pred_filt[idx, 7]), float(pred_filt[idx, 8])
                score  = float(scores_filt[idx])

                fx, fy = _model_to_raster(
                    kx, ky, scale, left, top_pad, col_off, row_off
                )
                lon, lat = _raster_to_geographic(fx, fy, transform, crs)

                all_candidates.append((fx, fy, lon, lat, score))

    # Cross-window deduplication (F6)
    if not all_candidates:
        return []

    # Sort by score descending
    all_candidates.sort(key=lambda c: c[4], reverse=True)

    clusters: list[tuple[float, float, float, float, float]] = []
    for candidate in all_candidates:
        fx, fy = candidate[0], candidate[1]
        too_close = False
        for rep in clusters:
            dist = math.hypot(fx - rep[0], fy - rep[1])
            if dist <= DEDUP_RADIUS_PX:
                too_close = True
                break
        if not too_close:
            clusters.append(candidate)

    # Build output list (F7)
    detections: list[dict[str, Any]] = []
    for fx, fy, lon, lat, score in clusters:
        detections.append({
            "pixel_x":    round(float(fx),    6),
            "pixel_y":    round(float(fy),    6),
            "longitude":  round(float(lon),  10),
            "latitude":   round(float(lat),  10),
            "confidence": round(float(score),  6),
        })

    return detections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skylark GCP Inference Pipeline -- detects GCP markers in drone orthomosaics."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--model",    required=True, help="Path to gcp_pose.onnx")
    parser.add_argument("--output",   required=True, help="Path for output predictions.json")
    args = parser.parse_args()

    # Load manifest
    manifest_path = Path(args.manifest)
    with manifest_path.open() as f:
        manifest = json.load(f)

    manifest_dir = manifest_path.parent

    # Load ONNX session
    print(f"Loading model: {args.model}", flush=True)
    session = ort.InferenceSession(args.model)
    print("Model loaded.", flush=True)

    # Process each scene
    output_scenes: list[dict[str, Any]] = []

    for scene in manifest["scenes"]:
        scene_id    = scene["scene_id"]
        raster_rel  = scene["raster_path"]
        raster_path = str(manifest_dir / raster_rel)

        print(f"\nProcessing scene: {scene_id} ({raster_path})", flush=True)

        if not os.path.exists(raster_path):
            print(f"  ERROR: raster not found at {raster_path}", file=sys.stderr)
            sys.exit(1)

        detections = _run_scene(scene_id, raster_path, session)
        print(f"  [{scene_id}] -> {len(detections)} detections", flush=True)

        output_scenes.append({
            "scene_id":   scene_id,
            "detections": detections,
        })

    # Write output atomically (F7)
    output_data = {
        "schema_version": "1.0",
        "scenes": output_scenes,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory, then os.replace (atomic on POSIX;
    # overwrites on Windows)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=str(output_path.parent),
        suffix=".tmp",
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(output_data, f, indent=2)
        os.replace(tmp_name, str(output_path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

    print(f"\nOutput written to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
