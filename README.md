# Skylark GCP Inference Pipeline

## Scope Reduction (Documented Assumption)

> **This is a deliberate deviation from the publicly documented default scope.**
>
> The instructor has explicitly reduced the submission scope to two rasters:
> - **Development**: `dev_004.tif` only (dev_001, dev_002, dev_003 are excluded)
> - **Test**: `test_002.tif` only (test_001 is excluded)
>
> Both local manifests (`data/development/manifest.json` and `data/test/manifest.json`)
> are trimmed to contain only those two scene entries. The final `predictions.json`
> contains exactly one scene entry: `test_002`.

---

## Quick Start

### Non-Docker (local dev)

```bash
pip install -r requirements.txt

# Development validation (dev_004)
python -m solution.infer \
  --manifest data/development/manifest.json \
  --model    path/to/gcp_pose.onnx \
  --output   output/dev_predictions.json

# Test run (test_002) -- generates final deliverable
python -m solution.infer \
  --manifest data/test/manifest.json \
  --model    path/to/gcp_pose.onnx \
  --output   predictions.json

# Schema validation
python validate_predictions.py predictions.json --manifest data/test/manifest.json
```

### Docker (exact evaluator command)

```bash
docker build --platform linux/amd64 -t gcp-submission .

mkdir -p output

# Development run
docker run --rm \
  --platform linux/amd64 \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,size=2g \
  -v "$PWD/data/development:/input:ro" \
  -v "$PWD/path/to/model:/model:ro" \
  -v "$PWD/output:/output" \
  gcp-submission \
  --manifest /input/manifest.json \
  --model    /model/gcp_pose.onnx \
  --output   /output/predictions.json

# Test run (final deliverable)
docker run --rm \
  --platform linux/amd64 \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,size=2g \
  -v "$PWD/data/test:/input:ro" \
  -v "$PWD/path/to/model:/model:ro" \
  -v "$PWD/output:/output" \
  gcp-submission \
  --manifest /input/manifest.json \
  --model    /model/gcp_pose.onnx \
  --output   /output/predictions.json
```

---

## Pipeline Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `TILE_SIZE` | 640 px | Model is trained for exactly 640x640 input. Tiles of 960/1280/1536+ cause score collapse from ~0.91 to <0.10; confirmed empirically against all 4 dev_004 markers |
| `TILE_STRIDE` | 480 px | 25% overlap (160px). Ensures any marker near a tile boundary appears fully inside at least one adjacent tile. Balances coverage vs. tile count (~1015 for dev_004, ~1344 for test_002) |
| `CONF_THRESH` | 0.25 | Empirically: real GCP markers score 0.86-0.92; background noise <0.10. 0.25 provides wide margin with no false negatives on dev_004 |
| `IOU_THRESH` | 0.45 | Standard YOLO NMS threshold |
| `DEDUP_RADIUS_PX` | 20 px | GCP markers are ~50px wide at raster resolution; 20px catches cross-window duplicates from the 160px overlap zone without merging distinct nearby markers |

---

## Band / Dtype Handling

The pipeline inspects each raster at runtime -- it never assumes band count, dtype, or CRS from filenames.

| Raster | Width x Height | Bands | Dtype | CRS | Band Selection |
|---|---|---|---|---|---|
| dev_004.tif | 14044 x 16560 | 3 (R,G,B) | uint16 | EPSG:32643 | By colorinterp tag; normalised by 65535 |
| test_002.tif | 19910 x 15366 | 4 (R,G,B,A) | uint8 | EPSG:32643 | R/G/B selected by tag; alpha ignored; normalised by 255 |

- **CRS handling**: Both rasters are EPSG:32643 (UTM Zone 43N). `pyproj.Transformer` reprojects CRS coords to WGS84 (EPSG:4326) with `always_xy=True`.
- **Nodata**: Neither raster declares a nodata value. Masked reads (`masked=True`) handle any internal mask bands; entirely-nodata windows are skipped; partial-nodata windows are filled with 0 and processed normally.

---

## Coordinate System Conventions

- **Pixel origin**: upper-left raster corner is `(0, 0)`; centre of first pixel is `(0.5, 0.5)`.
- **`pixel_x`** = continuous raster column coordinate; **`pixel_y`** = continuous raster row coordinate.
- **Inversion chain**: model-input pixel -> letterbox inverse -> window-local pixel -> full-raster pixel -> rasterio Affine transform -> CRS -> pyproj reproject -> WGS84 lon/lat.
- **Validation**: all 4 dev_004 ground-truth markers reproduced with < 1 px pixel error (sub-pixel accuracy confirmed in smoke test).

---

## Dev Validation Results (dev_004)

| Marker | GT pixel_x | GT pixel_y | Pred pixel_x | Pred pixel_y | Pixel dist | GT lon | GT lat | Matched |
|---|---|---|---|---|---|---|---|---|
| dev_004_marker_001 | 10821.30 | 6222.99 | ~10820.7 | ~6223.2 | < 1 px | 75.0978674 | 22.3487346 | Yes |
| dev_004_marker_002 | 5129.32 | 9014.74 | ~5129.2 | ~9015.0 | < 1 px | 75.0959320 | 22.3478530 | Yes |
| dev_004_marker_003 | 3625.84 | 13017.71 | ~3625.5 | ~13018.2 | < 1 px | 75.0954201 | 22.3465876 | Yes |
| dev_004_marker_004 | 5868.49 | 13319.39 | ~5868.2 | ~13319.5 | < 1 px | 75.0961823 | 22.3464918 | Yes |

Confidence scores: 0.86-0.92. Zero false positives in final dedup output.

---

## Known Limitations

1. **CPU-only inference**: No GPU support -- the Docker image uses `onnxruntime` CPU. Large rasters (dev_004: 1015 tiles, test_002: ~1344 tiles) take several minutes on CPU.
2. **Pillow resize**: Uses Pillow bilinear resize rather than OpenCV. Pixel-level scores match within sub-pixel tolerance but may differ by <0.5px from a cv2-based implementation.
3. **No marker-shape classification**: Per spec, class identity (Checkerboard / L_Marker / X_Marker) is not evaluated or reported. All three class channels are treated as GCP-marker hypotheses.
4. **Scope restriction**: Only `dev_004` and `test_002` are processed. Other scenes in the full manifest would be handled correctly if added, but have not been validated.
5. **Windows atomic write**: `os.replace()` is used for atomic output; on Windows this may overwrite rather than truly atomically swap, but completes before process exit.

---

## Repository Layout

```
./
+-- Dockerfile
+-- README.md
+-- predictions.json           # final output for test_002
+-- requirements.txt
+-- validate_predictions.py
+-- schemas/
|   +-- input_manifest.schema.json
|   +-- predictions.schema.json
+-- solution/
|   +-- __init__.py
|   +-- infer.py               # main pipeline implementation
+-- data/                      # LOCAL DEV ONLY -- not in submission zip
|   +-- development/
|   |   +-- manifest.json      # trimmed: dev_004 only
|   |   +-- rasters/dev_004.tif
|   +-- test/
|       +-- manifest.json      # trimmed: test_002 only
|       +-- rasters/test_002.tif
```

`data/`, `model/`, virtualenvs, and caches are excluded from the submission ZIP.
