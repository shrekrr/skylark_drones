"""
validate_predictions.py
=======================
Validates a predictions.json file against:
  1. The JSON schema  (schemas/predictions.schema.json)
  2. The manifest     (--manifest path)
     • Every scene in the manifest must appear exactly once in predictions.json.
     • No scene in predictions.json may be absent from the manifest.

Usage:
    python validate_predictions.py predictions.json --manifest data/test/manifest.json

Exit code 0 = pass, non-zero = fail.
"""

import argparse
import json
import sys
from pathlib import Path

import jsonschema


SCHEMA_PATH = Path(__file__).parent / "schemas" / "predictions.schema.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate predictions.json")
    parser.add_argument("predictions", help="Path to predictions.json")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    args = parser.parse_args()

    errors: list[str] = []

    # ── Load files ────────────────────────────────────────────────────────────
    try:
        with open(args.predictions) as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"FAIL: cannot load predictions file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.manifest) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"FAIL: cannot load manifest file: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
    except Exception as e:
        print(f"FAIL: cannot load schema: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Schema validation ─────────────────────────────────────────────────────
    try:
        jsonschema.validate(instance=predictions, schema=schema)
        print("PASS: JSON schema validation")
    except jsonschema.ValidationError as e:
        errors.append(f"Schema error: {e.message}")
        print(f"FAIL: JSON schema validation — {e.message}", file=sys.stderr)

    # ── Manifest cross-check ──────────────────────────────────────────────────
    manifest_ids: set[str] = {s["scene_id"] for s in manifest.get("scenes", [])}
    pred_ids: list[str] = [s["scene_id"] for s in predictions.get("scenes", [])]

    # Duplicates in predictions
    if len(pred_ids) != len(set(pred_ids)):
        errors.append("Duplicate scene_ids in predictions.json")

    pred_id_set = set(pred_ids)

    # Missing scenes
    missing = manifest_ids - pred_id_set
    if missing:
        errors.append(f"Missing scenes from manifest: {sorted(missing)}")

    # Extra scenes
    extra = pred_id_set - manifest_ids
    if extra:
        errors.append(f"Unknown scenes not in manifest: {sorted(extra)}")

    if not missing and not extra:
        print("PASS: manifest scene cross-check")
    else:
        for e in [f"Missing: {sorted(missing)}", f"Extra: {sorted(extra)}"]:
            print(f"FAIL: {e}", file=sys.stderr)

    # ── Coordinate range checks (belt-and-suspenders beyond schema) ───────────
    coord_errors = 0
    for scene in predictions.get("scenes", []):
        for det in scene.get("detections", []):
            if not (-180 <= det.get("longitude", 0) <= 180):
                coord_errors += 1
            if not (-90 <= det.get("latitude", 0) <= 90):
                coord_errors += 1
            conf = det.get("confidence", -1)
            if not (0 <= conf <= 1 and conf == conf):  # also catches NaN
                coord_errors += 1

    if coord_errors == 0:
        print("PASS: coordinate range checks")
    else:
        errors.append(f"{coord_errors} coordinate range violation(s)")
        print(f"FAIL: {coord_errors} coordinate range violation(s)", file=sys.stderr)

    # ── Summary ───────────────────────────────────────────────────────────────
    if errors:
        print(f"\n{'='*50}")
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)
    else:
        total_dets = sum(len(s["detections"]) for s in predictions["scenes"])
        print(f"\nVALIDATION PASSED — {len(pred_ids)} scene(s), {total_dets} total detection(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
