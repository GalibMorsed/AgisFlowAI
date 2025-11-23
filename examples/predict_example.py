"""Example script: load a trained forecast model and print predictions for cells.

Usage:
  python examples/predict_example.py --csv sampel_train.csv --model models/forecast_conv1d.pt --cells 0,1,2,3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pathlib
from typing import List, Dict, Any

# When running this script directly, ensure the repository root is on sys.path
# so `src` can be imported. Examples are located in `examples/`, so add the
# parent directory of the examples folder to sys.path.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_meta_if_exists(csv_path: str):
    meta_path = csv_path + ".meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def parse_cells(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out


def get_all_cell_indices(csv_path: str) -> List[int]:
    """Scan CSV and return a sorted list of unique cell indices."""
    import csv

    cell_indices = set()
    try:
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "cell_idx" in row:
                    cell_indices.add(int(row["cell_idx"]))
    except (IOError, ValueError, KeyError):
        pass
    return sorted(list(cell_indices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Training CSV produced by the pipeline")
    parser.add_argument("--model", required=True, help="Path to saved model (.pt)")
    parser.add_argument("--cells", help="Comma-separated list of cell indices to predict for")
    parser.add_argument("--all-cells", action="store_true", help="Predict for all cells found in the CSV")
    parser.add_argument("--seq-len", type=int, default=16, help="Input sequence length (frames)")
    parser.add_argument("--fps", type=float, default=None, help="FPS override (if omitted, read from CSV meta)")
    parser.add_argument("--output", help="Optional path to save predictions (CSV or JSON format)")
    args = parser.parse_args()

    meta = read_meta_if_exists(args.csv)
    fps = args.fps if args.fps is not None else (meta.get("fps") if meta else None)
    if fps is None:
        print("Warning: FPS unknown. Provide --fps if predictions look incorrect.")

    if args.all_cells:
        cells = get_all_cell_indices(args.csv)
        print(f"Found {len(cells)} cells in CSV: {cells}")
    elif args.cells:
        cells = parse_cells(args.cells)
    else:
        print("Please specify cells to predict using --cells <list> or --all-cells.")
        print("No valid cells specified; exiting.")
        return 2

    try:
        from src.forecast import predict_from_csv_last_sequence
    except Exception as e:
        print("Failed to import forecasting utilities:", e)
        return 3

    predictions: List[Dict[str, Any]] = []
    for c in cells:
        try:
            pred = predict_from_csv_last_sequence(args.csv, args.model, c, seq_len=args.seq_len, fps=float(fps) if fps else 10.0)
            print(f"Cell {c}: predicted future average density = {pred:.4f}")
            predictions.append({"cell_idx": c, "predicted_density": pred})
        except Exception as e:
            print(f"Cell {c}: prediction failed: {e}")
            predictions.append({"cell_idx": c, "predicted_density": None, "error": str(e)})

    if args.output and predictions:
        out_path = args.output
        print(f"Saving {len(predictions)} predictions to {out_path}...")
        try:
            if out_path.lower().endswith(".json"):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, indent=2)
            else:  # default to CSV
                import csv

                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
                    writer.writeheader()
                    writer.writerows(predictions)
            print("Save complete.")
        except Exception as e:
            print(f"Failed to save output file: {e}")
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
