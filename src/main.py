"""Small CLI and core functions for the starter project."""

from __future__ import annotations

import argparse
from typing import Iterable

from .pipeline import instability_score, risk_category


def greet(name: str) -> str:
    return f"Hello, {name}!"


from typing import Sequence

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simple starter CLI for AgisFlowAI")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run the pipeline on a video")
    run.add_argument("--video", "-v", required=True, help="Path to input video file")
    run.add_argument("--rows", type=int, default=4, help="Grid rows")
    run.add_argument("--cols", type=int, default=4, help="Grid cols")
    run.add_argument("--detector", choices=["dnn", "bg"], default="dnn", help="Detector to use: 'dnn' (OpenCV DNN) or 'bg' (background subtractor)")
    run.add_argument("--output", "-o", help="Path to annotated output video (optional)")
    run.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
    run.add_argument("--show", action="store_true", help="Show annotated frames during processing")
    run.add_argument("--csv", help="Optional CSV path to write per-frame per-cell features")
    run.add_argument("--csv-sample-rate", type=int, default=1, help="Write CSV every N frames (1 = every frame)")
    run.add_argument("--csv-compress", action="store_true", help="Compress CSV output with gzip if set")
    run.add_argument("--region-offset", help="Optional per-region offsets (format: 'r0c0:0.1,r0c1:-0.05' or global float '0.05')")
    run.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    run.add_argument("--topk", type=int, default=3, help="Number of top risky cells to highlight in HUD")
    run.add_argument("--export-train", help="Path to write a training CSV and .meta.json sidecar for forecasting")

    hello = sub.add_parser("hello", help="Simple greeting")
    hello.add_argument("--name", "-n", default="World", help="Name to greet")

    # Forecasting subcommands
    tf = sub.add_parser("train-forecast", help="Train a short-term forecast model from a CSV")
    tf.add_argument("--csv", required=True, help="CSV path produced by the pipeline")
    tf.add_argument("--model", choices=["conv1d", "lstm", "gru"], default="conv1d", help="Model type")
    tf.add_argument("--seq-len", type=int, default=16, help="Input sequence length (frames)")
    tf.add_argument("--horizon", type=float, default=10.0, help="Forecast horizon in seconds")
    tf.add_argument("--fps", type=float, default=10.0, help="Frames per second of the CSV/video")
    tf.add_argument("--epochs", type=int, default=10, help="Training epochs")
    tf.add_argument("--batch", type=int, default=64, help="Batch size")
    tf.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    tf.add_argument("--out", help="Where to save the trained model (defaults to models/forecast_{model}.pt)")

    pred = sub.add_parser("predict-forecast", help="Predict using a trained forecast model")
    pred.add_argument("--csv", required=True, help="CSV path produced by the pipeline")
    pred.add_argument("--model-path", required=True, help="Path to saved forecast model (.pt)")
    pred.add_argument("--cell", type=int, required=True, help="Cell index to predict for")
    pred.add_argument("--seq-len", type=int, default=16, help="Input sequence length (frames)")
    pred.add_argument("--fps", type=float, default=10.0, help="Frames per second of the CSV/video")

    args = parser.parse_args(argv)

    if args.cmd == "hello":
        print(greet(args.name))
        return 0

    if args.cmd == "run":
        print("Starting pipeline run...")
        try:
            from .video_processor import process_video
        except Exception as e:
            print("Failed to import video processor:", e)
            return 2

        detector_obj = None
        if args.detector == "dnn":
            try:
                from .detector import OpenCVDNNPersonDetector

                detector_obj = OpenCVDNNPersonDetector()
            except Exception as e:
                print("OpenCV DNN detector unavailable:", e)
                print("Falling back to background-subtractor detector.")
                try:
                    from .detector import BackgroundSubtractorDetector

                    detector_obj = BackgroundSubtractorDetector()
                except Exception as e2:
                    print("Fallback detector unavailable:", e2)
                    detector_obj = None
        else:
            try:
                from .detector import BackgroundSubtractorDetector

                detector_obj = BackgroundSubtractorDetector()
            except Exception as e:
                print("Background-subtractor unavailable:", e)
                detector_obj = None

        try:
            # configure basic logging level for the called modules
            import logging

            logging.basicConfig(level=getattr(logging, args.log_level))

            # parse region offsets
            region_offsets = None
            if args.region_offset:
                s = args.region_offset.strip()
                region_offsets = {}
                # global float
                if ":" not in s and "," not in s:
                    try:
                        region_offsets["_global"] = float(s)
                    except Exception:
                        region_offsets = None
                else:
                    parts = [p.strip() for p in s.split(",") if p.strip()]
                    for p in parts:
                        if ":" in p:
                            k, v = p.split(":", 1)
                            try:
                                region_offsets[k.strip()] = float(v)
                            except Exception:
                                pass

            process_video(
                args.video,
                output_path=args.output,
                rows=args.rows,
                cols=args.cols,
                detector=detector_obj,
                max_frames=args.max_frames,
                show=args.show,
                csv_path=args.csv,
                csv_sample_rate=args.csv_sample_rate,
                csv_compress=args.csv_compress,
                export_training_csv=args.export_train,
                region_offsets=region_offsets,
                log_level=getattr(logging, args.log_level),
                top_k=args.topk,
            )
        except Exception as e:
            print("Processing failed:", e)
            return 3

        print("Processing finished.")
        return 0

    if args.cmd == "train-forecast":
        try:
            from .forecast import train_forecast
        except Exception as e:
            print("Forecasting utilities unavailable:", e)
            return 4

        try:
            out = train_forecast(
                args.csv,
                model_type=args.model,
                seq_len=args.seq_len,
                horizon_sec=args.horizon,
                fps=args.fps,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                out_path=args.out,
            )
            print("Saved model to:", out)
        except Exception as e:
            print("Training failed:", e)
            return 5

        return 0

    if args.cmd == "predict-forecast":
        try:
            from .forecast import predict_from_csv_last_sequence
        except Exception as e:
            print("Forecasting utilities unavailable:", e)
            return 4

        try:
            pred = predict_from_csv_last_sequence(
                args.csv, args.model_path, args.cell, seq_len=args.seq_len, fps=args.fps
            )
            print(f"Predicted future average density for cell {args.cell}: {pred:.4f}")
        except Exception as e:
            print("Prediction failed:", e)
            return 6

        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
