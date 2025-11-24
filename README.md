# AgisFlowAI — Starter Project

Minimal Python project scaffold created by the assistant.

Quickstart

1. Run the CLI:

```powershell
python code.py --name "Your Name"
```

2. Run unit tests:

```powershell
python -m unittest discover -v
```

Modify `src/main.py` to add your application logic.

## OpenCV DNN detector

This starter implements an OpenCV DNN-based person detector using the
MobileNet-SSD (Caffe) model. The first time you run the video pipeline the
model files will be downloaded into the `models/` folder.

Install runtime deps:

```powershell
python -m pip install -r requirements.txt
```

Run the pipeline on a video (example):

```powershell
python -m src.main run --video path\to\input.mp4 --rows 4 --cols 4
```

If you don't want to install heavy ML libs yet, unit tests for the
pure-Python pipeline helpers still run without OpenCV.

## CSV export and logging

You can export per-frame per-cell features to CSV for offline analysis and
enable logging/progress output. Example:

````powershell
python -m src.main run --video path\to\input.mp4 --detector bg --output out.mp4 --csv out.csv --log-level INFO

HUD: top-k risky cells
----------------------
You can highlight the top-k risky cells on the video using `--topk` (default 3):

```powershell
python -m src.main run --video path\to\input.mp4 --detector bg --output out.mp4 --csv out.csv --topk 5
````

This draws a ranked HUD and highlights the top cells on each frame.

````

The CSV has columns: `frame_idx,time_s,cell_idx,density,acceleration,entropy,score,category`.

Legend and timestamps are overlaid on the annotated output video.

CSV sampling & compression
--------------------------
To reduce CSV size you can sample frames and/or compress the CSV with gzip:

```powershell
python -m src.main run --video path\to\input.mp4 --csv out.csv --csv-sample-rate 5 --csv-compress
````

## Region offsets

You can apply small per-region score offsets to tune sensitivity. Use the
`--region-offset` flag with pairs like `r0c0:0.1,r1c1:-0.05` or a single
global value like `0.05` (applies to all cells):

```powershell
python -m src.main run --video path\to\input.mp4 --csv out.csv --region-offset "r0c0:0.1,r0c1:-0.05"
```

## CI coverage

The GitHub Actions CI runs tests under `coverage` and uploads a `coverage.xml`
artifact on each run. You can download this artifact from the workflow run to
inspect coverage reports.

## Forecasting (supervised)

You can train a short-term per-cell forecaster that predicts the average
future `density` for a given cell over a forecast horizon (e.g. 10s). The
pipeline CSV is used as training data. Two model classes are supported:

- `conv1d`: efficient 1D-CNN over the time axis (recommended for speed)
- `lstm` / `gru`: recurrent models for sequence modeling

Example: train a 1D-CNN for a 10s horizon (assumes `out.csv` produced by `run`):

```powershell
python -m src.main train-forecast --csv out.csv --model conv1d --seq-len 16 --horizon 10.0 --fps 10 --epochs 20 --out models/forecast_conv1d.pt
```

Predict using the last `seq-len` frames for a specific cell (index as in CSV):

```powershell
python -m src.main predict-forecast --csv out.csv --model-path models/forecast_conv1d.pt --cell 3 --seq-len 16 --fps 10
```

Notes:

- Training uses PyTorch; install it before running `train-forecast` (`pip install torch`).
- The trainer defers importing PyTorch until training time so tests and the
  rest of the pipeline do not require it.

```

```
