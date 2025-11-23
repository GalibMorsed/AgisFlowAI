"""Video processing pipeline using OpenCV for DNN detection and optical flow.

This module provides `process_video` which reads a video, divides frames
into a grid, runs the person detector, computes per-cell density, optical
flow direction entropy, density acceleration, and a rule-based instability
score per cell. The function can write an annotated output video.

Note: `cv2` is imported inside functions to avoid failing unit tests where
OpenCV isn't installed.
"""

from __future__ import annotations

import csv
import gzip
import logging
import math
from typing import Optional, Tuple

import numpy as np

from .pipeline import (
    make_grid,
    bbox_center_to_cell,
    angles_from_flow,
    direction_histogram,
    normalized_entropy,
    normalize_feature,
    instability_score,
    risk_category,
)


def _color_for_category(cat: str) -> Tuple[int, int, int]:
    if cat == "Green":
        return (0, 200, 0)
    if cat == "Yellow":
        return (0, 200, 200)
    return (0, 0, 200)


def process_video(
    input_path: str,
    output_path: Optional[str] = None,
    rows: int = 4,
    cols: int = 4,
    detector=None,
    max_frames: Optional[int] = None,
    show: bool = False,
    csv_path: Optional[str] = None,
    csv_sample_rate: int = 1,
    csv_compress: bool = False,
    export_training_csv: Optional[str] = None,
    region_offsets: dict | None = None,
    log_level: int = logging.INFO,
    top_k: int = 3,
) -> None:
    """Process `input_path` and optionally write annotated `output_path`.

    detector: object with `detect(frame, conf_thresh)` method. If None,
        this function will attempt to import `OpenCVDNNPersonDetector` from
        `src.detector` and construct one.
    """
    import cv2

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    grid = make_grid((height, width), rows, cols)
    n_cells = len(grid)

    writer = None
    if output_path:
        # obtain a compatible fourcc function across OpenCV builds
        fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
        if fourcc_func is None:
            vw = getattr(cv2, "VideoWriter", None)
            fourcc_func = getattr(vw, "fourcc", None) if vw is not None else None
        if fourcc_func is not None:
            fourcc = fourcc_func(*"mp4v")
        else:
            # fallback: use 0 which typically selects a default codec on many platforms
            fourcc = 0
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # logger
    logger = logging.getLogger("agisflow.video")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(log_level)

    # optional CSV writer (support gzip compression and sampling)
    csv_file = None
    csv_writer = None
    if csv_path:
        if csv_compress:
            csv_file = gzip.open(csv_path, "wt", newline="", encoding="utf-8")
        else:
            csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_idx", "time_s", "cell_idx", "density", "acceleration", "entropy", "score", "category"]) 

    # optional training CSV exporter + metadata
    train_file = None
    train_writer = None
    train_meta_path = None
    if export_training_csv:
        # always write uncompressed training CSV for easier downstream tooling
        train_file = open(export_training_csv, "w", newline="", encoding="utf-8")
        train_writer = csv.writer(train_file)
        # same column set as main CSV; forecasting utilities read density,acceleration,entropy
        train_writer.writerow(["frame_idx", "time_s", "cell_idx", "density", "acceleration", "entropy", "score", "category"]) 
        # write metadata sidecar (json) with fps and grid shape
        try:
            import json

            train_meta = {
                "fps": float(fps),
                "rows": int(rows),
                "cols": int(cols),
                "csv_sample_rate": int(csv_sample_rate),
                "note": "This file is intended for training forecasting models. Columns: frame_idx,time_s,cell_idx,density,acceleration,entropy,score,category",
            }
            train_meta_path = export_training_csv + ".meta.json"
            with open(train_meta_path, "w", encoding="utf-8") as mf:
                json.dump(train_meta, mf)
        except Exception:
            train_meta_path = None

    if detector is None:
        try:
            from .detector import OpenCVDNNPersonDetector

            detector = OpenCVDNNPersonDetector()
        except Exception:
            detector = None

    prev_gray = None
    densities = []  # list of length T, each is array(n_cells,)

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    logger.info("Video opened: %s (%dx%d) fps=%.2f total_frames=%d", input_path, width, height, fps, total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        if total_frames > 0 and frame_idx % max(1, int(fps)) == 0:
            pct = frame_idx / float(total_frames) * 100.0
            logger.info("Progress: frame %d / %d (%.1f%%)", frame_idx, total_frames, pct)

        # detection
        counts = np.zeros(n_cells, dtype=float)
        if detector is not None:
            boxes = detector.detect(frame, conf_thresh=0.5)
            for b in boxes:
                cell_idx = bbox_center_to_cell(b, (height, width), rows, cols)
                counts[cell_idx] += 1.0

        densities.append(counts)

        # optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        entropies = np.zeros(n_cells, dtype=float)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32),
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            u = flow[:, :, 0]
            v = flow[:, :, 1]
            for i, (x0, y0, x1, y1) in enumerate(grid):
                cell_u = u[y0:y1, x0:x1].ravel()
                cell_v = v[y0:y1, x0:x1].ravel()
                # filter near-zero vectors to reduce noise
                mag = np.hypot(cell_u, cell_v)
                mask = mag > 0.2
                if mask.sum() == 0:
                    entropies[i] = 0.0
                    continue
                angs = angles_from_flow(cell_u[mask], cell_v[mask])
                hist = direction_histogram(angs, bins=8)
                entropies[i] = normalized_entropy(hist)

        prev_gray = gray

        # compute density acceleration if enough frames
        d_arr = np.array(densities[-3:]) if len(densities) >= 3 else np.array(densities)
        # density acceleration per cell: d_t - 2*d_{t-1} + d_{t-2}
        acc = np.zeros(n_cells, dtype=float)
        if d_arr.shape[0] >= 3:
            acc = d_arr[-1] - 2 * d_arr[-2] + d_arr[-3]

        # normalize features across cells for this frame
        d_norm = normalize_feature(densities[-1])
        a_norm = normalize_feature(acc)
        e_norm = normalize_feature(entropies)

        scores = np.zeros(n_cells, dtype=float)
        cats = ["Green"] * n_cells
        for i in range(n_cells):
            base_score = instability_score(float(d_norm[i]), float(a_norm[i]), float(e_norm[i]))
            # apply region offset if provided; support a global fallback key '_global'
            adj = 0.0
            if region_offsets:
                # map cell index to (r, c)
                r = i // cols
                c = i % cols
                region_key = f"r{r}c{c}"
                adj = float(region_offsets.get(region_key, 0.0)) if region_offsets else 0.0
                # global offset
                adj += float(region_offsets.get("_global", 0.0)) if region_offsets.get("_global") is not None else 0.0
            score_val = max(0.0, min(1.0, base_score + adj))
            scores[i] = score_val
            cats[i] = risk_category(score_val)

        # write CSV rows for this frame (respect sampling rate)
        if csv_writer is not None and csv_sample_rate and (frame_idx % max(1, csv_sample_rate) == 0):
            time_s = frame_idx / float(fps) if fps > 0 else 0.0
            for i in range(n_cells):
                csv_writer.writerow([frame_idx, f"{time_s:.3f}", i, float(densities[-1][i]), float(acc[i]) if acc is not None else 0.0, float(e_norm[i]), float(scores[i]), cats[i]])
        # also write training CSV row if requested (same sampling)
        if train_writer is not None and csv_sample_rate and (frame_idx % max(1, csv_sample_rate) == 0):
            time_s = frame_idx / float(fps) if fps > 0 else 0.0
            for i in range(n_cells):
                train_writer.writerow([frame_idx, f"{time_s:.3f}", i, float(densities[-1][i]), float(acc[i]) if acc is not None else 0.0, float(e_norm[i]), float(scores[i]), cats[i]])

        # color ramp helper: map score (0..1) to BGR color (green->yellow->red)
        def color_for_score(s: float) -> Tuple[int, int, int]:
            s = max(0.0, min(1.0, float(s)))
            if s <= 0.5:
                t = s / 0.5
                # green (0,200,0) -> yellow (0,200,200) in BGR terms we vary G and R
                b = 0
                g = int(200 + t * 55)
                r = int(0 + t * 200)
                return (b, g, r)
            else:
                t = (s - 0.5) / 0.5
                # yellow -> red
                b = 0
                g = int(255 - t * 200)
                r = int(200 + t * 55)
                return (b, g, r)

        # annotate frame
        overlay = frame.copy()
        for i, (x0, y0, x1, y1) in enumerate(grid):
            color = color_for_score(scores[i])
            alpha = 0.35 * max(0.12, scores[i])
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            # blend
            cv2.addWeighted(overlay[y0:y1, x0:x1], alpha, frame[y0:y1, x0:x1], 1 - alpha, 0, frame[y0:y1, x0:x1])
            # draw border and text
            cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)
            cv2.putText(frame, f"{scores[i]:.2f}", (x0 + 5, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # HUD: highlight top-k risky cells and draw a ranked list
        try:
            k = max(0, int(top_k))
        except Exception:
            k = 0
        if k > 0:
            # indices of top-k by score
            top_idx = np.argsort(-scores)[:k]
            # highlight boxes with thicker border and draw small label
            for rank, idx in enumerate(top_idx, start=1):
                if idx < 0 or idx >= n_cells:
                    continue
                x0, y0, x1, y1 = grid[int(idx)]
                # thicker border and brighter color
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(frame, f"#{rank} {scores[int(idx)]:.2f}", (x0 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # improved HUD: top-k list with bars (top-left)
            hud_x = 12
            hud_y = 34
            hud_w = 220
            hud_h = max(24, k * 22 + 10)
            # semi-transparent background
            sub = frame.copy()
            cv2.rectangle(sub, (hud_x - 8, hud_y - 22), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
            cv2.addWeighted(sub, 0.45, frame, 0.55, 0, frame)
            for i_h, idx in enumerate(top_idx, start=1):
                score_val = float(scores[int(idx)])
                label = f"{i_h}. Cell {int(idx)}"
                y_pos = hud_y + (i_h - 1) * 22
                cv2.putText(frame, label, (hud_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                # draw bar proportional to score
                bar_x = hud_x + 110
                bar_w = int(min(100, max(0, score_val) * 100))
                cv2.rectangle(frame, (bar_x, y_pos - 12), (bar_x + bar_w, y_pos + 4), (0, 180, 255), -1)
                cv2.putText(frame, f"{score_val:.2f}", (bar_x + 104, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
            logger.debug("Top-%d cells (frame %d): %s", k, frame_idx, ", ".join(str(int(i)) for i in top_idx))

            # per-region summary (split grid into 2x2 regions if possible)
            try:
                r_split = 2 if rows >= 2 else 1
                c_split = 2 if cols >= 2 else 1
                region_h = max(1, rows // r_split)
                region_w = max(1, cols // c_split)
                region_labels = []
                for rr in range(r_split):
                    for cc in range(c_split):
                        # compute cell indices in this region
                        cells_in_region = []
                        for r_i in range(rr * region_h, min(rows, (rr + 1) * region_h)):
                            for c_i in range(cc * region_w, min(cols, (cc + 1) * region_w)):
                                cells_in_region.append(r_i * cols + c_i)
                        if not cells_in_region:
                            continue
                        # find top cell in region
                        region_scores = [(i, float(scores[i])) for i in cells_in_region]
                        top_region_idx, top_region_score = max(region_scores, key=lambda t: t[1])
                        region_labels.append((rr, cc, top_region_idx, float(top_region_score)))
                # draw region badges (bottom-left)
                badge_x = 12
                badge_y = height - 12 - (len(region_labels) * 18)
                for j, (rr, cc, idx, sc) in enumerate(region_labels, start=1):
                    txt = f"R{rr}C{cc}: C{int(idx)} {sc:.2f}"
                    yb = badge_y + (j - 1) * 18
                    cv2.putText(frame, txt, (badge_x, yb), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            except Exception:
                pass

        # timestamp overlay (top-left)
        time_s = frame_idx / float(fps) if fps > 0 else 0.0
        hrs = int(time_s // 3600)
        mins = int((time_s % 3600) // 60)
        secs = int(time_s % 60)
        ms = int((time_s - int(time_s)) * 1000)
        timestr = f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"
        # put a subtle background for readability
        cv2.rectangle(frame, (5, 5), (220, 28), (0, 0, 0), -1)
        cv2.putText(frame, timestr, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # legend (top-right)
        legend_x = max(10, width - 200)
        legend_y = 8
        entries = [("Green", (0, 200, 0)), ("Yellow", (0, 200, 200)), ("Red", (0, 0, 200))]
        for i_e, (label, col) in enumerate(entries):
            rx = legend_x + 10
            ry = legend_y + i_e * 22
            cv2.rectangle(frame, (rx, ry), (rx + 18, ry + 14), col, -1)
            cv2.putText(frame, label, (rx + 24, ry + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if show:
            cv2.imshow("Annotated", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    # ensure CSV file is closed if opened
    if csv_file is not None:
        try:
            csv_file.close()
        except Exception:
            pass
    if train_file is not None:
        try:
            train_file.close()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
