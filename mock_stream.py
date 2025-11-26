# mock_stream.py
import cv2
import numpy as np
import csv
import time
import argparse
import sys
from pathlib import Path
from collections import deque


# Ensure the project root is on the Python path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from src.detector import YOLOv8PersonDetector
    from src.tracker import MultiObjectTracker
    from src.pipeline import (
        make_grid,
        bbox_center_to_cell,
        angles_from_flow,
        direction_histogram,
        normalized_entropy,
        normalize_feature,
        instability_score,
    )
except ImportError as e:
    print(f"Error importing detector: {e}")
    print("Please ensure 'ultralytics' is installed (`pip install ultralytics`) and you are running from the project root.")
    sys.exit(1)

def draw_flow(img, flow, step=16):
    """Draws optical flow vectors on an image."""
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)

def color_for_score(s: float) -> tuple[int, int, int]:
    """Helper for visualization: map score (0..1) to BGR color."""
    s = max(0.0, min(1.0, float(s)))
    # Interpolate between Green -> Yellow -> Red
    # Green (0,255,0), Yellow (0,255,255), Red (0,0,255) in BGR
    return (0, int(255 * (1-s)**2), int(255 * s**2))

def draw_history_chart(frame, history: dict, top_cell_idx: int, width: int = 300, height: int = 150):
    """Draws a time-series chart for the metrics of a specific cell."""
    if top_cell_idx is None or not history['score']:
        return

    chart_img = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
    
    # Get data for the top cell
    score_hist = [h[top_cell_idx] for h in history['score']]
    density_hist = [h[top_cell_idx] for h in history['density']]
    accel_hist = [h[top_cell_idx] for h in history['accel']]
    entropy_hist = [h[top_cell_idx] for h in history['entropy']]

    # Draw Title
    title = f"Cell {top_cell_idx} Metrics"
    cv2.putText(chart_img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw data lines
    series = {
        "Score": (score_hist, (0, 0, 255)),       # Red
        "Density": (density_hist, (0, 255, 0)),   # Green
        "Accel": (accel_hist, (255, 255, 0)),     # Cyan
        "Entropy": (entropy_hist, (255, 0, 255)), # Magenta
    }

    y_offset = 40
    for i, (name, (data, color)) in enumerate(series.items()):
        if not data:
            continue
        
        # Draw legend
        cv2.putText(chart_img, name, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw polyline for the data history
        points = np.array([[int(width * (j / len(data))), int(height - 20 - (val * (height - 40)))] for j, val in enumerate(data) if val is not None], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(chart_img, [points], isClosed=False, color=color, thickness=1)

    # Overlay chart onto the main frame (top-right corner)
    x_offset = frame.shape[1] - width - 10
    y_offset = 10
    frame[y_offset:y_offset+height, x_offset:x_offset+width] = cv2.addWeighted(frame[y_offset:y_offset+height, x_offset:x_offset+width], 0.3, chart_img, 0.7, 0)

def simulate_live_stream(video_source: str | int, output_path: str | None = None, csv_path: str | None = None, fps: int = 30):
    """
    Reads a video file and loops it to simulate a live camera feed.

    Args:
        video_source (str | int): Path to the video file or an integer for camera index.
        fps (int): The desired frames per second for playback.
    """
    is_camera = isinstance(video_source, int)
    delay = 1 / fps
    window_name = "CrowdGuard Mock Stream"
    
    writer = None
    csv_file = None

    # --- Initialization ---
    # Step A: Initialize the YOLOv8 person detector
    print("Initializing YOLOv8 person detector...")
    detector = YOLOv8PersonDetector()
    # Step B: Initialize variables for optical flow
    prev_gray = None
    
    # --- Phase 3: Initialization ---
    # 1. Grid Segmentation
    rows, cols = 10, 10
    grid = None # Will be initialized on first frame
    n_cells = rows * cols

    # 2. Per-cell metrics history
    densities = [] # Stores density history to calculate acceleration
    accelerations = np.zeros(n_cells, dtype=float) # Stores the last calculated acceleration
    
    # History for plotting charts
    history_len = 100
    metric_history = {
        'score': deque(maxlen=history_len),
        'density': deque(maxlen=history_len),
        'accel': deque(maxlen=history_len),
        'entropy': deque(maxlen=history_len),
    }

    # --- Phase 5: Initialization ---
    print("Initializing multi-object tracker...")
    tracker = MultiObjectTracker()


    while True:
        # Open the video file
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {video_source}")
            return # Exit if the source can't be opened

        # --- Setup Output Writers on First Loop ---
        if writer is None and output_path:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
            print(f"Will write annotated video to: {output_path}")

        if csv_file is None and csv_path:
            csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            # Write header
            header = ["frame_idx", "time_s", "cell_idx", "density", "acceleration", "entropy", "score"]
            csv_writer.writerow(header)
            print(f"Will write data log to: {csv_path}")


        print("--- Starting video loop ---")
        frame_idx = 0
        while cap.isOpened():
            frame_start_time = time.time()

            ret, frame = cap.read()

            # If the frame is not returned, we've reached the end of the video
            if not ret:
                if not is_camera:
                    print("--- End of video, looping... ---")
                break  # Break the inner loop to reopen the file
            
            frame_idx += 1

            # Initialize grid on first frame
            if grid is None:
                h, w = frame.shape[:2]
                grid = make_grid((h, w), rows, cols)

            # --- Vision Pipeline (Detection & Flow) ---
            # Object Detection
            # Get bounding boxes for every person in the frame.
            boxes = detector.detect(frame, conf_thresh=0.25)

            # --- Phase 5: Trajectory Projection ---
            # Update tracker with new detections
            tracked_objects = tracker.update(boxes)

            # Optical Flow
            # Convert the frame to grayscale for flow calculation.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = None
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

            prev_gray = gray

            # --- Phase 3: Mathematical Core ---
            
            # 2. Calculate Metrics per Cell
            # Density: Count people in each cell
            current_counts = np.zeros(n_cells, dtype=float)
            for (x1, y1, x2, y2) in boxes:
                cell_idx = bbox_center_to_cell((x1, y1, x2, y2), frame.shape[:2], rows, cols)
                if 0 <= cell_idx < n_cells:
                    current_counts[cell_idx] += 1
            densities.append(current_counts)
            if len(densities) > 3:
                densities.pop(0) # Keep history size fixed

            # Motion Entropy & Density Acceleration
            entropies = np.zeros(n_cells, dtype=float)
            
            if flow is not None:
                u, v = flow[:, :, 0], flow[:, :, 1]
                for i, (x0, y0, x1, y1) in enumerate(grid):
                    # Entropy
                    cell_u, cell_v = u[y0:y1, x0:x1].ravel(), v[y0:y1, x0:x1].ravel()
                    magnitudes = np.hypot(cell_u, cell_v)
                    # Filter out small, noisy movements
                    mask = magnitudes > 0.5
                    if np.any(mask):
                        angles = angles_from_flow(cell_u[mask], cell_v[mask])
                        hist = direction_histogram(angles, bins=8)
                        entropies[i] = normalized_entropy(hist)

            if len(densities) == 3:
                # Acceleration
                d_t, d_t1, d_t2 = densities[2], densities[1], densities[0]
                accelerations = d_t - 2 * d_t1 + d_t2

            # Instability Fusion
            # Normalize features across all cells for this frame
            density_norm = normalize_feature(current_counts)
            accel_norm = normalize_feature(accelerations, clip_min=0) # Only positive acceleration is a risk
            entropy_norm = normalize_feature(entropies)
            
            # New "Path Blockage" feature
            blockage = density_norm * (1 - entropy_norm) # High density, low entropy
            blockage_norm = normalize_feature(blockage)

            scores = np.zeros(n_cells, dtype=float)
            for i in range(n_cells):
                # w1=accel, w2=entropy, w3=blockage
                scores[i] = instability_score(accel_norm[i], entropy_norm[i], blockage_norm[i], w1=0.5, w2=0.2, w3=0.3)

            # Update metric history for charts
            metric_history['score'].append(scores.copy())
            metric_history['density'].append(density_norm.copy())
            metric_history['accel'].append(accel_norm.copy())
            metric_history['entropy'].append(entropy_norm.copy())

            # --- CSV Logging ---
            if csv_file and csv_writer:
                time_s = frame_idx / (cap.get(cv2.CAP_PROP_FPS) or fps)
                for i in range(n_cells):
                    row_data = [frame_idx, f"{time_s:.3f}", i, current_counts[i], accelerations[i], entropies[i], scores[i]]
                    csv_writer.writerow(row_data)

            # --- Visualization ---
            overlay = frame.copy()
            for i, (x0, y0, x1, y1) in enumerate(grid):
                score = scores[i]
                if score > 0.1: # Only draw for non-trivial scores
                    color = color_for_score(score)
                    alpha = 0.2 + 0.4 * score # Make high scores more opaque
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Draw grid lines and score text
            for i, (x0, y0, x1, y1) in enumerate(grid):
                cv2.rectangle(frame, (x0, y0), (x1, y1), (150, 150, 150), 1)
                cv2.putText(frame, f"{scores[i]:.2f}", (x0 + 5, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Draw trajectories
            video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
            for obj_state in tracked_objects:
                cx, cy, w, h, vx, vy = obj_state
                # Draw current position
                pt1 = (int(cx - w/2), int(cy - h/2))
                pt2 = (int(cx + w/2), int(cy + h/2))
                cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2) # Blue box for tracked objects

                # Project 5 seconds into the future
                future_cx = int(cx + vx * 5 * video_fps)
                future_cy = int(cy + vy * 5 * video_fps)
                
                cv2.line(frame, (int(cx), int(cy)), (future_cx, future_cy), (255, 255, 0), 2)

            # Add a text overlay for the detection count
            detection_text = f"Detections: {len(boxes)}"
            cv2.putText(frame, detection_text, (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 
                        cv2.LINE_AA)
            
            # Add the video filename
            if is_camera:
                video_name_text = f"Camera Index: {video_source}"
            else:
                video_name_text = f"Video: {Path(video_source).name}"
            cv2.putText(frame, video_name_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)            

            # Draw prediction chart for the most dangerous cell
            top_cell_idx = np.argmax(scores) if scores.any() else None
            if top_cell_idx is not None:
                draw_history_chart(frame, metric_history, top_cell_idx)

            cv2.imshow(window_name, frame)

            # --- Save Frame to Video ---
            if writer:
                writer.write(frame)

            # Control the playback speed
            elapsed_time = time.time() - frame_start_time
            wait_time = max(1, int((delay - elapsed_time) * 1000))
            # Exit on 'q' key press
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                print("Exiting stream.")
                cap.release()
                cv2.destroyAllWindows()
                if writer:
                    writer.release()
                if csv_file:
                    csv_file.close()
                return

        # Release the capture object before the next loop iteration
        cap.release()

        # If using a camera, we don't want to loop.
        if is_camera:
            break

    cv2.destroyAllWindows()
    if writer:
        writer.release()
    if csv_file:
        # --- Analysis Explanation File Generation ---
        # Ensure the file is closed before trying to read it
        csv_file.close()
        
        explanation_path = ROOT / "analysis_explanation.txt"
        logger.info(f"Generating analysis explanation file at: {explanation_path}")
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)

            # Find the frame with the highest overall score
            max_score_frame = df.loc[df['score'].idxmax()]
            peak_frame = int(max_score_frame['frame_idx'])
            peak_score = max_score_frame['score']
            peak_cell = int(max_score_frame['cell_idx'])

            # Count frames with "Danger" and "Warning"
            danger_frames = df[df['score'] >= 0.6]['frame_idx'].nunique()
            warning_frames = df[(df['score'] >= 0.3) & (df['score'] < 0.6)]['frame_idx'].nunique()

            # Find the most frequently dangerous cell
            danger_cells = df[df['score'] >= 0.6]
            most_dangerous_cell = danger_cells['cell_idx'].mode()[0] if not danger_cells.empty else "None"

            # Write the explanation
            with open(explanation_path, "w", encoding="utf-8") as f:
                f.write("CrowdGuard Analysis Report\n")
                f.write("="*30 + "\n\n")
                f.write(f"Video Source: {Path(video_source).name if isinstance(video_source, str) else f'Camera {video_source}'}\n")
                f.write(f"Total Frames Analyzed: {df['frame_idx'].max()}\n\n")
                f.write("Key Findings:\n")
                f.write("-------------\n")
                f.write(f"- Peak Hazard Score: {peak_score:.3f} occurred at frame {peak_frame} in cell {peak_cell}.\n")
                f.write(f"- Total frames with 'Danger' level alerts (score >= 0.6): {danger_frames}\n")
                f.write(f"- Total frames with 'Warning' level alerts (0.3 <= score < 0.6): {warning_frames}\n")
                f.write(f"- Most frequently dangerous cell: {most_dangerous_cell}\n")
            
            logger.info("Successfully wrote analysis explanation.")

        except ImportError:
            logger.warning("Pandas is not installed. Skipping explanation file generation. Please run 'pip install pandas'.")
        except Exception as e:
            logger.error(f"Failed to generate explanation file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a live video stream from a file.")
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Path to a video file or an integer for the camera index (e.g., '0' for the default camera)."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to save the annotated output video file (e.g., output.mp4)."
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to save the logged per-cell data (e.g., log.csv)."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the simulated stream."
    )
    args = parser.parse_args()

    video_source_arg = args.video
    video_source: str | int

    try:
        # Try to convert to integer for camera index
        video_source = int(video_source_arg)
    except ValueError:
        # If it's not an integer, treat it as a file path
        video_path = Path(video_source_arg)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        video_source = str(video_path)

    simulate_live_stream(video_source, args.output, args.csv, args.fps)
    