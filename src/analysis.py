import cv2
import numpy as np

def calculate_instability_score(clip_frames):
    """
    Calculates an instability score for a clip based on optical flow.
    A higher score means more chaotic movement.

    Args:
        clip_frames (list): A list of grayscale frames for one clip.

    Returns:
        float: The calculated instability score. Returns 0.0 if calculation is not possible.
    """
    if not clip_frames or len(clip_frames) < 2:
        return 0.0

    prev_frame = clip_frames[0]
    flow_magnitudes = []
    flow_angles_std = []

    for i in range(1, len(clip_frames)):
        next_frame = clip_frames[i]
        
        # Ensure frames are valid for optical flow
        if prev_frame is None or next_frame is None or prev_frame.shape != next_frame.shape:
            continue

        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Only consider frames with significant movement to avoid noise
            mean_magnitude = np.mean(magnitude)
            if mean_magnitude > 1.0:
                flow_magnitudes.append(mean_magnitude)
                flow_angles_std.append(np.std(angle))

        except cv2.error as e:
            print(f"OpenCV error during optical flow calculation: {e}")
            continue # Skip this frame pair

        prev_frame = next_frame

    if not flow_magnitudes or not flow_angles_std:
        return 0.0

    # Instability is a product of movement intensity and directional chaos
    mean_magnitude = np.mean(flow_magnitudes)
    mean_angle_std = np.mean(flow_angles_std)
    
    # Normalize angle standard deviation (in radians) to a 0-1 range for scoring
    instability_score = mean_magnitude * (mean_angle_std / (np.pi / 2))
    
    return instability_score

def calculate_density_score(frame):
    if frame is None:
        return 0.0
    edges = cv2.Canny(frame, 50, 150)
    density_score = np.sum(edges > 0) / edges.size
    return density_score