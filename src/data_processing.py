import cv2
import numpy as np
import os

def load_video_clips(video_path, clip_duration_secs=2, fps=10):
    """
    Loads a video and splits it into clips of a specified duration with robust error handling.
    
    Args:
        video_path (str): Path to the video file.
        clip_duration_secs (int): Duration of each clip in seconds.
        fps (int): The target frames per second to sample from the video.

    Yields:
        list: A list of preprocessed frames representing a single clip.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print(f"Warning: Could not read FPS from video {video_path}. Using a default of 30.")
        original_fps = 30

    frame_skip = int(original_fps / fps)
    if frame_skip < 1:
        frame_skip = 1 # Ensure we skip at least one frame if target fps is higher than source

    frames_per_clip = clip_duration_secs * fps
    
    clip_frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            # End of video or cannot read frame
            break

        if frame_count % frame_skip == 0:
            # Ensure frame has 3 channels (BGR) before converting
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # Frame is likely already grayscale or has an unsupported format
                gray_frame = frame

            resized_frame = cv2.resize(gray_frame, (320, 240), interpolation=cv2.INTER_AREA)
            clip_frames.append(resized_frame)

        if len(clip_frames) == frames_per_clip:
            yield clip_frames
            clip_frames = [] # Start a new clip

        frame_count += 1

    # After the loop, yield any remaining frames as a final partial clip
    if clip_frames:
        yield clip_frames

    cap.release()