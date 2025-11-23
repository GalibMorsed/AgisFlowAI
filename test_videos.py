import os
import matplotlib.pyplot as plt
from src.data_processing import load_video_clips
from src.analysis import calculate_instability_score, calculate_density_score

def run_specific_tests(video_files_to_test):
    """
    Processes a list of specified videos, calculates scores, and generates a comparison plot.

    Args:
        video_files_to_test (list): A list of video file names to test (e.g., ['vd4.mp4', 'vd5.mp4']).
    """
    DATA_DIR = "data"
    all_video_scores = {}

    # 1. Process each video and calculate scores
    for video_file in video_files_to_test:
        video_path = os.path.join(DATA_DIR, video_file)
        absolute_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found. The script looked for it at the absolute path: {absolute_path}. Skipping.")
            continue

        print(f"Processing {video_path}...")
        clip_scores = []
        for clip in load_video_clips(video_path):
            instability = calculate_instability_score(clip)
            # Use the middle frame of the clip for density calculation
            density = calculate_density_score(clip[len(clip) // 2])
            clip_scores.append({"instability": instability, "density": density})
        
        all_video_scores[video_file] = clip_scores
        print(f"Finished processing {video_file}. Found {len(clip_scores)} clips.")

    # 2. Visualize the results
    if not all_video_scores:
        print("No videos were processed. Exiting.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('AgisFlowAI: Video Analysis Report', fontsize=16)

    ax1.set_title("Instability Score (Motion Chaos)")
    ax1.set_ylabel("Instability Score")
    ax1.grid(True)

    ax2.set_title("Density Score (Edge Detection Proxy)")
    ax2.set_ylabel("Density Score")
    ax2.set_xlabel("Clip Number")
    ax2.grid(True)

    for video_file, scores in all_video_scores.items():
        instability_scores = [s['instability'] for s in scores]
        density_scores = [s['density'] for s in scores]
        ax1.plot(instability_scores, marker='o', linestyle='-', label=f"Instability - {video_file}")
        ax2.plot(density_scores, marker='x', linestyle='--', label=f"Density - {video_file}")

    ax1.legend()
    ax2.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("video_analysis_report.png")
    print("\nAnalysis complete. Plot saved to video_analysis_report.png")
    plt.show()

if __name__ == "__main__":
    # Automatically find all video files in the data directory
    DATA_DIR = "data"
    supported_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    videos_to_test = []
    for ext in supported_extensions:
        # Find all files with the current extension and add them to the list
        videos_found = [os.path.basename(p) for p in sorted(os.listdir(DATA_DIR)) if p.lower().endswith(ext.replace('*', ''))]
        videos_to_test.extend(videos_found)

    if videos_to_test:
        run_specific_tests(list(dict.fromkeys(videos_to_test))) # Use dict.fromkeys to remove duplicates
    else:
        print(f"No video files found in the '{DATA_DIR}' directory with supported extensions.")