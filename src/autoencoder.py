# src/autoencoder.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import glob

# Ensure the project root is on the Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class ConvAutoencoder(nn.Module):
    """
    A Convolutional Autoencoder to learn representations of "normal" frames.
    Anomalies will result in high reconstruction error.
    """
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 128 x 128
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # -> 16 x 64 x 64
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> 32 x 32 x 32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7), # -> 64 x 26 x 26
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # -> 32 x 32 x 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # -> 16 x 64 x 64
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # -> 1 x 128 x 128
            nn.Sigmoid() # Output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_dataset_from_videos(
    video_dir: Path,
    target_size=(128, 128),
    max_frames_per_video=500
) -> TensorDataset:
    """
    Loads frames from video files, preprocesses them, and creates a PyTorch TensorDataset.

    Args:
        video_dir (Path): Directory containing video files of "normal" behavior.
        target_size (tuple): The (width, height) to resize frames to.
        max_frames_per_video (int): Maximum number of frames to extract from each video.

    Returns:
        TensorDataset: A dataset containing the preprocessed image tensors.
    """
    frames = []
    video_paths = glob.glob(str(video_dir / "*.mp4"))
    print(f"Found {len(video_paths)} videos in {video_dir} for training dataset.")

    for video_path in video_paths:
        print(f"Processing {Path(video_path).name}...")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened() and frame_count < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess: Grayscale -> Resize -> Normalize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, target_size)
            normalized = resized / 255.0
            frames.append(normalized)
            frame_count += 1
        cap.release()

    print(f"Created dataset with {len(frames)} frames.")
    # Convert to PyTorch tensor: (N, H, W) -> (N, C, H, W)
    frame_tensors = torch.from_numpy(np.array(frames, dtype=np.float32)).unsqueeze(1)
    return TensorDataset(frame_tensors)

def main():
    """
    Main training loop for the Convolutional Autoencoder.
    """
    # --- Configuration ---
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    VIDEO_DATA_DIR = ROOT / "data"
    MODEL_SAVE_DIR = ROOT / "models"
    MODEL_SAVE_PATH = MODEL_SAVE_DIR / "autoencoder.pth"

    # --- Setup ---
    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # --- Data ---
    # We use videos of normal crowd flow to train the autoencoder.
    # The file `data/sampel2.mp4` is a good candidate.
    dataset = create_dataset_from_videos(VIDEO_DATA_DIR)
    if not dataset:
        print("Error: No data found. Please place videos of normal crowd behavior in the 'data' directory.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Training Loop ---
    print("\nStarting autoencoder training...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        # Use tqdm for a nice progress bar
        for data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch"):
            img_tensor = data[0].to(device)
            
            # Forward pass
            output = model(img_tensor)
            loss = criterion(output, img_tensor)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.6f}")

    # --- Save Model ---
    print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
    print("Model saved successfully.")

if __name__ == "__main__":
    main()