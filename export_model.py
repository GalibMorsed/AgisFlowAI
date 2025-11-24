# export_model.py
from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    """
    Loads the YOLOv8n model and exports it to the TensorRT format.
    """
    model_file = Path("yolov8n.pt")
    engine_file = model_file.with_suffix(".engine")

    # Automatically detect the correct device
    if torch.cuda.is_available():
        print("NVIDIA GPU detected. Exporting for device 0.")
        device = 0
    else:
        print("No NVIDIA GPU detected. Exporting for CPU.")
        device = 'cpu'

    print(f"Attempting to export '{model_file}' to '{engine_file}' format.")

    try:
        # Load the YOLOv8 nano model.
        # This will automatically download yolov8n.pt if it's not present.
        print("Loading base model...")
        model = YOLO(str(model_file))

        print("Starting TensorRT export. This can take several minutes...")
        # Export the model to TensorRT format.
        model.export(format="engine", half=True, device=device)

        print(f"\nSuccessfully exported model to {engine_file}")
    except Exception as e:
        print(f"\nAn error occurred during export: {e}")

if __name__ == "__main__":
    main()