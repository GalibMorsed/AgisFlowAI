"""OpenCV DNN-based person detector with on-demand model download.

This module implements a lightweight wrapper around OpenCV's DNN
using the MobileNet-SSD Caffe model. The model files will be
downloaded into `models/` when first used.

Notes:
- Importing this module does NOT import `cv2` at top-level to keep
  unit tests lightweight. `cv2` is imported inside the detector class.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

MODEL_PROTO_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt"
MODEL_WEIGHTS_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"


def _download_file(url: str, target: Path) -> None:
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {target}")
    urllib.request.urlretrieve(url, str(target))


class OpenCVDNNPersonDetector:
    """Person detector using MobileNet-SSD (Caffe) via OpenCV DNN.

    Example:
        det = OpenCVDNNPersonDetector()
        boxes = det.detect(frame, conf_thresh=0.5)
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.proto_path = self.model_dir / "MobileNetSSD_deploy.prototxt"
        self.model_path = self.model_dir / "MobileNetSSD_deploy.caffemodel"
        self._ensure_model_files()
        # Lazy import
        try:
            import cv2  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenCV is required for the detector. Install opencv-python.") from e

        # load net
        self.net = cv2.dnn.readNetFromCaffe(str(self.proto_path), str(self.model_path))
        # prefer CPU
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

        # Class index for 'person' in this model is 15
        self.person_class_id = 15

    def _ensure_model_files(self) -> None:
        if self.proto_path.exists() and self.model_path.exists():
            return
        # download
        _download_file(MODEL_PROTO_URL, self.proto_path)
        _download_file(MODEL_WEIGHTS_URL, self.model_path)

    def detect(self, frame, conf_thresh: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Run detection on BGR image `frame`. Returns list of bboxes (x0,y0,x1,y1).

        The returned coordinates are pixel coordinates relative to the input frame.
        """
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes: List[Tuple[int, int, int, int]] = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            cls_id = int(detections[0, 0, i, 1])
            if cls_id != self.person_class_id:
                continue
            if conf < conf_thresh:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            # clip
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
        return boxes


class YOLOv8PersonDetector:
    """Person detector using YOLOv8 via the `ultralytics` package.

    Requires `pip install ultralytics`. The model is downloaded on first use.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "YOLOv8 detector requires `ultralytics`. Please install it with: pip install ultralytics"
            ) from e

        self.model = YOLO(model_name)
        # The 'person' class is typically index 0 in COCO-trained models
        self.person_class_id = 0

    def detect(self, frame, conf_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Run YOLOv8 detection. Returns list of bboxes (x0,y0,x1,y1)."""
        # The model expects RGB, but OpenCV provides BGR. The model handles conversion.
        results = self.model(frame, classes=[self.person_class_id], conf=conf_thresh, verbose=False)
        boxes: List[Tuple[int, int, int, int]] = []

        # Process results
        for result in results:
            # result.boxes contains the detected boxes for a single image
            for box in result.boxes:
                # Get coordinates as (x1, y1, x2, y2)
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)

                # Basic validation
                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append((x1, y1, x2, y2))

        return boxes


class BackgroundSubtractorDetector:
    """Lightweight foreground-based detector using OpenCV background subtraction.

    This is a fast fallback that does not require model downloads. It will
    produce coarse bounding boxes for moving objects (people) and is useful
    for testing the pipeline without heavy dependencies.
    """

    def __init__(self, min_area: int = 500):
        self.min_area = int(min_area)
        self.subtractor = None

    def detect(self, frame, conf_thresh: float = 0.5):
        """Return list of bboxes (x0,y0,x1,y1) of foreground blobs above min_area."""
        try:
            import cv2
        except Exception as e:
            raise RuntimeError("OpenCV is required for the background subtractor. Install opencv-python.") from e

        if self.subtractor is None:
            # create with default params; history can be tuned
            self.subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        fgmask = self.subtractor.apply(frame)
        # remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = frame.shape[:2]
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw * ch < self.min_area:
                continue
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x + cw)
            y2 = min(h - 1, y + ch)
            boxes.append((x1, y1, x2, y2))
        return boxes