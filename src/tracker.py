"""
A simple multi-object tracker using Kalman Filters and Hungarian algorithm for assignment.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class KalmanTracker:
    """A Kalman filter for tracking a single object."""

    count = 0

    def __init__(self, bbox: np.ndarray):
        import cv2

        self.id = KalmanTracker.count
        KalmanTracker.count += 1

        # State: [cx, cy, w, h, vx, vy]
        self.kf = cv2.KalmanFilter(6, 4)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]], np.float32
        )
        # Transition matrix with constant velocity model
        self.kf.transitionMatrix = np.array(
            [[1, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32
        )
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1

        # Initialize state with the first bounding box
        self.kf.statePost = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0], dtype=np.float32)

        self.history = []
        self.hits = 0
        self.misses = 0

    def predict(self) -> np.ndarray:
        """Predict the next state."""
        return self.kf.predict()

    def update(self, bbox: np.ndarray):
        """Update the filter with a new measurement."""
        self.kf.correct(bbox)
        self.misses = 0
        self.hits += 1

    @property
    def state(self) -> np.ndarray:
        return self.kf.statePost.flatten()


def bbox_to_measurement(bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return np.array([cx, cy, w, h], dtype=np.float32)


def iou(boxA, boxB):
    """Calculate Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val


class MultiObjectTracker:
    """Manages multiple KalmanTrackers."""

    def __init__(self, iou_threshold=0.3, max_misses=5):
        if linear_sum_assignment is None:
            raise ImportError("scipy is required for MultiObjectTracker. Please `pip install scipy`")
        self.trackers: list[KalmanTracker] = []
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses

    def update(self, detections: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        if not self.trackers:
            for det in detections:
                self.trackers.append(KalmanTracker(bbox_to_measurement(det)))
            return [t.state for t in self.trackers]

        predicted_bboxes = [t.predict() for t in self.trackers]
        
        cost_matrix = np.zeros((len(detections), len(self.trackers)))
        for i, det in enumerate(detections):
            for j, pred in enumerate(predicted_bboxes):
                pred_box = [pred[0]-pred[2]/2, pred[1]-pred[3]/2, pred[0]+pred[2]/2, pred[1]+pred[3]/2]
                cost_matrix[i, j] = 1 - iou(det, pred_box)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_trackers = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1 - self.iou_threshold:
                self.trackers[c].update(bbox_to_measurement(detections[r]))
                matched_trackers.add(c)

        for i, tracker in enumerate(self.trackers):
            if i not in matched_trackers:
                tracker.misses += 1

        self.trackers = [t for t in self.trackers if t.misses < self.max_misses]

        unmatched_detections = [det for i, det in enumerate(detections) if i not in row_ind]
        for det in unmatched_detections:
            self.trackers.append(KalmanTracker(bbox_to_measurement(det)))

        return [t.state for t in self.trackers]