import csv
import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

from src.video_processor import process_video


class CsvIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="agisflow_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _create_video(self, path: str, frames: int = 6, w: int = 160, h: int = 120) -> None:
        # obtain a fourcc function defensively to avoid static-analysis errors
        fourcc_func = getattr(cv2, "VideoWriter_fourcc", None)
        if callable(fourcc_func):
            fourcc = fourcc_func(*"MJPG")
        else:
            # Fall back to a manual construction of the fourcc int (avoids referencing cv2.cv)
            def _make_fourcc(s: str) -> int:
                # build 32-bit integer from four characters in little-endian order
                a, b, c, d = (ord(ch) & 0xFF for ch in s[:4])
                return a | (b << 8) | (c << 16) | (d << 24)
            try:
                fourcc = _make_fourcc("MJPG")
            except Exception:
                fourcc = 0

        # ensure fourcc is a plain int for static type checkers
        try:
            fourcc = int(fourcc)  # type: ignore[arg-type]
        except Exception:
            fourcc = 0

        out = cv2.VideoWriter(path, fourcc, 10.0, (w, h))
        for i in range(frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # draw a moving white square so there is some motion
            x = 10 + (i % max(1, frames)) * 5
            cv2.rectangle(frame, (x, 20), (x + 20, 50), (255, 255, 255), -1)
            out.write(frame)
        out.release()

    def test_csv_export_and_video_generated_with_mock_detector(self):
        video_path = os.path.join(self.tmpdir, "input.avi")
        out_video = os.path.join(self.tmpdir, "out.avi")
        csv_path = os.path.join(self.tmpdir, "out.csv")

        n_frames = 6
        rows = 4
        cols = 4
        self._create_video(video_path, frames=n_frames, w=160, h=120)

        class MockDetector:
            def detect(self, frame, conf_thresh: float = 0.5):
                # return a single bbox near the center for any frame
                h, w = frame.shape[:2]
                return [(w // 2 - 10, h // 2 - 10, w // 2 + 10, h // 2 + 10)]

        # run processing with mock detector (no network/model download)
        process_video(
            video_path,
            output_path=out_video,
            rows=rows,
            cols=cols,
            detector=MockDetector(),
            max_frames=n_frames,
            csv_path=csv_path,
            log_level=10,
            top_k=2,
        )

        self.assertTrue(os.path.exists(out_video), "Output video was not created")
        self.assertTrue(os.path.exists(csv_path), "CSV output was not created")

        # validate CSV contents
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows_list = list(reader)

        self.assertGreaterEqual(len(rows_list), 2)
        header = rows_list[0]
        expected_header = ["frame_idx", "time_s", "cell_idx", "density", "acceleration", "entropy", "score", "category"]
        self.assertEqual(header, expected_header)

        n_cells = rows * cols
        expected_rows = n_frames * n_cells
        self.assertEqual(len(rows_list) - 1, expected_rows)

        # check some value types and category values
        categories = {"Green", "Yellow", "Red"}
        for rec in rows_list[1:1 + n_cells]:
            # record example: [frame_idx, time_s, cell_idx, density, acceleration, entropy, score, category]
            self.assertEqual(len(rec), 8)
            self.assertTrue(rec[0].isdigit())
            self.assertTrue(float(rec[2]) >= 0)
            self.assertIn(rec[7], categories)


if __name__ == "__main__":
    unittest.main()
