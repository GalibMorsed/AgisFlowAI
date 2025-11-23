import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from src import video_processor


class FakeCapture:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        f = self.frames[self.i]
        self.i += 1
        return True, f.copy()

    def get(self, prop):
        # support CAP_PROP_FRAME_COUNT, WIDTH, HEIGHT, FPS
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return int(self.frames[0].shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return int(self.frames[0].shape[0])
        if prop == cv2.CAP_PROP_FPS:
            return 10.0
        return 0

    def release(self):
        pass


class FakeWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        pass


class MockDetector:
    def detect(self, frame, conf_thresh=0.5):
        h, w = frame.shape[:2]
        return [(w // 2 - 4, h // 2 - 4, w // 2 + 4, h // 2 + 4)]


class MockFlow:
    @staticmethod
    def calcOpticalFlowFarneback(a, b, *args, **kwargs):
        # return zero flow
        h, w = a.shape[:2]
        return np.zeros((h, w, 2), dtype=float)


class MockCV2Module:
    def __init__(self, frames):
        import cv2 as real_cv2

        self._frames = frames
        self.video_writer = FakeWriter()
        # expose commonly used constants and delegate drawing to real cv2
        self.CAP_PROP_FRAME_COUNT = real_cv2.CAP_PROP_FRAME_COUNT
        self.CAP_PROP_FRAME_WIDTH = real_cv2.CAP_PROP_FRAME_WIDTH
        self.CAP_PROP_FRAME_HEIGHT = real_cv2.CAP_PROP_FRAME_HEIGHT
        self.CAP_PROP_FPS = real_cv2.CAP_PROP_FPS
        self.COLOR_BGR2GRAY = real_cv2.COLOR_BGR2GRAY
        self.FONT_HERSHEY_SIMPLEX = real_cv2.FONT_HERSHEY_SIMPLEX
        # delegate drawing and convenience functions
        self.rectangle = real_cv2.rectangle
        self.addWeighted = real_cv2.addWeighted
        self.putText = real_cv2.putText
        self.cvtColor = real_cv2.cvtColor
        self.waitKey = real_cv2.waitKey
        self.destroyAllWindows = real_cv2.destroyAllWindows
        # obtain VideoWriter_fourcc safely; some cv2 versions expose it differently or not at all
        self.VideoWriter_fourcc = getattr(real_cv2, 'VideoWriter_fourcc', lambda *args, **kwargs: 0)

    def VideoCapture(self, path):
        return FakeCapture(self._frames)

    def VideoWriter(self, *args, **kwargs):
        return self.video_writer

    def calcOpticalFlowFarneback(self, *args, **kwargs):
        return MockFlow.calcOpticalFlowFarneback(*args, **kwargs)


class MockCaptureTest(unittest.TestCase):
    def test_process_with_mocked_cv2(self):
        # create 4 small frames
        frames = []
        for i in range(4):
            f = np.zeros((48, 64, 3), dtype=np.uint8)
            cv = 10 + i * 5
            f[10:20, cv:cv + 10] = 255
            frames.append(f)

        # patch cv2 in the video_processor module
        mock_cv2 = MockCV2Module(frames)

        with mock.patch.dict('sys.modules', {'cv2': mock_cv2}):
            # run with mock detector
            tmp = tempfile.mkdtemp(prefix='agisflow_test_')
            csv_path = os.path.join(tmp, 'out.csv')
            try:
                video_processor.process_video('ignored', output_path=None, rows=4, cols=4, detector=MockDetector(), max_frames=4, csv_path=csv_path, show=False, top_k=2)
                # check CSV exists
                self.assertTrue(os.path.exists(csv_path))
                with open(csv_path, 'r', encoding='utf-8') as fh:
                    data = fh.read()
                self.assertIn('frame_idx', data)
            finally:
                try:
                    if csv_path and os.path.exists(csv_path):
                        os.remove(csv_path)
                except Exception:
                    pass
                os.rmdir(tmp)


if __name__ == '__main__':
    unittest.main()
