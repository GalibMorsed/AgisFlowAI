import math
import unittest

import numpy as np

from src.pipeline import (
    make_grid,
    bbox_center_to_cell,
    angles_from_flow,
    direction_histogram,
    entropy_of_distribution,
    normalized_entropy,
    instability_score,
    risk_category,
)


class PipelinePureFunctionsTest(unittest.TestCase):
    def test_make_grid_and_mapping(self):
        shape = (100, 200)  # h, w
        rects = make_grid(shape, 4, 4)
        self.assertEqual(len(rects), 16)
        # center of left-top cell bbox should map to cell 0
        bbox = (10, 10, 30, 30)
        idx = bbox_center_to_cell(bbox, shape, 4, 4)
        self.assertEqual(idx, 0)

    def test_angles_and_histogram_entropy(self):
        u = [1.0, 0.0, -1.0, 0.0]
        v = [0.0, 1.0, 0.0, -1.0]
        angles = angles_from_flow(u, v)
        # should give four orthogonal directions
        self.assertEqual(angles.shape[0], 4)
        hist = direction_histogram(angles, bins=4)
        self.assertTrue(math.isclose(hist.sum(), 1.0))
        e = entropy_of_distribution(hist)
        self.assertGreater(e, 0)
        ne = normalized_entropy(hist)
        self.assertTrue(0.0 <= ne <= 1.0)

    def test_instability_and_risk(self):
        s = instability_score(d=0.2, a=0.9, e=0.9)
        self.assertTrue(0.0 <= s <= 1.0)
        cat = risk_category(s)
        self.assertIn(cat, ["Green", "Yellow", "Red"])


if __name__ == "__main__":
    unittest.main()
