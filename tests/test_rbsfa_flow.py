import cv2
import numpy as np

from lib.test.tracker.fartrack_sparse_rbsfa import sparse_flow_displacement


def _textured_image():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    for y in range(18, 76, 8):
        for x in range(18, 76, 8):
            cv2.circle(image, (x, y), 2, (220, 120, 30), -1)
    return image


def test_sparse_flow_recovers_translation():
    previous = _textured_image()
    transform = np.float32([[1, 0, 3], [0, 1, -2]])
    current = cv2.warpAffine(previous, transform, (96, 96))
    result = sparse_flow_displacement(previous, current, [12, 12, 72, 72])
    assert result is not None
    displacement, inliers = result
    assert inliers >= 6
    assert np.allclose(displacement, [3, -2], atol=0.5)


def test_sparse_flow_rejects_blank_or_tiny_regions():
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    assert sparse_flow_displacement(blank, blank, [4, 4, 40, 40]) is None
    assert sparse_flow_displacement(_textured_image(), _textured_image(), [1, 1, 2, 2]) is None


def test_sparse_flow_rejects_mismatched_frames():
    assert sparse_flow_displacement(_textured_image(), np.zeros((80, 80, 3), dtype=np.uint8), [4, 4, 50, 50]) is None
