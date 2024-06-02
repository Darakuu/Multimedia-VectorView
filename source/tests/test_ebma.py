import unittest
import numpy as np
from source.ebma import ebma_search


class TestEBMASearch(unittest.TestCase):

    def test_invalid_shapes(self):
        # Test when the current frame and reference frame have different shapes
        current_frame = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        reference_frame = np.random.randint(0, 256, (64, 32), dtype=np.uint8)
        with self.assertRaises(ValueError):
            ebma_search(current_frame, reference_frame)

    def test_no_motion(self):
        # Test with no motion
        frame = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        result = ebma_search(frame, frame, block_size=16, search_radius=8)
        expected = np.zeros((1, 1, 2), dtype=int)  # No motion between the same frames
        np.testing.assert_array_equal(result, expected)

if __name__ == '__main__':
    unittest.main()
