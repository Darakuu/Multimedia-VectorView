import sys
import unittest
import numpy as np
import logging
from source.threestepsearch import tss_search  # Assuming your function is saved in a file named tss_search.py

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))


class TestTSSSearch(unittest.TestCase):

    def setUp(self):
        self.block_size = 16
        self.search_radius = 8

    def test_identical_frames_return_zero_motion_vectors(self):
        frame = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        motion_vectors = tss_search(frame, frame, self.block_size, self.search_radius)
        self.assertTrue((motion_vectors == 0).all())

    def test_different_frames_return_non_zero_motion_vectors(self):
        frame1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        motion_vectors = tss_search(frame1, frame2, self.block_size, self.search_radius)
        self.assertFalse((motion_vectors == 0).all())

    def test_partially_different_frames_return_mixed_motion_vectors(self):
        frame1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        frame2 = frame1.copy()
        frame2[32:48, 32:48] = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        motion_vectors = tss_search(frame1, frame2, self.block_size, self.search_radius)
        self.assertTrue((motion_vectors[2, 2] != 0).any())
        self.assertTrue((motion_vectors[0, 0] == 0).all())

    def test_no_motion(self):
        logger.info("Starting test_no_motion")

        current_frame = np.array([[1, 2, 1, 2],
                                  [2, 1, 2, 1],
                                  [1, 2, 1, 2],
                                  [2, 1, 2, 1]])
        reference_frame = np.array([[1, 2, 1, 2],
                                    [2, 1, 2, 1],
                                    [1, 2, 1, 2],
                                    [2, 1, 2, 1]])
        block_size = 2
        search_radius = 1
        expected_motion_vectors = np.zeros((2, 2, 2), dtype=int)

        logger.debug("Current Frame:\n%s", current_frame)
        logger.debug("Reference Frame:\n%s", reference_frame)

        result = tss_search(current_frame, reference_frame, block_size, search_radius)

        logger.debug("Expected Motion Vectors:\n%s", expected_motion_vectors)
        logger.debug("Result Motion Vectors:\n%s", result)

        np.testing.assert_array_equal(result, expected_motion_vectors)
        logger.info("test_no_motion passed")

    def test_boundary_conditions(self):
        logger.info("Starting test_boundary_conditions")

        current_frame = np.array([[1, 2, 3, 4],
                                  [5, 6, 7, 8],
                                  [9, 10, 11, 12],
                                  [13, 14, 15, 16]])
        reference_frame = np.array([[1, 2, 3, 4],
                                    [5, 6, 7, 8],
                                    [9, 10, 11, 12],
                                    [13, 14, 15, 16]])
        block_size = 4
        search_radius = 2
        expected_motion_vectors = np.zeros((1, 1, 2), dtype=int)

        logger.debug("Current Frame:\n%s", current_frame)
        logger.debug("Reference Frame:\n%s", reference_frame)

        result = tss_search(current_frame, reference_frame, block_size, search_radius)

        logger.debug("Expected Motion Vectors:\n%s", expected_motion_vectors)
        logger.debug("Result Motion Vectors:\n%s", result)

        np.testing.assert_array_equal(result, expected_motion_vectors)
        logger.info("test_boundary_conditions passed")


if __name__ == '__main__':
    unittest.main()
