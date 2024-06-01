import numpy as np


def ebma_search(current_frame, reference_frame, block_size=16, search_param=7):
    """
    Performs the Exhaustive Block Matching Algorithm (EBMA) to find motion vectors between two frames.

    Input:
    - current_frame (np.array): The current frame as a 2D numpy array.
    - reference_frame (np.array): The reference frame as a 2D numpy array.
    - block_size (int, optional): The size of the block. Default is 16.
    - search_param (int, optional): The search parameter. Default is 7.

    Returns:
    - np.array: A 3D numpy array containing the motion vectors for each block. The third dimension contains the y and x offsets.

    Raises:
    - ValueError: If the current_frame and reference_frame do not have the same shape.
    """

    if current_frame.shape != reference_frame.shape:
        raise ValueError("current_frame and reference_frame must have the same shape.")

    height, width = current_frame.shape
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=int)

    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            min_dist = float('inf')
            best_offset_y, best_offset_x = 0, 0

            # Block position in current_frame
            start_y = block_y * block_size
            start_x = block_x * block_size
            block_current = current_frame[start_y:start_y + block_size, start_x:start_x + block_size]
            # Search window
            for offset_y in range(-search_param, search_param + 1):
                for offset_x in range(-search_param, search_param + 1):
                    ref_y = start_y + offset_y
                    ref_x = start_x + offset_x

                    if (0 <= ref_y < height - block_size + 1) and (0 <= ref_x < width - block_size + 1):
                        block_reference = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        dist = np.sum((block_current - block_reference) ** 2)

                        if dist < min_dist:
                            min_dist = dist
                            best_offset_y = offset_y
                            best_offset_x = offset_x

            motion_vectors[block_y, block_x] = [best_offset_y, best_offset_x]

    return motion_vectors