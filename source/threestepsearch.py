import numpy as np

def tss_search(current_frame, reference_frame, block_size=16, search_param=7):
    """
    Three-Step Search Algorithm

    Input:
    - current_frame (np.array): The current frame
    - reference_frame (np.array): The reference frame
    - block_size (int): The size of the block
    - search_param (int): The maximum search parameter

    Returns:
    - np.array: A 3D numpy array containing the motion vectors for each block. The third dimension contains the y and x offsets.
    """
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

            step_size = search_param

            # Perform the search in three steps
            while step_size > 0:
                for offset_y in range(-step_size, step_size + 1, step_size):
                    for offset_x in range(-step_size, step_size + 1, step_size):
                        ref_y = start_y + best_offset_y + offset_y
                        ref_x = start_x + best_offset_x + offset_x

                        if (0 <= ref_y < height - block_size + 1) and (0 <= ref_x < width - block_size + 1):
                            block_reference = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                            dist = np.sum((block_current - block_reference) ** 2)

                            if dist < min_dist:
                                min_dist = dist
                                best_candidate_y = offset_y
                                best_candidate_x = offset_x

                best_offset_y += best_candidate_y
                best_offset_x += best_candidate_x
                step_size //= 2

            motion_vectors[block_y, block_x] = [best_offset_y, best_offset_x]

    return motion_vectors
