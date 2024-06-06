import numpy as np


def tss_search(current_frame, reference_frame, block_size=16, search_radius=8, similarity_metric='MAD'):
    """
    Three-Step Search Algorithm for Motion Estimation

    Input:
    - current_frame (np.array): The current frame
    - reference_frame (np.array): The reference frame
    - block_size (int): The size of the block
    - search_radius (int): The maximum search radius

    Returns:
    - np.array: A 3D numpy array containing the motion vectors for each block.
        The third dimension contains the y and x offsets.
    """
    # Check if frames are identical
    if np.array_equal(current_frame, reference_frame):
        return np.zeros((current_frame.shape[0] // block_size, current_frame.shape[1] // block_size, 2), dtype=int)

    height, width = current_frame.shape
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=int)

    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            # Initialize minimum distance and best offsets
            min_distance = float('inf')
            best_offset_y, best_offset_x = 0, 0

            # Calculate block position in the current frame
            start_y = block_y * block_size
            start_x = block_x * block_size
            block_current = current_frame[start_y:start_y + block_size, start_x:start_x + block_size]

            # Initial step size for the search
            step_size = search_radius // 2

            # Perform the search
            first_step = True
            while step_size >= 1:
                best_candidate_y, best_candidate_x = 0, 0  # Initialize candidates
                search_points = [(0, 0)] if first_step else []  # Include center point only in the first step

                for offset_y in range(-step_size, step_size + 1, step_size):
                    for offset_x in range(-step_size, step_size + 1, step_size):
                        if offset_y == 0 and offset_x == 0 and not first_step:
                            continue
                        search_points.append((offset_y, offset_x))

                for offset_y, offset_x in search_points:
                    # Calculate reference block position
                    ref_y = start_y + best_offset_y + offset_y
                    ref_x = start_x + best_offset_x + offset_x

                    # Check if the reference block is within frame bounds
                    if (0 <= ref_y < height - block_size + 1) and (0 <= ref_x < width - block_size + 1):
                        block_reference = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

                        # Calculate the distance between current block and reference block
                        if similarity_metric == 'MAD':
                            distance = np.mean(np.abs(block_current - block_reference))
                        elif similarity_metric == 'SSD':
                            distance = np.sum((block_current - block_reference) ** 2)
                        else:
                            raise ValueError("Invalid similarity metric. Use 'MAD' or 'SSD'.")

                        # Update minimum distance and best candidate offsets
                        if distance < min_distance:
                            min_distance = distance
                            best_candidate_y = offset_y
                            best_candidate_x = offset_x

                # Update the best offsets and reduce the step size by half (floor)
                best_offset_y += best_candidate_y
                best_offset_x += best_candidate_x
                step_size //= 2
                first_step = False

            # Store the best offsets as the motion vector for the current block
            motion_vectors[block_y, block_x] = [best_offset_y, best_offset_x]

    return motion_vectors
