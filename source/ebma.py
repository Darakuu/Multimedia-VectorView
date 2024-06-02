import numpy as np

def ebma_search(current_frame, reference_frame, block_size=16, search_radius=8):
    """
    Performs the Exhaustive Block Matching Algorithm (EBMA) to find motion vectors between two frames.

    Parameters:
    - current_frame (np.array): The current frame as a 2D numpy array.
    - reference_frame (np.array): The reference frame as a 2D numpy array.
    - block_size (int, optional): The size of the block. Default is 16.
    - search_radius (int, optional): The search radius. Default is 7.

    Returns:
    - np.array: A 3D numpy array containing the motion vectors for each block. The third dimension contains the y and x offsets.

    Raises:
    - ValueError: If the current_frame and reference_frame do not have the same shape.
    """

    # Ensure the current and reference frames are the same shape
    if current_frame.shape != reference_frame.shape:
        raise ValueError("The current frame and reference frame must have the same shape.")

    frame_height, frame_width = current_frame.shape
    num_blocks_y = frame_height // block_size
    num_blocks_x = frame_width // block_size

    # Initialize motion vectors array
    motion_vectors = np.zeros((num_blocks_y, num_blocks_x, 2), dtype=int)

    # Loop through each block in the current frame
    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            min_distance = float('inf')
            best_offset_y, best_offset_x = 0, 0

            # Calculate the starting coordinates of the block in the current frame
            block_start_y = block_y * block_size
            block_start_x = block_x * block_size

            # Extract the block from the current frame
            current_block = current_frame[block_start_y:block_start_y + block_size, block_start_x:block_start_x + block_size]

            # Define the search window around the block in the reference frame
            for offset_y in range(-search_radius, search_radius + 1):
                for offset_x in range(-search_radius, search_radius + 1):
                    ref_y = block_start_y + offset_y
                    ref_x = block_start_x + offset_x

                    # Check if the reference block is within frame boundaries
                    if 0 <= ref_y < frame_height - block_size + 1 and 0 <= ref_x < frame_width - block_size + 1:
                        reference_block = reference_frame[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                        distance = np.sum((current_block - reference_block) ** 2)
                        # ref_y: starting row, ref_x: starting column, block_size: height and width of the block

                        # Update the best offset if a smaller distance is found
                        if distance < min_distance:
                            min_distance = distance
                            best_offset_y = offset_y
                            best_offset_x = offset_x

            # Store the best offsets (motion vectors) for the current block
            motion_vectors[block_y, block_x] = [best_offset_y, best_offset_x]

    return motion_vectors
