import numpy as np


def ebma_search(imgP, imgI, blockSize=16, p=7):
    """
    Exhaustive Block Matching Algorithm

    Input:
    - imgP (np.array): The current frame
    - imgI (np.array): The reference frame
    - blockSize (int): The size of the block
    - p (int): The search parameter

    Returns:
    - np.array: The motion vectors
    """
    height, width = imgP.shape
    motion_vectors = np.zeros((height // blockSize, width // blockSize, 2), dtype=np.int)

    for m in range(blockSize // 2, height - blockSize // 2 + 1, blockSize):
        for n in range(blockSize // 2, width - blockSize // 2 + 1, blockSize):
            dist_min = float('inf')
            for k in range(max(blockSize // 2 - p, 0), min(height - blockSize // 2 + 1, blockSize // 2 + p + 1)):
                for l in range(max(blockSize // 2 - p, 0), min(width - blockSize // 2 + 1, blockSize // 2 + p + 1)):
                    dist = np.sum((imgP[m - blockSize // 2:m + blockSize // 2,
                                   n - blockSize // 2:n + blockSize // 2] - imgI[k - blockSize // 2:k + blockSize // 2,
                                                                            l - blockSize // 2:l + blockSize // 2]) ** 2)
                    if dist < dist_min:
                        dist_min = dist
                        dy = k - m
                        dx = l - n
            motion_vectors[m // blockSize, n // blockSize, 0] = dy
            motion_vectors[m // blockSize, n // blockSize, 1] = dx

    return motion_vectors
