import cv2


def open_video(file_path):
    """
    Open a video file using OpenCV

    Input:
    - file_path (str): The path to the video file

    Returns:
    - cv2.VideoCapture: A cv2.VideoCapture object for the video
    """
    video = cv2.VideoCapture(file_path)
    if not video.isOpened():
        print(f"Error opening video file {file_path}")
        return None
    return video


def get_frame(video, index):
    """
    Get a specific frame from a video.

    Input:
    - video (cv2.VideoCapture): The video
    - index (int): The index of the frame to retrieve

    Returns:
    np.array: The frame as a numpy array, or None if the frame could not be retrieved
    """
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = video.read()
    if not ret:
        print(f"Error retrieving frame {index}")
        return None
    return frame


def RGB_to_grayscale(frame):
    """
    Convert an RGB frame to grayscale.

    Input:
    frame (np.array): The RGB frame

    Returns:
    np.array: The grayscale frame
    """
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayscale_frame


def channel_to_grayscale(frame, channel):
    """
    Convert a specific color channel of an RGB frame to grayscale and discard the rest.

    Input:
    - frame (np.array): The RGB frame
    - channel (str): The color channel to keep ('r', 'g', or 'b')

    Returns:
    - np.array: The grayscale frame
    """
    # Remember that CV2 uses BGR instead of RGB
    if channel == 'r' or channel == 'R':
        channel_index = 2
    elif channel == 'g' or channel == 'G':
        channel_index = 1
    elif channel == 'b' or channel == 'B':
        channel_index = 0
    else:
        print(f"Invalid channel {channel}")
        return None

    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayscale_frame[frame[:, :, (channel_index+1) % 3] > 0] = 0
    grayscale_frame[frame[:, :, (channel_index+2) % 3] > 0] = 0

    return grayscale_frame

