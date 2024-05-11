import cv2


def display_frame(video, frame_index, window_name='Frame'):
    """
    Display a specific frame from a video using OpenCV.

    Input:
    - video (cv2.VideoCapture): The video
    - frame_index (int): The index of the frame to display
    - window_name (str): The name of the window in which to display the frame
    """
    # Set the video's position to the specified frame index
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Read the frame at the current position
    ret, frame = video.read()

    # If the frame was read successfully, display it
    if ret:
        cv2.imshow(window_name, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error retrieving frame {frame_index}")