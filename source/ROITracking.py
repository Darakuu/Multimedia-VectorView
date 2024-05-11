import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QApplication
import utils.utils_video as vutils

# Global variables to store the coordinates of the bounding box
point1 = None
point2 = None

def mouse_events(event, x, y, flags, param):
    global point1, point2, frame, is_drawing
    # If the left mouse button is clicked, record the starting coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        is_drawing = True
    # If the left mouse button is released, record the ending coordinates
    elif event == cv2.EVENT_LBUTTONUP:
        # Sort the points to ensure point1 is top left and point2 is bottom right
        point1, point2 = sorted([(x, y), point1])
        is_drawing = False
    # If the mouse is moved while the left button is down, draw the selection area
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, point1, (x, y), (255, 191, 0), 2)  # Light blue color
        cv2.imshow("Tracking Window", temp_frame)

def track_roi(video_path):
    global point1, point2, frame, is_drawing
    # Open the video file
    video = vutils.open_video(video_path)

    # Read the first frame
    ret, frame = video.read()

    # Create a window and set the mouse callback function
    cv2.namedWindow("Tracking Window")
    cv2.setMouseCallback("Tracking Window", mouse_events)

    # Initialize is_drawing
    is_drawing = False

    # Wait for the user to draw a bounding box
    while point1 is None or point2 is None:
        cv2.imshow("Tracking Window", frame)
        cv2.waitKey(1)

    # Create a CV2 tracker object
    tracker = cv2.TrackerMIL_create()

    # Initialize tracker with the bounding box drawn by the user, all directions allowed
    bbox = (point1[0], point1[1], point2[0] - point1[0], point2[1] - point1[1])
    ret = tracker.init(frame, bbox)

    app = QApplication([])
    label = QLabel()
    label.setMinimumSize(640, 480)  # Set a minimum resolution for the window

    while True:
        # Read a new frame
        ret, frame = video.read()
        if not ret:
            break

        # Update tracker
        ret, bbox = tracker.update(frame)

        # Draw bounding box
        if ret:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

        # Convert the frame to QImage and display it using PyQt
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap)
        label.show()

        # Break the loop if 'esc' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    track_roi('C:/Users/elvio/Downloads/ffmpeg/input2.mp4')